import sqlite3
from unittest.mock import MagicMock

import pytest

from parser.db import init_db
from parser.repo_jobs import enqueue_job, list_jobs
from parser.repo_records import upsert_file_path, upsert_file_record
from parser.service_retire import retire_root


@pytest.fixture
def conn(tmp_path):
    return init_db(tmp_path / "p.db")


def _seed(conn, fid, paths):
    upsert_file_record(conn, file_id=fid, sha256_full="s" + fid, size=1, mime="text/plain",
                       modalities_done={"text": "v1"}, parser_version="parser/0.3.0", indexed_at=1)
    for root_id, path in paths:
        upsert_file_path(conn, root_id=root_id, path=path, file_id=fid, mtime_ms=1)


def test_retire_root_tombstones_exclusive_files_and_rehomes_shared(conn):
    _seed(conn, "only", [("r1", "/mnt/a.md")])
    _seed(conn, "shared", [("r1", "/mnt/b.md"), ("r2", "/DATA/b.md")])
    _seed(conn, "other", [("r2", "/DATA/c.md")])
    enqueue_job(conn, root_id="r1", path="/mnt/a.md", op="reindex", priority=2000, now_ms=1)
    enqueue_job(conn, root_id="r2", path="/DATA/c.md", op="index", priority=100, now_ms=1)
    qstore = MagicMock()

    out = retire_root(conn, qstore, root_id="r1", now_ms=999)

    assert out == {"root_id": "r1", "files_seen": 2, "tombstoned": 1, "rehomed": 1, "jobs_dropped": 1}
    qstore.tombstone_file.assert_called_once_with(file_id="only", tombstoned_at=999)
    qstore.set_root_ids_for_file.assert_called_once_with(file_id="shared", root_ids=["r2"])
    assert conn.execute("SELECT COUNT(*) FROM file_paths WHERE root_id='r1'").fetchone()[0] == 0
    assert conn.execute("SELECT tombstoned_at FROM file_records WHERE file_id='only'").fetchone()[0] == 999
    assert conn.execute("SELECT tombstoned_at FROM file_records WHERE file_id='shared'").fetchone()[0] is None
    remaining = list_jobs(conn, status="pending", limit=10)
    assert [j["path"] for j in remaining] == ["/DATA/c.md"], "only the retired root's jobs are dropped"


def test_retire_root_is_a_noop_for_unknown_root(conn):
    qstore = MagicMock()
    out = retire_root(conn, qstore, root_id="ghost", now_ms=1)
    assert out["files_seen"] == 0 and out["tombstoned"] == 0
    qstore.tombstone_file.assert_not_called()


def test_retire_root_requires_qdrant(conn):
    _seed(conn, "only", [("r1", "/mnt/a.md")])
    with pytest.raises(RuntimeError):
        retire_root(conn, None, root_id="r1", now_ms=1)
    assert conn.execute("SELECT COUNT(*) FROM file_paths WHERE root_id='r1'").fetchone()[0] == 1, \
        "nothing is removed from SQLite when Qdrant can't be updated"


def test_retire_root_partial_qstore_failure_leaves_unprocessed_files_retryable(conn):
    _seed(conn, "f1", [("r1", "/mnt/f1.md")])
    _seed(conn, "f2", [("r1", "/mnt/f2.md")])
    _seed(conn, "f3", [("r1", "/mnt/f3.md")])
    qstore = MagicMock()
    qstore.tombstone_file.side_effect = [None, RuntimeError("qdrant down"), None]

    with pytest.raises(RuntimeError):
        retire_root(conn, qstore, root_id="r1", now_ms=999)

    # f1 (processed first, per ORDER BY file_id) is fully retired.
    assert conn.execute(
        "SELECT COUNT(*) FROM file_paths WHERE root_id='r1' AND file_id='f1'"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT tombstoned_at FROM file_records WHERE file_id='f1'"
    ).fetchone()[0] == 999
    # f2 (the one whose qstore call raised) and f3 (never reached) are
    # untouched in SQLite, so a retry will redo exactly them.
    for fid in ("f2", "f3"):
        assert conn.execute(
            "SELECT COUNT(*) FROM file_paths WHERE root_id='r1' AND file_id=?", (fid,)
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT tombstoned_at FROM file_records WHERE file_id=?", (fid,)
        ).fetchone()[0] is None

    qstore.tombstone_file.side_effect = None
    out = retire_root(conn, qstore, root_id="r1", now_ms=1000)

    assert out["files_seen"] == 2
    assert out["tombstoned"] == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM file_paths WHERE root_id='r1'"
    ).fetchone()[0] == 0


def test_retire_root_keeps_its_own_in_flight_job(conn):
    _seed(conn, "only", [("r1", "/mnt/a.md")])
    enqueue_job(conn, root_id="r1", path="", op="retire_root", priority=0, now_ms=1)
    enqueue_job(conn, root_id="r1", path="/mnt/a.md", op="index", priority=100, now_ms=1)
    qstore = MagicMock()

    out = retire_root(conn, qstore, root_id="r1", now_ms=999)

    assert out["jobs_dropped"] == 1
    remaining = list_jobs(conn, status="pending", limit=10)
    assert [j["op"] for j in remaining] == ["retire_root"], \
        "the in-flight retire_root job itself must survive; the worker completes it"


class _FailingConn:
    """Passes everything through to the real connection but raises on the
    statement whose SQL contains `marker`, once."""

    def __init__(self, conn, marker):
        self.conn = conn
        self.marker = marker
        self.armed = True

    def __getattr__(self, name):
        return getattr(self.conn, name)

    def execute(self, sql, *args, **kw):
        if self.armed and self.marker in sql:
            self.armed = False
            raise sqlite3.OperationalError("disk I/O error")
        return self.conn.execute(sql, *args, **kw)


def test_retire_root_tombstone_survives_a_failing_path_delete(conn):
    # Statement order matters: with DELETE FROM file_paths first, a failure of
    # the following set_tombstone left the file with no path AND no
    # tombstoned_at — invisible to a retry (no path to find it by) and to gc
    # (no tombstone to collect). The tombstone must land first.
    _seed(conn, "only", [("r1", "/mnt/a.md")])
    qstore = MagicMock()
    guarded = _FailingConn(conn, "DELETE FROM file_paths")

    with pytest.raises(sqlite3.OperationalError):
        retire_root(guarded, qstore, root_id="r1", now_ms=999)

    assert conn.execute(
        "SELECT tombstoned_at FROM file_records WHERE file_id='only'"
    ).fetchone()[0] == 999, "the tombstone is written before the path is dropped"
    assert conn.execute(
        "SELECT COUNT(*) FROM file_paths WHERE root_id='r1'"
    ).fetchone()[0] == 1, "the path is still there, so a retry can finish the job"

    out = retire_root(conn, qstore, root_id="r1", now_ms=1000)

    assert out["files_seen"] == 1 and out["tombstoned"] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM file_paths WHERE root_id='r1'"
    ).fetchone()[0] == 0
