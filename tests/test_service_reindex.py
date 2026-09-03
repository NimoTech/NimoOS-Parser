"""Tests for parser.service_reindex.reindex_files.

The reindex transaction is the heart of the force-rebuild feature. These tests
hit the function directly (no HTTP) to verify both modes and edge cases.
"""
from pathlib import Path

import pytest

from parser.db import init_db
from parser.repo_jobs import enqueue_job
from parser.repo_records import (
    upsert_file_record, upsert_file_path, set_tombstone,
)


class FakeQstore:
    def __init__(self, raise_on=None):
        self.tombstoned = []
        self.raise_on = raise_on or set()
    def tombstone_file(self, *, file_id, tombstoned_at):
        if file_id in self.raise_on:
            raise RuntimeError("qdrant unreachable")
        self.tombstoned.append((file_id, tombstoned_at))


@pytest.fixture
def conn(tmp_path):
    return init_db(tmp_path / "p.db")


def _seed_file(conn, *, fid, root="r1", path=None,
               mime="text/plain", tombstoned=False):
    if path is None:
        path = f"/p/{fid}"
    upsert_file_record(
        conn, file_id=fid, sha256_full="sha-" + fid, size=10, mime=mime,
        modalities_done={}, parser_version="parser/0.2.0", indexed_at=100,
    )
    upsert_file_path(conn, root_id=root, path=path, file_id=fid, mtime_ms=0)
    if tombstoned:
        set_tombstone(conn, file_id=fid, at_ms=200)


def test_reindex_tombstones_and_enqueues_per_path(conn):
    from parser.service_reindex import reindex_files
    _seed_file(conn, fid="a")
    _seed_file(conn, fid="b")
    qs = FakeQstore()
    res = reindex_files(
        conn, qstore=qs, file_ids=["a", "b"], filter=None,
        reason="test", now_ms=500,
    )
    assert res["queued"] == 2
    assert res["tombstoned"] == 2
    assert len(res["job_ids"]) == 2
    assert res["skipped"] == []
    # SQLite tombstone
    rows = conn.execute(
        "SELECT file_id, tombstoned_at FROM file_records"
    ).fetchall()
    assert all(r["tombstoned_at"] == 500 for r in rows)
    # parse_jobs
    jobs = conn.execute(
        "SELECT * FROM parse_jobs WHERE op = 'reindex' ORDER BY id"
    ).fetchall()
    assert len(jobs) == 2
    assert all(j["priority"] == 1000 for j in jobs)
    # Qdrant called best-effort
    assert {t[0] for t in qs.tombstoned} == {"a", "b"}


def test_reindex_multi_root_file_enqueues_one_job_per_path(conn):
    from parser.service_reindex import reindex_files
    _seed_file(conn, fid="a", root="r1", path="/p1")
    # add a second path for the same file_id, different root
    upsert_file_path(conn, root_id="r2", path="/p2", file_id="a", mtime_ms=0)
    res = reindex_files(
        conn, qstore=FakeQstore(), file_ids=["a"], filter=None,
        reason=None, now_ms=500,
    )
    assert res["queued"] == 2  # one job per (root, path)
    assert res["tombstoned"] == 1
    assert len(res["job_ids"]) == 2


def test_reindex_skips_nonexistent(conn):
    from parser.service_reindex import reindex_files
    res = reindex_files(
        conn, qstore=FakeQstore(), file_ids=["ghost"], filter=None,
        reason=None, now_ms=500,
    )
    assert res["skipped"] == [{"file_id": "ghost", "reason": "not_found"}]
    assert res["queued"] == 0


def test_reindex_skips_already_tombstoned(conn):
    from parser.service_reindex import reindex_files
    _seed_file(conn, fid="dead", tombstoned=True)
    res = reindex_files(
        conn, qstore=FakeQstore(), file_ids=["dead"], filter=None,
        reason=None, now_ms=500,
    )
    assert res["skipped"] == [
        {"file_id": "dead", "reason": "already_tombstoned"}
    ]


def test_reindex_skips_orphan_without_paths(conn):
    """file_record exists, no file_paths rows — must skip + warn, NOT tombstone."""
    from parser.service_reindex import reindex_files
    upsert_file_record(
        conn, file_id="orphan", sha256_full="sha", size=0, mime="text/plain",
        modalities_done={}, parser_version="parser/0.2.0", indexed_at=100,
    )
    # NOTE: no upsert_file_path
    res = reindex_files(
        conn, qstore=FakeQstore(), file_ids=["orphan"], filter=None,
        reason=None, now_ms=500,
    )
    assert res["skipped"] == [
        {"file_id": "orphan", "reason": "no_paths_orphan"}
    ]
    row = conn.execute(
        "SELECT tombstoned_at FROM file_records WHERE file_id='orphan'"
    ).fetchone()
    assert row["tombstoned_at"] is None, "must not tombstone orphans"


def test_reindex_rejects_oversize_file_ids(conn):
    from parser.service_reindex import reindex_files, MAX_REINDEX_FILE_IDS
    ids = [f"f{i}" for i in range(MAX_REINDEX_FILE_IDS + 1)]
    with pytest.raises(ValueError, match="too many file_ids"):
        reindex_files(conn, qstore=FakeQstore(), file_ids=ids,
                      filter=None, reason=None, now_ms=500)


def test_reindex_qdrant_failure_does_not_rollback_sqlite(conn):
    from parser.service_reindex import reindex_files
    _seed_file(conn, fid="a")
    qs = FakeQstore(raise_on={"a"})
    res = reindex_files(
        conn, qstore=qs, file_ids=["a"], filter=None,
        reason=None, now_ms=500,
    )
    assert res["tombstoned"] == 1
    row = conn.execute(
        "SELECT tombstoned_at FROM file_records WHERE file_id='a'"
    ).fetchone()
    assert row["tombstoned_at"] == 500


def test_reindex_by_filter_root_id_matches_subset(conn):
    from parser.service_reindex import reindex_files
    _seed_file(conn, fid="a", root="r1")
    _seed_file(conn, fid="b", root="r1")
    _seed_file(conn, fid="c", root="r2")
    res = reindex_files(
        conn, qstore=FakeQstore(), file_ids=None,
        filter={"root_id": "r1"}, reason=None, now_ms=500,
    )
    assert res["tombstoned"] == 2
    assert res["queued"] == 2


def test_reindex_by_filter_rejects_over_max(conn):
    from parser.service_reindex import reindex_files
    # Patch the cap so we don't have to seed 10001 rows.
    import parser.service_reindex as sr
    original = sr.MAX_REINDEX_BY_FILTER
    sr.MAX_REINDEX_BY_FILTER = 2
    try:
        _seed_file(conn, fid="a")
        _seed_file(conn, fid="b")
        _seed_file(conn, fid="c")
        with pytest.raises(ValueError, match="filter matches"):
            reindex_files(
                conn, qstore=FakeQstore(), file_ids=None,
                filter={"root_id": "r1"}, reason=None, now_ms=500,
            )
    finally:
        sr.MAX_REINDEX_BY_FILTER = original


def test_reindex_rejects_both_file_ids_and_filter(conn):
    from parser.service_reindex import reindex_files
    with pytest.raises(ValueError, match="exactly one"):
        reindex_files(
            conn, qstore=FakeQstore(), file_ids=["a"], filter={"root_id": "r1"},
            reason=None, now_ms=500,
        )


def test_reindex_rejects_neither_file_ids_nor_filter(conn):
    from parser.service_reindex import reindex_files
    with pytest.raises(ValueError, match="exactly one"):
        reindex_files(
            conn, qstore=FakeQstore(), file_ids=None, filter=None,
            reason=None, now_ms=500,
        )


# --- parser_version drift → automatic incremental re-index (audit 2026-09-03) ---

def _seed_versioned(conn, fid, path, version, root="r1"):
    from parser.repo_records import upsert_file_record, upsert_file_path
    upsert_file_record(conn, file_id=fid, sha256_full="s" + fid, size=1, mime="text/markdown",
                       modalities_done={"text": "bge-m3/v1"}, parser_version=version, indexed_at=1)
    upsert_file_path(conn, root_id=root, path=path, file_id=fid, mtime_ms=1)


def test_enqueue_version_drift_reindexes_only_stale_files(tmp_path):
    from parser.db import init_db
    from parser.repo_jobs import list_jobs
    from parser.service_reindex import enqueue_version_drift
    conn = init_db(tmp_path / "p.db")
    _seed_versioned(conn, "old1", "/DATA/a.md", "parser/0.2.0")
    _seed_versioned(conn, "old2", "/DATA/b.md", "parser/0.2.0")
    _seed_versioned(conn, "cur", "/DATA/c.md", "parser/0.3.0")
    n = enqueue_version_drift(conn, parser_version="parser/0.3.0", now_ms=1000)
    assert n == 2
    pending = list_jobs(conn, status="pending", limit=10)
    assert sorted(j["path"] for j in pending) == ["/DATA/a.md", "/DATA/b.md"]
    assert all(j["op"] == "reindex" for j in pending)


def test_enqueue_version_drift_is_idempotent(tmp_path):
    from parser.db import init_db
    from parser.repo_jobs import list_jobs
    from parser.service_reindex import enqueue_version_drift
    conn = init_db(tmp_path / "p.db")
    _seed_versioned(conn, "old1", "/DATA/a.md", "parser/0.2.0")
    assert enqueue_version_drift(conn, parser_version="parser/0.3.0", now_ms=1000) == 1
    assert enqueue_version_drift(conn, parser_version="parser/0.3.0", now_ms=2000) == 0, \
        "a restart must not queue a second job while the first is still open"
    assert len(list_jobs(conn, status="pending", limit=10)) == 1


def test_enqueue_version_drift_skips_tombstoned(tmp_path):
    from parser.db import init_db
    from parser.repo_records import set_tombstone
    from parser.service_reindex import enqueue_version_drift
    conn = init_db(tmp_path / "p.db")
    _seed_versioned(conn, "old1", "/DATA/a.md", "parser/0.2.0")
    set_tombstone(conn, file_id="old1", at_ms=5)
    assert enqueue_version_drift(conn, parser_version="parser/0.3.0", now_ms=1000) == 0
