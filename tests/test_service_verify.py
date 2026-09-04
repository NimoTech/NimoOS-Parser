import asyncio
from unittest.mock import MagicMock

import pytest

from parser import repo_allowlist
from parser.db import init_db
from parser.repo_jobs import enqueue_job, list_jobs
from parser.repo_records import upsert_file_path, upsert_file_record
from parser.repo_state import get_verify_last, set_verify_last
from parser.service_verify import (
    RootDiff,
    VerifyRunner,
    apply_root_diff,
    diff_root,
    repair_interrupted_verify,
    run_verify,
)
from parser.wiki_client import WikiRootNotFound


@pytest.fixture
def conn(tmp_path):
    return init_db(tmp_path / "p.db")


def _seed(conn, fid, root_id, path, mtime_ms):
    upsert_file_record(conn, file_id=fid, sha256_full="s" + fid, size=1, mime="text/plain",
                       modalities_done={"text": "v1"}, parser_version="parser/0.3.0", indexed_at=1)
    upsert_file_path(conn, root_id=root_id, path=path, file_id=fid, mtime_ms=mtime_ms)


def test_diff_root_classifies_missing_stale_extra():
    wiki = {"/a.md": 1000, "/b.md": 5000, "/c.md": 9000}
    local = {"/b.md": 3000, "/c.md": 9500, "/d.md": 1}
    d = diff_root(wiki, local)
    assert d == RootDiff(missing=["/a.md"], stale=["/b.md"], extra=["/d.md"])


def test_apply_root_diff_enqueues_and_deletes_with_gate(conn):
    repo_allowlist.set_extension_enabled(conn, ".pdf", False)
    _seed(conn, "old", "r1", "/DATA/old.md", 1)
    enqueue_job(conn, root_id="r1", path="/DATA/queued.md", op="index", priority=100, now_ms=1)
    pipe = MagicMock()
    diff = RootDiff(missing=["/DATA/new.md", "/DATA/skip.pdf", "/DATA/queued.md"],
                    stale=["/DATA/old.md"], extra=["/DATA/gone.md"])

    out = apply_root_diff(conn, pipe, root_id="r1", diff=diff, now_ms=7)

    assert out == {"missing_enqueued": 1, "stale_enqueued": 1, "extra_deleted": 1, "missing_gated": 1}
    jobs = {(j["path"], j["op"], j["priority"]) for j in list_jobs(conn, status="pending", limit=10)}
    assert jobs == {("/DATA/queued.md", "index", 100), ("/DATA/new.md", "index", 500),
                    ("/DATA/old.md", "reindex", 500)}
    pipe.delete_path.assert_called_once_with(root_id="r1", path="/DATA/gone.md", now_ms=7)


class FakeWiki:
    def __init__(self, roots, files, fail_roots=False, disabled=(), always_not_found=False):
        self.roots, self.files, self.fail_roots = roots, files, fail_roots
        self.disabled = set(disabled)
        self.always_not_found = always_not_found
        self.file_calls = []
    async def list_roots(self):
        if self.fail_roots:
            raise RuntimeError("wiki down")
        return [{"id": r, "path": "/" + r, "enabled": r not in self.disabled}
                for r in self.roots]
    async def fetch_root_files(self, root_id, *, after="", limit=1000):
        self.file_calls.append(root_id)
        if self.always_not_found or root_id not in self.files:
            raise WikiRootNotFound(root_id)
        rows = [f for f in self.files[root_id] if f["path"] > after][:limit]
        return {"files": rows, "next_after": rows[-1]["path"] if len(rows) == limit else ""}


@pytest.mark.asyncio
async def test_run_verify_retires_unknown_roots_and_reports_counts(conn):
    _seed(conn, "a", "r1", "/r1/a.md", 1000)
    _seed(conn, "z", "zombie", "/zombie/z.md", 1)
    wiki = FakeWiki(roots=["r1"], files={"r1": [
        {"path": "/r1/a.md", "mtime_ms": 1000, "size": 1},
        {"path": "/r1/b.md", "mtime_ms": 2000, "size": 1},
    ]})
    qstore, pipe = MagicMock(), MagicMock()

    res = await run_verify(conn, qstore, wiki, pipe, root_ids=None, trigger="manual", now_ms=50)

    assert res["ok"] is True and res["trigger"] == "manual"
    assert res["finished_at"] >= res["started_at"] == 50, \
        "finished_at is measured when the run ends, not copied from started_at"
    assert res["retired_roots"] == ["zombie"]
    assert res["roots"] == [{"root_id": "r1", "wiki_files": 2, "local_files": 1,
                             "missing": 1, "stale": 0, "extra": 0}]
    qstore.tombstone_file.assert_called_once_with(file_id="z", tombstoned_at=50)
    assert get_verify_last(conn) == res


@pytest.mark.asyncio
async def test_run_verify_pages_the_wiki_feed(conn):
    files = [{"path": f"/r1/{i:03d}.md", "mtime_ms": 1, "size": 1} for i in range(5)]
    wiki = FakeWiki(roots=["r1"], files={"r1": files})
    import parser.service_verify as sv
    sv_page = sv.VERIFY_PAGE_SIZE
    sv.VERIFY_PAGE_SIZE = 2
    try:
        res = await run_verify(conn, MagicMock(), wiki, MagicMock(), root_ids=["r1"], trigger="manual", now_ms=1)
    finally:
        sv.VERIFY_PAGE_SIZE = sv_page
    assert res["roots"][0]["wiki_files"] == 5 and res["roots"][0]["missing"] == 5


@pytest.mark.asyncio
async def test_run_verify_does_not_retire_when_roots_listing_fails(conn):
    _seed(conn, "z", "zombie", "/zombie/z.md", 1)
    qstore = MagicMock()
    res = await run_verify(conn, qstore, FakeWiki(roots=[], files={}, fail_roots=True), MagicMock(),
                           root_ids=None, trigger="cursor_gap", now_ms=1)
    assert res["ok"] is False and "wiki down" in res["error"]
    qstore.tombstone_file.assert_not_called()
    assert res["retired_roots"] == []


@pytest.mark.asyncio
async def test_verify_runner_is_single_flight(conn):
    gate = asyncio.Event()

    class SlowWiki(FakeWiki):
        async def list_roots(self):
            await gate.wait()
            return await super().list_roots()

    runner = VerifyRunner(conn, MagicMock(), SlowWiki(roots=["r1"], files={"r1": []}),
                          MagicMock())
    assert runner.start(root_ids=None, trigger="manual") is True
    assert runner.running is True
    assert runner.start(root_ids=None, trigger="manual") is False, "second start while running is refused"
    gate.set()
    await runner.wait()
    assert runner.running is False
    assert get_verify_last(conn)["ok"] is True


@pytest.mark.asyncio
async def test_run_verify_refuses_to_act_when_wiki_lists_no_roots(conn):
    # THE data-safety gate: local_roots - {} == every root we hold, so the
    # retire loop would tombstone the entire ledger while reporting ok.
    _seed(conn, "a", "r1", "/r1/a.md", 1000)
    _seed(conn, "b", "r2", "/r2/b.md", 1000)
    qstore, pipe = MagicMock(), MagicMock()

    res = await run_verify(conn, qstore, FakeWiki(roots=[], files={}), pipe,
                           root_ids=None, trigger="manual", now_ms=50)

    assert res["ok"] is False and res["error"] == "wiki returned no roots"
    assert res["retired_roots"] == [] and res["roots"] == []
    assert res["finished_at"] is not None
    qstore.tombstone_file.assert_not_called()
    qstore.set_root_ids_for_file.assert_not_called()
    pipe.delete_path.assert_not_called()
    assert conn.execute("SELECT COUNT(*) FROM file_paths").fetchone()[0] == 2
    assert get_verify_last(conn) == res


@pytest.mark.asyncio
async def test_run_verify_reports_unavailable_files_endpoint_on_old_wiki(conn):
    # A Wiki without /_internal/files 404s every root: every root is skipped
    # and the old code still claimed ok. Spec §7 wants an explicit failure.
    _seed(conn, "a", "r1", "/r1/a.md", 1000)
    wiki = FakeWiki(roots=["r1", "r2"], files={}, always_not_found=True)
    pipe = MagicMock()

    res = await run_verify(conn, MagicMock(), wiki, pipe, root_ids=None,
                           trigger="manual", now_ms=50)

    assert res["ok"] is False and res["error"] == "wiki files endpoint unavailable"
    assert res["roots"] == [] and res["retired_roots"] == []
    pipe.delete_path.assert_not_called()


@pytest.mark.asyncio
async def test_run_verify_records_requested_roots_wiki_does_not_know(conn):
    wiki = FakeWiki(roots=["r1"], files={"r1": []})
    res = await run_verify(conn, MagicMock(), wiki, MagicMock(),
                           root_ids=["ghost"], trigger="manual", now_ms=50)
    assert res["unknown_roots"] == ["ghost"]
    assert res["roots"] == [] and res["ok"] is True
    assert wiki.file_calls == [], "a root Wiki never listed is not fetched"


@pytest.mark.asyncio
async def test_run_verify_skips_disabled_roots_without_retiring(conn):
    # Wiki disables a root when its disk disappears but keeps the file_index:
    # the root is still present (never retire it) and its files are
    # unreachable (never enqueue index/reindex for them).
    _seed(conn, "a", "gone", "/gone/a.md", 1000)
    wiki = FakeWiki(roots=["gone"], files={"gone": [
        {"path": "/gone/a.md", "mtime_ms": 9999, "size": 1},
        {"path": "/gone/b.md", "mtime_ms": 1, "size": 1},
    ]}, disabled=["gone"])
    qstore, pipe = MagicMock(), MagicMock()

    res = await run_verify(conn, qstore, wiki, pipe, root_ids=None,
                           trigger="manual", now_ms=50)

    assert res["ok"] is True and res["retired_roots"] == []
    assert res["roots"] == [{"root_id": "gone", "skipped": "disabled"}]
    assert wiki.file_calls == []
    assert list_jobs(conn, status="pending", limit=10) == []
    qstore.tombstone_file.assert_not_called()
    pipe.delete_path.assert_not_called()


@pytest.mark.asyncio
async def test_run_verify_skips_extra_deletes_when_wiki_reports_zero_files(conn):
    # "Wiki lists the root but zero files under it" is as likely a broken
    # file_index as a genuinely empty root; deleting our whole side of it
    # would be unrecoverable. Report the count instead of acting.
    _seed(conn, "a", "r1", "/r1/a.md", 1000)
    _seed(conn, "b", "r1", "/r1/b.md", 1000)
    pipe = MagicMock()

    res = await run_verify(conn, MagicMock(), FakeWiki(roots=["r1"], files={"r1": []}),
                           pipe, root_ids=None, trigger="manual", now_ms=50)

    assert res["ok"] is True
    assert res["roots"] == [{"root_id": "r1", "wiki_files": 0, "local_files": 2,
                             "missing": 0, "stale": 0, "extra": 2, "extra_skipped": 2}]
    pipe.delete_path.assert_not_called()
    assert conn.execute("SELECT COUNT(*) FROM file_paths WHERE root_id='r1'").fetchone()[0] == 2


def test_apply_root_diff_can_withhold_extra_deletes(conn):
    pipe = MagicMock()
    diff = RootDiff(extra=["/DATA/gone.md"])
    out = apply_root_diff(conn, pipe, root_id="r1", diff=diff, now_ms=7,
                          delete_extra=False)
    assert out["extra_deleted"] == 0
    pipe.delete_path.assert_not_called()


@pytest.mark.asyncio
async def test_wiki_files_stops_when_paging_does_not_advance(conn):
    # A Wiki that echoes the same next_after would spin forever.
    class StuckWiki(FakeWiki):
        async def fetch_root_files(self, root_id, *, after="", limit=1000):
            self.file_calls.append(after)
            if len(self.file_calls) > 10:
                raise AssertionError("paging did not terminate")
            return {"files": [{"path": "/r1/a.md", "mtime_ms": 1, "size": 1}],
                    "next_after": "/r1/a.md"}

    wiki = StuckWiki(roots=["r1"], files={"r1": []})
    res = await run_verify(conn, MagicMock(), wiki, MagicMock(), root_ids=None,
                           trigger="manual", now_ms=50)
    assert res["ok"] is True
    assert wiki.file_calls == ["", "/r1/a.md"], "one repeat is enough to detect the stall"
    assert res["roots"][0]["wiki_files"] == 1


def test_repair_interrupted_verify_lands_a_terminal_record(conn):
    set_verify_last(conn, {"trigger": "manual", "started_at": 1, "finished_at": None,
                           "ok": False, "roots": [], "retired_roots": [], "error": None})
    assert repair_interrupted_verify(conn, 4242) is True
    last = get_verify_last(conn)
    assert last["finished_at"] == 4242 and last["ok"] is False
    assert last["error"] == "interrupted by restart"
    # Idempotent: a finished record is left alone.
    assert repair_interrupted_verify(conn, 9999) is False
    assert get_verify_last(conn)["finished_at"] == 4242


def test_repair_interrupted_verify_without_a_record(conn):
    assert repair_interrupted_verify(conn, 1) is False


@pytest.mark.asyncio
async def test_verify_runner_stop_cancels_and_lands_a_terminal_record(conn):
    gate = asyncio.Event()

    class HangingWiki(FakeWiki):
        async def list_roots(self):
            await gate.wait()
            return await super().list_roots()

    runner = VerifyRunner(conn, MagicMock(), HangingWiki(roots=["r1"], files={"r1": []}),
                          MagicMock())
    assert runner.start(root_ids=None, trigger="manual") is True
    await asyncio.sleep(0)  # let run_verify persist the in-progress record
    assert get_verify_last(conn)["finished_at"] is None

    await runner.stop()

    assert runner.running is False
    last = get_verify_last(conn)
    assert last["finished_at"] is not None and last["ok"] is False
    assert last["error"] == "cancelled"


@pytest.mark.asyncio
async def test_verify_runner_stop_is_a_noop_when_idle(conn):
    runner = VerifyRunner(conn, MagicMock(), FakeWiki(roots=["r1"], files={"r1": []}),
                          MagicMock())
    await runner.stop()
    assert get_verify_last(conn) is None
