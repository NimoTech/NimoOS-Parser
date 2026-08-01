import asyncio
import sqlite3
import time

import pytest

from parser.db import init_db
from parser.repo_jobs import enqueue_job, list_jobs
from parser.workers import WorkerPool


class FakePipeline:
    def __init__(self):
        self.calls = []
    def index_file(self, *, root_id, path, now_ms):
        self.calls.append((root_id, path))


class FakeDelPipeline:
    def __init__(self):
        self.deletes = []
    def delete_path(self, *, root_id, path, now_ms):
        self.deletes.append((root_id, path))


@pytest.fixture
def conn(tmp_path):
    return init_db(tmp_path / "p.db")


def test_worker_picks_and_processes_job(conn):
    enqueue_job(conn, root_id="r", path="/a.md", op="index",
                priority=100, now_ms=100)
    pipe = FakePipeline()
    pool = WorkerPool(conn, text_pipeline=pipe, concurrency=1,
                      lease_s=10)

    async def runner():
        await pool.start()
        # let one job get picked
        for _ in range(50):
            if pipe.calls:
                break
            await asyncio.sleep(0.02)
        await pool.stop()

    asyncio.run(runner())
    assert pipe.calls == [("r", "/a.md")]
    done = list_jobs(conn, status="failed", limit=10)
    assert done == []


def test_worker_handles_pipeline_exception(conn):
    enqueue_job(conn, root_id="r", path="/bad.md", op="index",
                priority=100, now_ms=100)

    class Boom:
        def index_file(self, **kw):
            raise RuntimeError("boom")

    pool = WorkerPool(conn, text_pipeline=Boom(), concurrency=1,
                      lease_s=10, max_attempts=1)

    async def runner():
        await pool.start()
        for _ in range(50):
            failed = list_jobs(conn, status="failed", limit=10)
            if failed:
                break
            await asyncio.sleep(0.05)
        await pool.stop()

    asyncio.run(runner())
    failed = list_jobs(conn, status="failed", limit=10)
    assert len(failed) == 1
    assert "boom" in failed[0]["last_error"]


class FakeWiki:
    def __init__(self):
        self.calls = []
    async def report_index_status(self, *, path, status, parser_version,
                                  modalities=None, error=None):
        self.calls.append({
            "path": path, "status": status, "parser_version": parser_version,
            "modalities": modalities, "error": error,
        })


def test_worker_reports_indexed_to_wiki(conn):
    enqueue_job(conn, root_id="r", path="/a.md", op="index",
                priority=100, now_ms=100)
    wiki = FakeWiki()
    pool = WorkerPool(conn, text_pipeline=FakePipeline(), concurrency=1,
                      lease_s=10, wiki_client=wiki,
                      parser_version="parser/0.1.0")

    async def runner():
        await pool.start()
        for _ in range(50):
            if wiki.calls:
                break
            await asyncio.sleep(0.02)
        await pool.stop()

    asyncio.run(runner())
    assert len(wiki.calls) == 1
    assert wiki.calls[0]["status"] == "indexed"
    assert wiki.calls[0]["path"] == "/a.md"
    assert wiki.calls[0]["parser_version"] == "parser/0.1.0"
    assert wiki.calls[0]["error"] is None


def test_worker_reports_failed_to_wiki(conn):
    enqueue_job(conn, root_id="r", path="/bad.md", op="index",
                priority=100, now_ms=100)

    class Boom:
        def index_file(self, **kw):
            raise RuntimeError("boom")

    wiki = FakeWiki()
    pool = WorkerPool(conn, text_pipeline=Boom(), concurrency=1,
                      lease_s=10, max_attempts=1, wiki_client=wiki,
                      parser_version="parser/0.1.0")

    async def runner():
        await pool.start()
        for _ in range(50):
            if wiki.calls:
                break
            await asyncio.sleep(0.05)
        await pool.stop()

    asyncio.run(runner())
    assert any(c["status"] == "failed" and "boom" in c["error"]
               for c in wiki.calls)


def test_worker_reports_deleted_to_wiki(conn):
    enqueue_job(conn, root_id="r", path="/x.md", op="delete",
                priority=100, now_ms=100)
    wiki = FakeWiki()
    pool = WorkerPool(conn, text_pipeline=FakeDelPipeline(), concurrency=1,
                      lease_s=10, wiki_client=wiki,
                      parser_version="parser/0.1.0")

    async def runner():
        await pool.start()
        for _ in range(50):
            if wiki.calls:
                break
            await asyncio.sleep(0.02)
        await pool.stop()

    asyncio.run(runner())
    assert any(c["status"] == "deleted" for c in wiki.calls)


def test_worker_without_wiki_client_does_not_crash(conn):
    enqueue_job(conn, root_id="r", path="/a.md", op="index",
                priority=100, now_ms=100)
    pool = WorkerPool(conn, text_pipeline=FakePipeline(), concurrency=1,
                      lease_s=10, wiki_client=None)

    async def runner():
        await pool.start()
        for _ in range(50):
            done = conn.execute(
                "SELECT COUNT(*) FROM parse_jobs WHERE done_at IS NOT NULL"
            ).fetchone()[0]
            if done:
                break
            await asyncio.sleep(0.02)
        await pool.stop()

    asyncio.run(runner())
    done = conn.execute(
        "SELECT COUNT(*) FROM parse_jobs WHERE done_at IS NOT NULL"
    ).fetchone()[0]
    assert done == 1


def test_worker_writes_last_error_to_file_records_on_failure(tmp_path):
    """Pipeline raises -> worker fail_job -> file_records.last_error is written synchronously."""
    from parser.db import init_db
    from parser.repo_records import upsert_file_record, upsert_file_path

    conn = init_db(tmp_path / "p.db")
    upsert_file_record(
        conn, file_id="fid1", sha256_full="abc", size=10, mime="text/plain",
        modalities_done={}, parser_version="parser/0.2.0", indexed_at=100,
    )
    upsert_file_path(
        conn, root_id="r1", path="/p1", file_id="fid1", mtime_ms=0,
    )

    class FailingPipeline:
        def index_file(self, *, root_id, path, now_ms):
            raise RuntimeError("docling exploded")

    enqueue_job(conn, root_id="r1", path="/p1", op="index",
                priority=100, now_ms=100)

    pool = WorkerPool(conn, text_pipeline=FailingPipeline(),
                      concurrency=1, lease_s=10, max_attempts=1)

    async def runner():
        await pool.start()
        for _ in range(200):
            if conn.execute(
                "SELECT COUNT(*) FROM parse_jobs WHERE done_at IS NULL"
            ).fetchone()[0] == 0:
                break
            await asyncio.sleep(0.02)
        await pool.stop()

    asyncio.run(runner())

    row = conn.execute(
        "SELECT last_error FROM file_records WHERE file_id='fid1'"
    ).fetchone()
    assert row["last_error"] == "docling exploded"


def test_worker_clears_last_error_on_success(tmp_path):
    """The success path overwrites it back to NULL."""
    from parser.db import init_db
    from parser.repo_records import (
        upsert_file_record, upsert_file_path, set_last_error,
    )

    conn = init_db(tmp_path / "p.db")
    upsert_file_record(
        conn, file_id="fid1", sha256_full="abc", size=10, mime="text/plain",
        modalities_done={}, parser_version="parser/0.2.0", indexed_at=100,
    )
    upsert_file_path(
        conn, root_id="r1", path="/p1", file_id="fid1", mtime_ms=0,
    )
    set_last_error(conn, root_id="r1", path="/p1", error="prev")  # dirty it first

    class OkPipeline:
        def index_file(self, *, root_id, path, now_ms):
            return None

    enqueue_job(conn, root_id="r1", path="/p1", op="index",
                priority=100, now_ms=100)

    pool = WorkerPool(conn, text_pipeline=OkPipeline(),
                      concurrency=1, lease_s=10)

    async def runner():
        await pool.start()
        for _ in range(200):
            if conn.execute(
                "SELECT COUNT(*) FROM parse_jobs WHERE done_at IS NULL"
            ).fetchone()[0] == 0:
                break
            await asyncio.sleep(0.02)
        await pool.stop()

    asyncio.run(runner())

    row = conn.execute(
        "SELECT last_error FROM file_records WHERE file_id='fid1'"
    ).fetchone()
    assert row["last_error"] is None


@pytest.mark.asyncio
async def test_pacing_sleep_interruptible_by_exit_flag():
    """The power-saving tier's pacing sleep (up to 60s) must be interruptible
    by a set_concurrency scale-down, otherwise a trimmed worker wouldn't
    exit for up to 60s (M2 final review, Medium)."""
    from parser.workers import WorkerPool

    pool = WorkerPool.__new__(WorkerPool)  # only testing the helper, skips full construction
    pool._stop = asyncio.Event()
    flag = asyncio.Event()

    async def trip():
        await asyncio.sleep(0.05)
        flag.set()

    t = asyncio.get_event_loop().time()
    await asyncio.gather(pool._interruptible_sleep(30.0, flag), trip())
    assert asyncio.get_event_loop().time() - t < 5.0
