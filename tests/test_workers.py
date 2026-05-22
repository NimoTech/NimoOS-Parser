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
