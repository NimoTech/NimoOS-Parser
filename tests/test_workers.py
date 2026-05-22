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
