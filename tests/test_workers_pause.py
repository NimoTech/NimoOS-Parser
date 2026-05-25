import asyncio
import sqlite3
import time
from pathlib import Path

import pytest

from parser.db import init_db
from parser.repo_jobs import enqueue_job, list_jobs
from parser.workers import WorkerPool


class FakePipeline:
    def __init__(self):
        self.indexed: list[str] = []

    def index_file(self, *, root_id: str, path: str, now_ms: int) -> None:
        self.indexed.append(path)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    return init_db(tmp_path / "parser.db")


@pytest.mark.asyncio
async def test_pool_pauses_workers_so_no_job_consumed(conn: sqlite3.Connection):
    pipeline = FakePipeline()
    pool = WorkerPool(conn, text_pipeline=pipeline, concurrency=2,
                     lease_s=60, max_attempts=5, idle_sleep_s=0.05)
    await pool.pause()
    await pool.start()
    now = int(time.time() * 1000)
    enqueue_job(conn, root_id="r", path="/a.md", op="index", priority=100, now_ms=now)
    await asyncio.sleep(0.3)
    # paused, so nothing indexed
    assert pipeline.indexed == []
    assert len(list_jobs(conn, status="pending", limit=10)) == 1
    await pool.stop()


@pytest.mark.asyncio
async def test_pool_resume_unblocks_workers(conn: sqlite3.Connection):
    pipeline = FakePipeline()
    pool = WorkerPool(conn, text_pipeline=pipeline, concurrency=1,
                     lease_s=60, max_attempts=5, idle_sleep_s=0.05)
    await pool.pause()
    await pool.start()
    now = int(time.time() * 1000)
    enqueue_job(conn, root_id="r", path="/a.md", op="index", priority=100, now_ms=now)
    await asyncio.sleep(0.2)
    assert pipeline.indexed == []
    await pool.resume()
    await asyncio.sleep(0.5)
    assert pipeline.indexed == ["/a.md"]
    await pool.stop()
