import asyncio
import sqlite3
import time
from pathlib import Path

import pytest

from parser.db import init_db
from parser.repo_jobs import enqueue_job, list_jobs
from parser.workers import WorkerPool


class SlowPipeline:
    """Each job takes 0.1s, simulating real work"""
    def __init__(self):
        self.indexed: list[str] = []

    def index_file(self, *, root_id: str, path: str, now_ms: int) -> None:
        time.sleep(0.1)
        self.indexed.append(path)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    return init_db(tmp_path / "parser.db")


@pytest.fixture(autouse=True)
def no_pacing(monkeypatch):
    """These tests exercise scaling mechanics, not pacing. concurrency doubles
    as the pacing tier (parser/pacing.py) so concurrency=1/2 here would
    otherwise incur multi-second inter-job delays regardless of load."""
    monkeypatch.setattr(WorkerPool, "_pacing_delay", lambda self: 0.0)


@pytest.mark.asyncio
async def test_scale_down_finishes_in_flight_job_cleanly(conn: sqlite3.Connection):
    """When scaling down, an in-flight job must run through complete_job to write to the DB; it must not be left dangling in-flight"""
    pipeline = SlowPipeline()
    pool = WorkerPool(conn, text_pipeline=pipeline, concurrency=2,
                     lease_s=60, max_attempts=5, idle_sleep_s=0.02)
    await pool.start()
    now = int(time.time() * 1000)
    for i in range(4):
        enqueue_job(conn, root_id="r", path=f"/a{i}.md", op="index",
                    priority=100, now_ms=now)
    # let the workers pick up the jobs
    await asyncio.sleep(0.05)
    await pool.set_concurrency(1)
    # wait for all jobs to finish
    await asyncio.sleep(1.0)
    # after scaling down there's only 1 worker; running 4 jobs serially needs at least 4 * 0.1s = 0.4s
    assert len(pipeline.indexed) == 4
    # all jobs are marked done (no in-flight leftovers)
    assert list_jobs(conn, status="running", limit=10) == []
    assert list_jobs(conn, status="pending", limit=10) == []
    await pool.stop()


@pytest.mark.asyncio
async def test_scale_up_adds_workers(conn: sqlite3.Connection):
    pipeline = SlowPipeline()
    pool = WorkerPool(conn, text_pipeline=pipeline, concurrency=1,
                     lease_s=60, max_attempts=5, idle_sleep_s=0.02)
    await pool.start()
    now = int(time.time() * 1000)
    for i in range(4):
        enqueue_job(conn, root_id="r", path=f"/a{i}.md", op="index",
                    priority=100, now_ms=now)
    await asyncio.sleep(0.05)
    await pool.set_concurrency(4)
    # 4 workers running in parallel take ~0.1s per round, vs. 0.4s serial with 1 worker
    await asyncio.sleep(0.5)
    assert len(pipeline.indexed) == 4
    await pool.stop()


@pytest.mark.asyncio
async def test_scale_down_returns_immediately_does_not_block(conn: sqlite3.Connection):
    """set_concurrency must return immediately, without waiting for drain"""
    pipeline = SlowPipeline()
    pool = WorkerPool(conn, text_pipeline=pipeline, concurrency=2,
                     lease_s=60, max_attempts=5, idle_sleep_s=0.02)
    await pool.start()
    now = int(time.time() * 1000)
    enqueue_job(conn, root_id="r", path="/a.md", op="index", priority=100, now_ms=now)
    await asyncio.sleep(0.02)  # worker picks up the job and starts running it
    t0 = time.perf_counter()
    await pool.set_concurrency(1)
    dt = time.perf_counter() - t0
    # even with a job in flight (~0.1s to complete), set_concurrency should return within ~10ms
    assert dt < 0.1, f"set_concurrency blocked {dt*1000:.1f}ms; should be fire-and-forget within 100ms"
    await asyncio.sleep(0.3)
    await pool.stop()


@pytest.mark.asyncio
async def test_set_concurrency_after_stop_raises(conn: sqlite3.Connection):
    pipeline = SlowPipeline()
    pool = WorkerPool(conn, text_pipeline=pipeline, concurrency=1,
                     lease_s=60, max_attempts=5, idle_sleep_s=0.02)
    await pool.start()
    await pool.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        await pool.set_concurrency(2)
