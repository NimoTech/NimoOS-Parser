import asyncio
import sqlite3
import time
from pathlib import Path

import pytest

from parser.db import init_db
from parser.repo_jobs import enqueue_job, list_jobs
from parser.workers import WorkerPool


class SlowPipeline:
    """每个 job 拖 0.1s, 模拟真实工作"""
    def __init__(self):
        self.indexed: list[str] = []

    def index_file(self, *, root_id: str, path: str, now_ms: int) -> None:
        time.sleep(0.1)
        self.indexed.append(path)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    return init_db(tmp_path / "parser.db")


@pytest.mark.asyncio
async def test_scale_down_finishes_in_flight_job_cleanly(conn: sqlite3.Connection):
    """缩容时,正在跑的 job 必须走完 complete_job 写库,不应残留 in-flight"""
    pipeline = SlowPipeline()
    pool = WorkerPool(conn, text_pipeline=pipeline, concurrency=2,
                     lease_s=60, max_attempts=5, idle_sleep_s=0.02)
    await pool.start()
    now = int(time.time() * 1000)
    for i in range(4):
        enqueue_job(conn, root_id="r", path=f"/a{i}.md", op="index",
                    priority=100, now_ms=now)
    # 让 workers 抓到 job
    await asyncio.sleep(0.05)
    await pool.set_concurrency(1)
    # 等所有 job 跑完
    await asyncio.sleep(1.0)
    # 缩容后只有 1 个 worker,4 个 job 串行需要至少 4 * 0.1s = 0.4s
    assert len(pipeline.indexed) == 4
    # 所有 job 都标记 done(没有 in-flight 残留)
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
    # 4 workers 并行 ~0.1s 一轮,vs 1 worker 串行 0.4s
    await asyncio.sleep(0.5)
    assert len(pipeline.indexed) == 4
    await pool.stop()


@pytest.mark.asyncio
async def test_scale_down_returns_immediately_does_not_block(conn: sqlite3.Connection):
    """set_concurrency 必须立即返回,不等 drain"""
    pipeline = SlowPipeline()
    pool = WorkerPool(conn, text_pipeline=pipeline, concurrency=2,
                     lease_s=60, max_attempts=5, idle_sleep_s=0.02)
    await pool.start()
    now = int(time.time() * 1000)
    enqueue_job(conn, root_id="r", path="/a.md", op="index", priority=100, now_ms=now)
    await asyncio.sleep(0.02)  # worker 抓到 job 开始跑
    t0 = time.perf_counter()
    await pool.set_concurrency(1)
    dt = time.perf_counter() - t0
    # 即使有正在跑的 job(~0.1s 完成),set_concurrency 也应在 ~10ms 内返回
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
