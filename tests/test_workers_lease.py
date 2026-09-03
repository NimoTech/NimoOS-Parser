"""Audit P5: a long job outlived its lease (JobLeaseSec) and a second worker
re-picked the same file while the first was still processing it — duplicate
CPU, duplicate upserts, attempts climbing to max. The in-flight worker must
keep its lease alive."""
import asyncio
import time
from pathlib import Path

import pytest

from parser.db import init_db
from parser.repo_jobs import dequeue_job, enqueue_job
from parser.workers import WorkerPool


class SlowPipeline:
    def __init__(self, seconds: float):
        self.seconds = seconds
        self.indexed: list[str] = []

    def index_file(self, *, root_id: str, path: str, now_ms: int) -> None:
        time.sleep(self.seconds)
        self.indexed.append(path)


@pytest.mark.asyncio
async def test_in_flight_job_keeps_its_lease(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(WorkerPool, "_pacing_delay", lambda self: 0.0)
    conn = init_db(tmp_path / "p.db")
    pipeline = SlowPipeline(2.5)
    pool = WorkerPool(conn, text_pipeline=pipeline, concurrency=1,
                      lease_s=1, max_attempts=5, idle_sleep_s=0.02)
    enqueue_job(conn, root_id="r", path="/slow.md", op="index", priority=100,
                now_ms=int(time.time() * 1000))
    await pool.start()
    await asyncio.sleep(1.6)  # past the original 1 s lease, job still running
    # Another worker asking for work now must NOT get the same job back.
    other = dequeue_job(conn, lease_s=1, now_ms=int(time.time() * 1000))
    assert other is None, "lease expired while the job was still in flight"
    for _ in range(100):
        if pipeline.indexed:
            break
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.1)
    await pool.stop()
    row = conn.execute("select attempts, done_at, last_error from parse_jobs").fetchone()
    assert pipeline.indexed == ["/slow.md"]
    assert row["done_at"] is not None and row["last_error"] is None
    assert row["attempts"] == 1, "one pick, one attempt — no re-pick while running"
