"""The single sqlite3 connection from init_db is shared by every worker
thread, the wiki poll loop and FastAPI's threadpool. dequeue_job opens an
explicit transaction on it. Without process-level serialization, concurrent
callers interleave BEGIN/COMMIT/ROLLBACK and raise
'cannot start a transaction within a transaction' — and a worker whose
dequeue raises dies silently."""
import asyncio
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from parser.db import init_db
from parser.repo_jobs import complete_job, dequeue_job, enqueue_job
from parser.workers import WorkerPool


def test_concurrent_dequeue_on_shared_connection_is_safe(tmp_path: Path):
    conn = init_db(tmp_path / "parser.db")
    now = int(time.time() * 1000)
    n_jobs = 200
    for i in range(n_jobs):
        enqueue_job(conn, root_id="r", path=f"/f{i}.md", op="index",
                    priority=100, now_ms=now)

    errors: list[BaseException] = []
    picked: list[int] = []
    lock = threading.Lock()

    def worker():
        while True:
            try:
                job = dequeue_job(conn, lease_s=60, now_ms=int(time.time() * 1000))
                if job is None:
                    return
                with lock:
                    picked.append(job["id"])
                complete_job(conn, job["id"], int(time.time() * 1000))
            except Exception as e:  # noqa: BLE001
                errors.append(e)
                return

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"{len(errors)} errors, first: {errors[0]!r}"
    assert sorted(picked) == list(range(1, n_jobs + 1)), "every job exactly once"


class NoopPipeline:
    def __init__(self):
        self.indexed: list[str] = []

    def index_file(self, *, root_id: str, path: str, now_ms: int) -> None:
        self.indexed.append(path)


@pytest.mark.asyncio
async def test_worker_survives_a_failing_dequeue(tmp_path: Path, monkeypatch):
    conn = init_db(tmp_path / "parser.db")
    calls = {"n": 0}
    real = dequeue_job

    def flaky(c, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return real(c, **kw)

    monkeypatch.setattr("parser.workers.dequeue_job", flaky)
    monkeypatch.setattr(WorkerPool, "_pacing_delay", lambda self: 0.0)
    pipeline = NoopPipeline()
    pool = WorkerPool(conn, text_pipeline=pipeline, concurrency=1,
                      lease_s=60, max_attempts=5, idle_sleep_s=0.02)
    await pool.start()
    enqueue_job(conn, root_id="r", path="/x.md", op="index", priority=100,
                now_ms=int(time.time() * 1000))
    for _ in range(100):
        if pipeline.indexed:
            break
        await asyncio.sleep(0.02)
    await pool.stop()
    assert pipeline.indexed == ["/x.md"], "worker must recover after a transient dequeue error"
