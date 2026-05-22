import asyncio
import logging
import sqlite3
import time
from typing import Optional

from parser.repo_jobs import dequeue_job, complete_job, fail_job

log = logging.getLogger("parser.workers")


class WorkerPool:
    def __init__(
        self, conn: sqlite3.Connection, *, text_pipeline, concurrency: int = 2,
        lease_s: int = 300, max_attempts: int = 5,
        idle_sleep_s: float = 0.5,
    ) -> None:
        self.conn = conn
        self.text_pipeline = text_pipeline
        self.concurrency = concurrency
        self.lease_s = lease_s
        self.max_attempts = max_attempts
        self.idle_sleep_s = idle_sleep_s
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()

    async def start(self) -> None:
        self._stop.clear()
        for i in range(self.concurrency):
            self._tasks.append(asyncio.create_task(self._loop(i)))

    async def stop(self) -> None:
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _loop(self, worker_id: int) -> None:
        while not self._stop.is_set():
            now = int(time.time() * 1000)
            job = await asyncio.to_thread(
                dequeue_job, self.conn, lease_s=self.lease_s, now_ms=now,
            )
            if job is None:
                try:
                    await asyncio.wait_for(self._stop.wait(),
                                            timeout=self.idle_sleep_s)
                except asyncio.TimeoutError:
                    pass
                continue
            try:
                await asyncio.to_thread(self._process, job)
                await asyncio.to_thread(
                    complete_job, self.conn, job["id"],
                    int(time.time() * 1000),
                )
            except Exception as e:
                log.exception("worker %s failed job id=%s", worker_id, job["id"])
                await asyncio.to_thread(
                    fail_job, self.conn, job_id=job["id"],
                    error=str(e), now_ms=int(time.time() * 1000),
                    max_attempts=self.max_attempts,
                )

    def _process(self, job: sqlite3.Row) -> None:
        op = job["op"]
        if op == "index" or op == "reindex":
            self.text_pipeline.index_file(
                root_id=job["root_id"], path=job["path"],
                now_ms=int(time.time() * 1000),
            )
        elif op == "delete":
            if hasattr(self.text_pipeline, "delete_path"):
                self.text_pipeline.delete_path(
                    root_id=job["root_id"], path=job["path"],
                    now_ms=int(time.time() * 1000),
                )
        else:
            raise ValueError(f"unknown op: {op}")
