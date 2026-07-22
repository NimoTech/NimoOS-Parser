import asyncio
import collections
import json
import logging
import sqlite3
import time
from typing import Optional

from parser import pacing
from parser.repo_jobs import dequeue_job, complete_job, fail_job
from parser.repo_records import set_last_error

log = logging.getLogger("parser.workers")


class WorkerPool:
    _WINDOW_S = 600.0

    def __init__(
        self, conn: sqlite3.Connection, *, text_pipeline, visual_pipeline=None,
        concurrency: int = 2,
        lease_s: int = 300, max_attempts: int = 5,
        idle_sleep_s: float = 0.5,
        wiki_client=None, parser_version: str = "parser/0.1.0",
        load_ratio_fn=pacing.load_ratio,
    ) -> None:
        self.conn = conn
        self.text_pipeline = text_pipeline
        self.visual_pipeline = visual_pipeline
        self.concurrency = concurrency
        self.lease_s = lease_s
        self.max_attempts = max_attempts
        self.idle_sleep_s = idle_sleep_s
        self.wiki_client = wiki_client
        self.parser_version = parser_version
        self._load_ratio_fn = load_ratio_fn
        self._done_ts: collections.deque = collections.deque()  # completion timestamps (10-min window)
        self._tasks: list[tuple[int, asyncio.Task]] = []
        self._worker_id_seq: int = 0
        self._worker_exit_flags: dict[int, asyncio.Event] = {}
        self._concurrency_lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._run_event = asyncio.Event()
        # set = running, clear = paused (Python asyncio has no wait-until-cleared API)
        self._run_event.set()

    def _pacing_delay(self) -> float:
        return pacing.sleep_seconds(self.concurrency, self._load_ratio_fn())

    def _record_done(self) -> None:
        now = time.time()
        self._done_ts.append(now)
        while self._done_ts and self._done_ts[0] < now - self._WINDOW_S:
            self._done_ts.popleft()

    def throughput(self) -> dict:
        """Rolling completion stats for /v1/parser/stats (spec §4.8)."""
        now = time.time()
        while self._done_ts and self._done_ts[0] < now - self._WINDOW_S:
            self._done_ts.popleft()
        n = len(self._done_ts)
        return {"done_last_10m": n, "rate_per_min": round(n / (self._WINDOW_S / 60.0), 2)}

    async def start(self) -> None:
        self._stop.clear()
        async with self._concurrency_lock:
            for _ in range(self.concurrency):
                wid = self._worker_id_seq
                self._worker_id_seq += 1
                flag = asyncio.Event()
                self._worker_exit_flags[wid] = flag
                t = asyncio.create_task(self._loop(wid))
                self._tasks.append((wid, t))

    async def stop(self) -> None:
        self._stop.set()
        tasks = [t for _, t in self._tasks]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._worker_exit_flags.clear()

    async def pause(self) -> None:
        self._run_event.clear()

    async def resume(self) -> None:
        self._run_event.set()

    async def set_concurrency(self, n: int) -> None:
        if n < 1:
            raise ValueError(f"concurrency must be >=1, got {n}")
        if self._stop.is_set():
            raise RuntimeError("cannot set_concurrency on a stopped pool")
        async with self._concurrency_lock:
            current = len(self._tasks)
            if n > current:
                for _ in range(n - current):
                    wid = self._worker_id_seq
                    self._worker_id_seq += 1
                    flag = asyncio.Event()
                    self._worker_exit_flags[wid] = flag
                    t = asyncio.create_task(self._loop(wid))
                    self._tasks.append((wid, t))
            elif n < current:
                to_drain = self._tasks[n:]
                self._tasks = self._tasks[:n]
                for wid, _ in to_drain:
                    self._worker_exit_flags[wid].set()
                asyncio.create_task(self._drain_workers(to_drain))
            self.concurrency = n

    async def _drain_workers(
        self, draining: list[tuple[int, asyncio.Task]]
    ) -> None:
        await asyncio.gather(*(t for _, t in draining), return_exceptions=True)
        for wid, _ in draining:
            self._worker_exit_flags.pop(wid, None)

    async def _interruptible_sleep(self, delay: float,
                                   exit_flag: asyncio.Event) -> None:
        """Sleep up to `delay`, waking early on pool stop OR this worker's
        drain flag (set_concurrency down used to strand drained workers in
        the pacing sleep for up to 60s in power-save mode)."""
        wait_stop = asyncio.create_task(self._stop.wait())
        wait_exit = asyncio.create_task(exit_flag.wait())
        try:
            await asyncio.wait(
                {wait_stop, wait_exit},
                timeout=delay, return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for w in (wait_stop, wait_exit):
                w.cancel()
            await asyncio.gather(wait_stop, wait_exit, return_exceptions=True)

    async def _loop(self, worker_id: int) -> None:
        exit_flag = self._worker_exit_flags[worker_id]
        while not self._stop.is_set() and not exit_flag.is_set():
            if not self._run_event.is_set():
                wait_run = asyncio.create_task(self._run_event.wait())
                wait_stop = asyncio.create_task(self._stop.wait())
                wait_exit = asyncio.create_task(exit_flag.wait())
                try:
                    done, pending = await asyncio.wait(
                        [wait_run, wait_stop, wait_exit],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for p in pending:
                        p.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                except asyncio.CancelledError:
                    for w in (wait_run, wait_stop, wait_exit):
                        w.cancel()
                    await asyncio.gather(wait_run, wait_stop, wait_exit, return_exceptions=True)
                    raise
                continue
            now = int(time.time() * 1000)
            job = await asyncio.to_thread(
                dequeue_job, self.conn, lease_s=self.lease_s, now_ms=now,
            )
            if job is None:
                await self._interruptible_sleep(self.idle_sleep_s, exit_flag)
                continue
            try:
                await asyncio.to_thread(self._process, job)
                await asyncio.to_thread(
                    complete_job, self.conn, job["id"],
                    int(time.time() * 1000),
                )
                self._record_done()
                # Clear any stale last_error on this file (write-through to
                # file_records so the file list API can answer status without
                # joining parse_jobs).
                await asyncio.to_thread(
                    set_last_error, self.conn,
                    root_id=job["root_id"], path=job["path"], error=None,
                )
                # visual op(视觉资产入库)不走 Wiki 回执协议——那是给 Wiki
                # 文件类 job 用的确认通道,visual_ingest 的调用方是 Photos。
                if not job["op"].startswith("visual"):
                    await self._notify_wiki(job, status="indexed" if job["op"] != "delete" else "deleted")
            except Exception as e:
                log.exception("worker %s failed job id=%s", worker_id, job["id"])
                await asyncio.to_thread(
                    fail_job, self.conn, job_id=job["id"],
                    error=str(e), now_ms=int(time.time() * 1000),
                    max_attempts=self.max_attempts,
                )
                # Mirror the error into file_records.last_error. No-op if the
                # file_path row doesn't exist yet (failure before file_record
                # creation).
                await asyncio.to_thread(
                    set_last_error, self.conn,
                    root_id=job["root_id"], path=job["path"], error=str(e),
                )
                if not job["op"].startswith("visual"):
                    await self._notify_wiki(job, status="failed", error=str(e))

            delay = self._pacing_delay()
            if delay > 0:
                await self._interruptible_sleep(delay, exit_flag)

    async def _notify_wiki(self, job: sqlite3.Row, *,
                           status: str, error: Optional[str] = None) -> None:
        if self.wiki_client is None:
            return
        modalities = None
        if status == "indexed":
            modalities = await asyncio.to_thread(self._fetch_modalities, job)
        try:
            await self.wiki_client.report_index_status(
                path=job["path"], status=status,
                parser_version=self.parser_version,
                modalities=modalities, error=error,
            )
        except Exception as e:
            log.warning("wiki report_index_status failed for %s: %s",
                        job["path"], e)

    def _fetch_modalities(self, job: sqlite3.Row) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT fr.modalities_done FROM file_paths fp "
            "JOIN file_records fr ON fp.file_id = fr.file_id "
            "WHERE fp.root_id = ? AND fp.path = ?",
            (job["root_id"], job["path"]),
        ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["modalities_done"])
        except (ValueError, TypeError):
            return None

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
        elif op == "visual_ingest":
            if self.visual_pipeline is None:
                # Qdrant 掉线时 startup 不接线;抛错让 job 走失败重试,
                # Qdrant 恢复后 retry_failed_jobs 可整批捞回。
                raise RuntimeError("visual pipeline not wired (qdrant down?)")
            payload = json.loads(job["sub_modality"] or "{}")
            self.visual_pipeline.ingest_asset(
                source=job["root_id"], asset_id=payload["asset_id"],
                image_path=job["path"], mime=payload.get("mime", "image/jpeg"),
                meta=payload.get("meta", {}),
                now_ms=int(time.time() * 1000),
            )
        else:
            raise ValueError(f"unknown op: {op}")
