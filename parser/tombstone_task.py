"""Async background sweep that tombstones file_ids whose current paths
no longer pass the allowlist. Triggered on allowlist mutation.

Concurrency: a single asyncio.Event "wake" signal coalesces multiple
mutations into one sweep — if 5 patches arrive in 100ms, we only sweep
once. Sweep itself takes a snapshot of file_records and tombstones in
order; new writes during the sweep are handled by the next wake.
"""
import asyncio
import logging
import sqlite3
import time
from typing import Optional

from parser import repo_allowlist

log = logging.getLogger("parser.tombstone_task")

# Module-level coalescing event. main.py creates the worker, routes call
# schedule_allowlist_sweep() to set it.
_wake_event: Optional[asyncio.Event] = None


def set_wake_event(ev: asyncio.Event) -> None:
    global _wake_event
    _wake_event = ev


def schedule_allowlist_sweep() -> None:
    """Tell the background worker to run a sweep ASAP."""
    if _wake_event is not None:
        _wake_event.set()


async def sweep_once(conn: sqlite3.Connection, *, qstore,
                      now_ms: Optional[int] = None) -> int:
    """Scan file_records, tombstone any file_id that no longer passes
    is_path_indexable for ANY of its current paths. Returns affected count."""
    if now_ms is None:
        now_ms = int(time.time() * 1000)

    rows = list(conn.execute(
        "SELECT fr.file_id FROM file_records fr "
        "WHERE fr.tombstoned_at IS NULL"
    ))
    affected = 0
    for r in rows:
        fid = r["file_id"]
        paths = list(conn.execute(
            "SELECT root_id, path FROM file_paths WHERE file_id = ?", (fid,)
        ))
        if not paths:
            continue
        # A file stays indexed if ANY of its paths still passes the allowlist
        # (a file may exist in multiple roots; keep it as long as one is OK).
        if any(repo_allowlist.is_path_indexable(conn,
                                                 root_id=p["root_id"],
                                                 path=p["path"]) for p in paths):
            continue
        qstore.tombstone_file(file_id=fid, tombstoned_at=now_ms)
        conn.execute(
            "UPDATE file_records SET tombstoned_at = ? WHERE file_id = ?",
            (now_ms, fid),
        )
        affected += 1
    return affected


async def worker_loop(conn: sqlite3.Connection, qstore,
                       wake: asyncio.Event) -> None:
    """Run forever: wait for wake → sweep → clear → repeat."""
    while True:
        await wake.wait()
        wake.clear()
        try:
            n = await sweep_once(conn, qstore=qstore)
            if n:
                log.info("allowlist sweep tombstoned %d files", n)
        except Exception:
            log.exception("allowlist sweep failed")
