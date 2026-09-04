"""Ledger verify: reconcile Parser's file_paths against Wiki's file_index.

Wiki is the authority on "what should be indexed"; Parser's ledger is "what
is indexed". The two only meet through the file_events feed, which can lose
rows (90-day archive, cursor gaps). verify compares the two by (root_id,
path) and mtime — no stat, no hashing; content drift is index_file's job.

Three classes per root:
  missing  Wiki has it, Parser doesn't  -> index job (priority 500), allowlist-gated
  stale    |mtime_wiki - mtime_local| > tolerance -> reindex job (priority 500)
  extra    Parser has it, Wiki doesn't  -> delete_path now
Roots Parser holds that Wiki no longer lists are retired wholesale.
"""
import asyncio
import logging
import sqlite3
import time
from dataclasses import dataclass, field

from parser.repo_allowlist import is_path_indexable
from parser.repo_jobs import enqueue_job
from parser.repo_state import set_verify_last
from parser.service_retire import retire_root
from parser.wiki_client import WikiRootNotFound

log = logging.getLogger("parser.service_verify")

VERIFY_PRIORITY = 500
VERIFY_PAGE_SIZE = 1000
MTIME_TOLERANCE_MS = 1000


@dataclass
class RootDiff:
    missing: list = field(default_factory=list)
    stale: list = field(default_factory=list)
    extra: list = field(default_factory=list)


def diff_root(wiki_files: dict, local_files: dict, *, tolerance_ms: int = MTIME_TOLERANCE_MS) -> RootDiff:
    d = RootDiff()
    for path, wm in wiki_files.items():
        lm = local_files.get(path)
        if lm is None:
            d.missing.append(path)
        elif abs(int(wm) - int(lm)) > tolerance_ms:
            d.stale.append(path)
    d.extra = [p for p in local_files if p not in wiki_files]
    d.missing.sort(); d.stale.sort(); d.extra.sort()
    return d


def _has_open_job(conn, root_id: str, path: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM parse_jobs WHERE root_id = ? AND path = ? AND done_at IS NULL LIMIT 1",
        (root_id, path)).fetchone() is not None


def apply_root_diff(conn: sqlite3.Connection, text_pipeline, *, root_id: str,
                    diff: RootDiff, now_ms: int) -> dict:
    out = {"missing_enqueued": 0, "stale_enqueued": 0, "extra_deleted": 0, "missing_gated": 0}
    for path in diff.missing:
        if not is_path_indexable(conn, root_id=root_id, path=path):
            out["missing_gated"] += 1
            continue
        if _has_open_job(conn, root_id, path):
            continue
        enqueue_job(conn, root_id=root_id, path=path, op="index",
                    priority=VERIFY_PRIORITY, now_ms=now_ms)
        out["missing_enqueued"] += 1
    for path in diff.stale:
        if _has_open_job(conn, root_id, path):
            continue
        enqueue_job(conn, root_id=root_id, path=path, op="reindex",
                    priority=VERIFY_PRIORITY, now_ms=now_ms)
        out["stale_enqueued"] += 1
    for path in diff.extra:
        text_pipeline.delete_path(root_id=root_id, path=path, now_ms=now_ms)
        out["extra_deleted"] += 1
    return out


def _local_files(conn, root_id: str) -> dict:
    return {r[0]: r[1] for r in conn.execute(
        "SELECT path, mtime_ms FROM file_paths WHERE root_id = ?", (root_id,))}


async def _wiki_files(wiki, root_id: str) -> dict:
    out, after = {}, ""
    while True:
        page = await wiki.fetch_root_files(root_id, after=after, limit=VERIFY_PAGE_SIZE)
        for f in page.get("files", []):
            out[f["path"]] = f.get("mtime_ms", 0)
        after = page.get("next_after") or ""
        if not after:
            return out


async def run_verify(conn, qstore, wiki, text_pipeline, *, root_ids, trigger: str,
                     now_ms=None) -> dict:
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    res = {"trigger": trigger, "started_at": now_ms, "finished_at": None, "ok": False,
           "roots": [], "retired_roots": [], "error": None}
    set_verify_last(conn, res)
    try:
        wiki_roots = {r["id"] for r in await wiki.list_roots()}
        local_roots = {r[0] for r in conn.execute("SELECT DISTINCT root_id FROM file_paths")}
        for rid in sorted(local_roots - wiki_roots):
            await asyncio.to_thread(retire_root, conn, qstore, root_id=rid, now_ms=now_ms)
            res["retired_roots"].append(rid)
        targets = sorted(wiki_roots) if root_ids is None else list(root_ids)
        for rid in targets:
            try:
                wiki_files = await _wiki_files(wiki, rid)
            except WikiRootNotFound:
                log.warning("verify: root %s vanished from Wiki mid-run; skipping", rid)
                continue
            local = _local_files(conn, rid)
            diff = diff_root(wiki_files, local)
            await asyncio.to_thread(apply_root_diff, conn, text_pipeline,
                                    root_id=rid, diff=diff, now_ms=now_ms)
            res["roots"].append({"root_id": rid, "wiki_files": len(wiki_files),
                                 "local_files": len(local), "missing": len(diff.missing),
                                 "stale": len(diff.stale), "extra": len(diff.extra)})
        res["ok"] = True
    except Exception as e:  # noqa: BLE001 — verify must always land a result
        log.exception("verify failed")
        res["error"] = f"{type(e).__name__}: {e}"
    res["finished_at"] = now_ms
    set_verify_last(conn, res)
    log.info("verify (%s) done: %s", trigger, res)
    return res


class VerifyRunner:
    """Single-flight owner of run_verify inside the service event loop."""

    def __init__(self, conn, qstore, wiki, text_pipeline) -> None:
        self.conn, self.qstore, self.wiki, self.text_pipeline = conn, qstore, wiki, text_pipeline
        self._task = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self, *, root_ids, trigger: str) -> bool:
        if self.running:
            return False
        self._task = asyncio.get_event_loop().create_task(run_verify(
            self.conn, self.qstore, self.wiki, self.text_pipeline,
            root_ids=root_ids, trigger=trigger))
        return True

    async def wait(self) -> None:
        if self._task is not None:
            await self._task
