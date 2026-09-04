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
from parser.repo_state import get_verify_last, set_verify_last
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
                    diff: RootDiff, now_ms: int, delete_extra: bool = True) -> dict:
    """Act on one root's diff. `delete_extra=False` withholds the `extra`
    deletes (run_verify sets it when Wiki reported zero files for a root we
    hold records for — see the mass-deletion guard there)."""
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
    if delete_extra:
        for path in diff.extra:
            text_pipeline.delete_path(root_id=root_id, path=path, now_ms=now_ms)
            out["extra_deleted"] += 1
    return out


def _local_files(conn, root_id: str) -> dict:
    return {r[0]: r[1] for r in conn.execute(
        "SELECT path, mtime_ms FROM file_paths WHERE root_id = ?", (root_id,))}


def _local_roots(conn) -> set:
    return {r[0] for r in conn.execute("SELECT DISTINCT root_id FROM file_paths")}


def _finalize_verify(conn, now_ms: int, error: str) -> bool:
    """Stamp a terminal result onto an unfinished verify_last record.
    Returns True when a record was repaired."""
    last = get_verify_last(conn)
    if not isinstance(last, dict) or last.get("finished_at") is not None:
        return False
    last["finished_at"] = now_ms
    last["ok"] = False
    last["error"] = error
    set_verify_last(conn, last)
    return True


def repair_interrupted_verify(conn, now_ms: int) -> bool:
    """Called at startup: a verify that was running when the process died
    leaves finished_at null forever, so /stats reports a verify in progress
    that nobody will ever finish."""
    return _finalize_verify(conn, now_ms, "interrupted by restart")


def mark_verify_cancelled(conn, now_ms: int) -> bool:
    """Called by VerifyRunner.stop(): CancelledError is a BaseException, so
    run_verify's own except clause never lands a result for it."""
    return _finalize_verify(conn, now_ms, "cancelled")


async def _wiki_files(wiki, root_id: str) -> dict:
    out, after = {}, ""
    while True:
        page = await wiki.fetch_root_files(root_id, after=after, limit=VERIFY_PAGE_SIZE)
        for f in page.get("files", []):
            out[f["path"]] = f.get("mtime_ms", 0)
        nxt = page.get("next_after") or ""
        if not nxt:
            return out
        if nxt <= after:
            # Wiki paging is keyset-based on path; a cursor that doesn't move
            # forward would loop this coroutine forever inside one verify.
            log.warning("verify: wiki file paging for root %s stalled at %r "
                        "after %d files; stopping", root_id, nxt, len(out))
            return out
        after = nxt


async def run_verify(conn, qstore, wiki, text_pipeline, *, root_ids, trigger: str,
                     now_ms=None) -> dict:
    """Reconcile Parser's ledger against Wiki's file_index and land the result
    on parser_state.verify_last (also served by GET /stats).

    Every SQLite touch goes through asyncio.to_thread: a 100k-path root would
    otherwise block the event loop long enough to stall HTTP, the Wiki
    consumer and the worker pool's lease heartbeats.

    On failure `retired_roots` / `roots` may be partially populated — they
    list what actually happened before the error, not a plan that was rolled
    back (there is no transaction; the connection is autocommit).
    """
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    res = {"trigger": trigger, "started_at": now_ms, "finished_at": None, "ok": False,
           "roots": [], "retired_roots": [], "unknown_roots": [], "error": None}
    await asyncio.to_thread(set_verify_last, conn, res)

    async def _land(error: str | None) -> dict:
        res["error"] = error
        res["ok"] = error is None
        # Fresh clock: copying started_at made every verify look instantaneous.
        res["finished_at"] = int(time.time() * 1000)
        await asyncio.to_thread(set_verify_last, conn, res)
        log.info("verify (%s) done: %s", trigger, res)
        return res

    try:
        wiki_roots = {r["id"]: r for r in await wiki.list_roots()}
        if not wiki_roots:
            # Data-safety gate. "Wiki lists nothing" is indistinguishable from
            # a shape/serialization break upstream (repo.WikiRoot has no json
            # tags), and local_roots - {} is the entire ledger — the retire
            # loop below would tombstone every file we hold and still report
            # ok. Refuse before touching anything.
            log.error("verify (%s) aborted: wiki returned no roots; refusing to "
                      "retire %d local roots", trigger,
                      len(await asyncio.to_thread(_local_roots, conn)))
            return await _land("wiki returned no roots")

        local_roots = await asyncio.to_thread(_local_roots, conn)
        for rid in sorted(local_roots - set(wiki_roots)):
            await asyncio.to_thread(retire_root, conn, qstore, root_id=rid, now_ms=now_ms)
            res["retired_roots"].append(rid)

        if root_ids is None:
            targets = sorted(wiki_roots)
        else:
            targets = [r for r in root_ids if r in wiki_roots]
            res["unknown_roots"] = [r for r in root_ids if r not in wiki_roots]
            if res["unknown_roots"]:
                log.warning("verify: requested roots unknown to wiki: %s",
                            res["unknown_roots"])

        attempted = not_found = 0
        for rid in targets:
            if not wiki_roots[rid].get("enabled", True):
                # Wiki disables a root when its disk vanishes but keeps the
                # file_index. The root is present (never retire it) yet its
                # files are unreachable, so missing/stale enqueues would only
                # burn worker attempts.
                res["roots"].append({"root_id": rid, "skipped": "disabled"})
                continue
            attempted += 1
            try:
                wiki_files = await _wiki_files(wiki, rid)
            except WikiRootNotFound:
                not_found += 1
                log.warning("verify: root %s not served by wiki /_internal/files; "
                            "skipping", rid)
                continue
            local = await asyncio.to_thread(_local_files, conn, rid)
            diff = diff_root(wiki_files, local)
            entry = {"root_id": rid, "wiki_files": len(wiki_files),
                     "local_files": len(local), "missing": len(diff.missing),
                     "stale": len(diff.stale), "extra": len(diff.extra)}
            # Mass-deletion guard: an empty Wiki side for a root we do hold
            # records for is as likely a broken file_index as a genuinely
            # emptied root, and delete_path is not reversible.
            delete_extra = not (len(wiki_files) == 0 and len(local) > 0)
            if not delete_extra:
                entry["extra_skipped"] = len(diff.extra)
                log.warning("verify: wiki reports 0 files for root %s while parser "
                            "holds %d; withholding %d extra deletes",
                            rid, len(local), len(diff.extra))
            await asyncio.to_thread(apply_root_diff, conn, text_pipeline,
                                    root_id=rid, diff=diff, now_ms=now_ms,
                                    delete_extra=delete_extra)
            res["roots"].append(entry)

        if attempted and not_found == attempted:
            # Old Wiki without the /_internal/files route: every root 404s and
            # every root gets skipped. Reporting ok there would advertise a
            # reconciliation that never compared anything (spec §7).
            return await _land("wiki files endpoint unavailable")
        return await _land(None)
    except Exception as e:  # noqa: BLE001 - verify must always land a result
        log.exception("verify failed")
        return await _land(f"{type(e).__name__}: {e}")


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
        self._task = asyncio.get_running_loop().create_task(run_verify(
            self.conn, self.qstore, self.wiki, self.text_pipeline,
            root_ids=root_ids, trigger=trigger))
        return True

    async def wait(self) -> None:
        if self._task is not None:
            await self._task

    async def stop(self) -> None:
        """Cancel an in-flight verify at shutdown. Without this the task keeps
        running against a conn/qstore the shutdown path is about to close, and
        verify_last stays finished_at=null forever."""
        task, self._task = self._task, None
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001 - run_verify already logged it
            log.exception("verify task raised while stopping")
        await asyncio.to_thread(mark_verify_cancelled, self.conn,
                                int(time.time() * 1000))
