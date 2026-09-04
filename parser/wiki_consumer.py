import asyncio
import logging
import posixpath
import sqlite3
import time

from parser.repo_jobs import enqueue_job
from parser.repo_models import get_wiki_cursor, set_wiki_cursor
from parser.repo_state import get_cursor_gap, set_cursor_gap

log = logging.getLogger("parser.wiki_consumer")

# Text-type allowlist: extensions formally supported by the text pipeline.
# Visual types like MOV/MP4/JPG will be enabled once the visual pipeline ships.
TEXT_EXT_ALLOWLIST = {
    # raw text + markdown — plain reader
    ".md", ".txt", ".rst",
    # docling-handled (PDF / Office / web)
    ".pdf",
    ".docx", ".doc", ".wps",
    ".pptx", ".ppt",
    ".xlsx", ".xls",
    ".odt",
    ".html", ".htm", ".xml",
    # source code — chunk_source
    ".py", ".go", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".c", ".cc", ".cpp", ".h", ".hpp", ".cs", ".rb", ".rs", ".php",
    ".sh", ".bash", ".zsh", ".fish",
    # structured text — chunk_plain
    ".json", ".yaml", ".yml", ".toml", ".ini", ".env",
    ".csv", ".tsv", ".sql",
    ".log",
}


def _is_text_indexable(conn: sqlite3.Connection, root_id: str,
                        path: str) -> bool:
    from parser import repo_allowlist
    return repo_allowlist.is_path_indexable(conn, root_id=root_id, path=path)


def _op_for_event(ev: dict, conn: sqlite3.Connection) -> str | None:
    if ev.get("op") == "root_removed":
        # One event per deleted root (Wiki >= 2026-09). Not subject to the
        # allowlist: it retires records, it never indexes.
        return "retire_root"
    if ev.get("is_dir"):
        return None
    op = ev.get("op")
    if op == "delete":
        # forward delete for any path (lets parser clean up any old vectors that may exist)
        return "delete"
    if op in ("create", "modify", "rename"):
        if not _is_text_indexable(conn, ev.get("root_id", ""),
                                  ev.get("path", "")):
            return None
        return "index"
    return None


class WikiConsumer:
    def __init__(
        self, conn: sqlite3.Connection, wiki, *,
        poll_interval_s: float = 2.0, poll_limit: int = 200,
        on_gap=None,
    ) -> None:
        self.conn = conn
        self.wiki = wiki
        self.poll_interval_s = poll_interval_s
        self.poll_limit = poll_limit
        self.on_gap = on_gap
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
            self._task = None

    async def _loop(self) -> None:
        backoff = self.poll_interval_s
        while not self._stop.is_set():
            try:
                since, seq = await asyncio.to_thread(get_wiki_cursor, self.conn)
                page = await self.wiki.fetch_file_events_page(
                    since_ms=since, after_seq=seq, limit=self.poll_limit,
                )
                events = page.get("events", [])
                if events:
                    await asyncio.to_thread(self._ingest, events)
                await self._check_archive_gap(page)
                backoff = self.poll_interval_s
            except Exception as e:
                log.warning("wiki fetch failed: %s; backing off %ss", e, backoff)
                backoff = min(backoff * 2, 60.0)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass

    async def _check_archive_gap(self, page: dict) -> None:
        """Wiki archives file_events after keepDays; the feed only serves
        archived=0. A cursor older than the archive cutoff means the events in
        between are gone for good. Record it once, hand it to on_gap (which
        runs a verify), and clear once the cursor is back inside the horizon.
        since_ms == 0 is a fresh Parser, not a gap. Old Wikis send neither
        field: no detection."""
        cutoff = page.get("archive_cutoff_ms")
        if cutoff is None or not page.get("has_archived"):
            return
        # Re-read: _ingest may just have advanced the cursor past the cutoff.
        cur_since, cur_seq = await asyncio.to_thread(get_wiki_cursor, self.conn)
        behind = cur_since > 0 and cur_since < cutoff
        existing = await asyncio.to_thread(get_cursor_gap, self.conn)
        if behind and existing is None:
            gap = {"detected_at": int(time.time() * 1000), "since_ms": cur_since,
                   "last_seq": cur_seq, "archive_cutoff_ms": int(cutoff),
                   "triggered": False}
            await asyncio.to_thread(set_cursor_gap, self.conn, gap)
            log.warning("wiki cursor %s is behind the archive cutoff %s: events were archived "
                        "before we consumed them; running verify", cur_since, cutoff)
            await self._fire_gap(gap)
        elif behind and existing is not None and existing.get("triggered") is False:
            # The record is written before the handler runs, so a handler that
            # raised used to leave a gap nobody would ever act on: the next
            # poll saw `existing` and skipped it forever. Retry until it lands.
            log.info("retrying the verify for the recorded cursor gap")
            await self._fire_gap(existing)
        elif not behind and existing is not None:
            await asyncio.to_thread(set_cursor_gap, self.conn, None)
            log.info("wiki cursor back inside the archive horizon; gap cleared")

    async def _fire_gap(self, gap: dict) -> None:
        """Run on_gap and record whether it actually took. Its failure must not
        reach _loop, which would only log 'wiki fetch failed' and double the
        poll backoff for something unrelated to fetching."""
        if self.on_gap is None:
            gap["triggered"] = True  # nothing to retry; don't loop forever
        else:
            try:
                await self.on_gap(gap)
                gap["triggered"] = True
            except Exception as e:  # noqa: BLE001 - retried on the next poll
                log.warning("cursor gap handler failed: %s", e)
                gap["triggered"] = False
        await asyncio.to_thread(set_cursor_gap, self.conn, gap)

    def _ingest(self, events: list[dict]) -> None:
        if not events:  # _loop guards this; keep future direct callers safe
            return
        now = int(time.time() * 1000)
        for ev in events:
            op = _op_for_event(ev, self.conn)
            if op is None:
                continue
            enqueue_job(
                self.conn, root_id=ev["root_id"],
                path="" if op == "retire_root" else ev["path"], op=op,
                priority=50 if op == "retire_root" else 100, now_ms=now,
            )
        # Wiki returns ORDER BY (detected_at, rowid); the last event IS the
        # cursor. seq missing (old Wiki) degrades to the legacy detected_at
        # cursor — never worse than before.
        last = events[-1]
        set_wiki_cursor(self.conn, since_ms=last.get("detected_at", 0),
                        last_seq=last.get("seq", 0), now_ms=now)
