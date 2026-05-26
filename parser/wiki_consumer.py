import asyncio
import logging
import posixpath
import sqlite3
import time

from parser.repo_jobs import enqueue_job
from parser.repo_models import get_wiki_cursor, set_wiki_cursor

log = logging.getLogger("parser.wiki_consumer")

# 文本类白名单:有正式 text pipeline 支持的扩展。
# MOV/MP4/JPG 等视觉类等 visual pipeline 上线后再开。
TEXT_EXT_ALLOWLIST = {
    # raw text + markdown — plain reader
    ".md", ".txt", ".rst",
    # docling-handled (PDF / Office / web)
    ".pdf",
    ".docx", ".doc",
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


def _is_text_indexable(path: str) -> bool:
    ext = posixpath.splitext(path)[1].lower()
    return ext in TEXT_EXT_ALLOWLIST


def _op_for_event(ev: dict) -> str | None:
    if ev.get("is_dir"):
        return None
    op = ev.get("op")
    if op == "delete":
        # delete 任何路径都转发(让 parser 清掉可能存在的旧向量)
        return "delete"
    if op in ("create", "modify", "rename"):
        if not _is_text_indexable(ev.get("path", "")):
            return None
        return "index"
    return None


class WikiConsumer:
    def __init__(
        self, conn: sqlite3.Connection, wiki, *,
        poll_interval_s: float = 2.0, poll_limit: int = 200,
    ) -> None:
        self.conn = conn
        self.wiki = wiki
        self.poll_interval_s = poll_interval_s
        self.poll_limit = poll_limit
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
                since = await asyncio.to_thread(get_wiki_cursor, self.conn)
                events = await self.wiki.fetch_file_events(
                    since_ms=since, limit=self.poll_limit,
                )
                if events:
                    await asyncio.to_thread(self._ingest, events)
                backoff = self.poll_interval_s
            except Exception as e:
                log.warning("wiki fetch failed: %s; backing off %ss", e, backoff)
                backoff = min(backoff * 2, 60.0)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass

    def _ingest(self, events: list[dict]) -> None:
        max_seen = 0
        now = int(time.time() * 1000)
        for ev in events:
            op = _op_for_event(ev)
            detected = ev.get("detected_at", 0)
            max_seen = max(max_seen, detected)
            if op is None:
                continue
            enqueue_job(
                self.conn, root_id=ev["root_id"], path=ev["path"], op=op,
                priority=100, now_ms=now,
            )
        if max_seen:
            set_wiki_cursor(self.conn, since_ms=max_seen, now_ms=now)
