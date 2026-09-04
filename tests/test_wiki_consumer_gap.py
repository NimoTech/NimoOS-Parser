import asyncio

import pytest

from parser.db import init_db
from parser.repo_models import set_wiki_cursor
from parser.repo_state import get_cursor_gap
from parser.wiki_consumer import WikiConsumer


class PagedWiki:
    def __init__(self, pages):
        self.pages = list(pages)
    async def fetch_file_events_page(self, *, since_ms, after_seq=0, limit):
        return self.pages.pop(0) if self.pages else {"events": []}


def _run(conn, wiki, polls=6):
    gaps = []
    async def on_gap(gap): gaps.append(gap)
    c = WikiConsumer(conn, wiki, poll_interval_s=0.01, poll_limit=10, on_gap=on_gap)
    async def runner():
        await c.start()
        await asyncio.sleep(0.01 * polls + 0.1)
        await c.stop()
    asyncio.run(runner())
    return gaps


@pytest.fixture
def conn(tmp_path):
    return init_db(tmp_path / "p.db")


def test_gap_detected_once_and_persisted(conn):
    set_wiki_cursor(conn, since_ms=1_000, last_seq=3, now_ms=1)
    page = {"events": [], "archive_cutoff_ms": 5_000, "has_archived": True}
    gaps = _run(conn, PagedWiki([page, page, page]))
    assert len(gaps) == 1, "same gap must not re-trigger while the cursor stays behind"
    assert gaps[0]["since_ms"] == 1_000 and gaps[0]["last_seq"] == 3 and gaps[0]["archive_cutoff_ms"] == 5_000
    assert get_cursor_gap(conn) == gaps[0]


def test_no_gap_when_wiki_never_archived_or_cursor_fresh(conn):
    set_wiki_cursor(conn, since_ms=1_000, last_seq=3, now_ms=1)
    assert _run(conn, PagedWiki([{"events": [], "archive_cutoff_ms": 5_000, "has_archived": False}])) == []
    set_wiki_cursor(conn, since_ms=0, last_seq=0, now_ms=1)
    assert _run(conn, PagedWiki([{"events": [], "archive_cutoff_ms": 5_000, "has_archived": True}])) == []
    assert _run(conn, PagedWiki([{"events": []}])) == [], "old Wiki without the fields: no detection"


def test_gap_cleared_once_cursor_catches_up(conn):
    set_wiki_cursor(conn, since_ms=1_000, last_seq=3, now_ms=1)
    behind = {"events": [], "archive_cutoff_ms": 5_000, "has_archived": True}
    caught_up = {"events": [{"id": "e", "root_id": "r", "path": "/x.md", "op": "create",
                             "is_dir": 0, "detected_at": 9_000, "seq": 9}],
                 "archive_cutoff_ms": 5_000, "has_archived": True}
    gaps = _run(conn, PagedWiki([behind, caught_up, behind]))
    # behind -> gap; caught_up advances cursor to 9000 (> cutoff) -> cleared;
    # the third page reports the cursor 9000 >= 5000, so no new gap.
    assert len(gaps) == 1
    assert get_cursor_gap(conn) is None
