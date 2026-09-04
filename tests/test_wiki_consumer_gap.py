import asyncio
import time

import pytest

from parser.db import init_db
from parser.repo_models import set_wiki_cursor
from parser.repo_state import get_cursor_gap
from parser.wiki_consumer import WikiConsumer

SETTLE_S = 0.05
DEADLINE_S = 2.0


class PagedWiki:
    def __init__(self, pages):
        self.pages = list(pages)
    async def fetch_file_events_page(self, *, since_ms, after_seq=0, limit):
        return self.pages.pop(0) if self.pages else {"events": []}


class SteadyWiki:
    """Serves the same page forever, so the consumer keeps polling."""

    def __init__(self, page):
        self.page = page
        self.polls = 0
    async def fetch_file_events_page(self, *, since_ms, after_seq=0, limit):
        self.polls += 1
        return dict(self.page)


def _drive(conn, wiki, *, until, on_gap):
    """Run the consumer until `until()` holds (or 2 s), then let the in-flight
    poll settle and stop. No fixed sleeps: the old
    `asyncio.sleep(0.01 * polls + 0.1)` was a load-sensitive guess."""
    c = WikiConsumer(conn, wiki, poll_interval_s=0.01, poll_limit=10, on_gap=on_gap)

    async def runner():
        await c.start()
        deadline = time.monotonic() + DEADLINE_S
        while time.monotonic() < deadline and not until():
            await asyncio.sleep(0.01)
        await asyncio.sleep(SETTLE_S)
        await c.stop()

    asyncio.run(runner())


def _run(conn, wiki, until=None):
    gaps = []

    async def on_gap(gap):
        gaps.append(gap)

    _drive(conn, wiki, on_gap=on_gap,
           until=until or (lambda: not wiki.pages))
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
    assert gaps[0]["triggered"] is True
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


def test_gap_handler_failure_is_retried_on_the_next_poll(conn):
    # The gap record is persisted before on_gap runs. When on_gap raised, the
    # next poll saw `existing` and never retried, so the automatic verify was
    # lost for good while the cursor stayed behind the horizon.
    set_wiki_cursor(conn, since_ms=1_000, last_seq=3, now_ms=1)
    wiki = SteadyWiki({"events": [], "archive_cutoff_ms": 5_000, "has_archived": True})
    calls = []

    async def on_gap(gap):
        calls.append(dict(gap))
        if len(calls) == 1:
            raise RuntimeError("verify start blew up")

    _drive(conn, wiki, on_gap=on_gap, until=lambda: len(calls) >= 2)

    assert len(calls) == 2, "retried exactly once more, then stopped retrying"
    assert calls[0]["triggered"] is False and calls[1]["triggered"] is False, \
        "the retry sees the record as not-yet-triggered"
    stored = get_cursor_gap(conn)
    assert stored["triggered"] is True and stored["since_ms"] == 1_000


def test_gap_marked_triggered_even_without_a_handler(conn):
    set_wiki_cursor(conn, since_ms=1_000, last_seq=3, now_ms=1)
    wiki = SteadyWiki({"events": [], "archive_cutoff_ms": 5_000, "has_archived": True})
    _drive(conn, wiki, on_gap=None,
           until=lambda: get_cursor_gap(conn) is not None)
    assert get_cursor_gap(conn)["triggered"] is True, \
        "nothing to retry without a handler; do not loop forever"


def test_gap_stays_untriggered_when_the_handler_refuses(conn):
    # main's on_gap returns False when VerifyRunner.start() refused because a
    # verify is already running. That verify may predate the gap, so the gap's
    # own verify has not happened: the record must stay triggered=False and be
    # retried on the next poll instead of being marked done.
    set_wiki_cursor(conn, since_ms=1_000, last_seq=3, now_ms=1)
    wiki = SteadyWiki({"events": [], "archive_cutoff_ms": 5_000, "has_archived": True})
    calls = []

    async def on_gap(gap):
        calls.append(dict(gap))
        return len(calls) >= 2  # refused once, then accepted

    _drive(conn, wiki, on_gap=on_gap, until=lambda: len(calls) >= 2)

    assert len(calls) == 2, "retried after the refusal, then stopped"
    stored = get_cursor_gap(conn)
    assert stored["triggered"] is True and stored["since_ms"] == 1_000
