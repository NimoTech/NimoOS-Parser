import asyncio

import pytest

from parser.db import init_db
from parser.repo_jobs import list_jobs
from parser.repo_models import get_wiki_cursor
from parser.wiki_consumer import WikiConsumer


class FakeWiki:
    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = []
    async def fetch_file_events(self, *, since_ms, after_seq=0, limit):
        self.calls.append(since_ms)
        if not self.batches:
            return []
        return self.batches.pop(0)
    async def fetch_file_events_page(self, *, since_ms, after_seq=0, limit):
        return {"events": await self.fetch_file_events(since_ms=since_ms, after_seq=after_seq, limit=limit)}


@pytest.fixture
def conn(tmp_path):
    return init_db(tmp_path / "p.db")


def test_consumer_enqueues_and_advances_cursor(conn):
    batches = [[
        {"id": "e1", "root_id": "r", "path": "/a.md", "op": "create",
         "is_dir": 0, "detected_at": 100},
        {"id": "e2", "root_id": "r", "path": "/b.md", "op": "modify",
         "is_dir": 0, "detected_at": 150},
        {"id": "e3", "root_id": "r", "path": "/c.md", "op": "delete",
         "is_dir": 0, "detected_at": 200},
    ]]
    wiki = FakeWiki(batches)
    c = WikiConsumer(conn, wiki, poll_interval_s=0.01, poll_limit=100)

    async def runner():
        await c.start()
        for _ in range(50):
            jobs = list_jobs(conn, status="pending", limit=10)
            if len(jobs) >= 3:
                break
            await asyncio.sleep(0.02)
        await c.stop()

    asyncio.run(runner())
    jobs = list_jobs(conn, status="pending", limit=10)
    paths = sorted(j["path"] for j in jobs)
    assert paths == ["/a.md", "/b.md", "/c.md"]
    ops = sorted(j["op"] for j in jobs)
    assert ops == ["delete", "index", "index"]
    assert get_wiki_cursor(conn) == (200, 0)


def test_consumer_skips_directory_events(conn):
    wiki = FakeWiki([[
        {"id": "e1", "root_id": "r", "path": "/dir", "op": "create",
         "is_dir": 1, "detected_at": 100},
    ]])
    c = WikiConsumer(conn, wiki, poll_interval_s=0.01, poll_limit=100)

    async def runner():
        await c.start()
        for _ in range(30):
            await asyncio.sleep(0.02)
            if get_wiki_cursor(conn) == (100, 0):
                break
        await c.stop()

    asyncio.run(runner())
    assert list_jobs(conn, status="pending", limit=10) == []
    assert get_wiki_cursor(conn) == (100, 0)


def test_ingest_advances_cursor_to_last_event_seq(conn):
    """When a batch of same-millisecond events arrives paginated, the cursor must advance with seq, and must not skip subsequent pages."""
    from parser.repo_models import get_wiki_cursor
    from parser.wiki_consumer import WikiConsumer

    consumer = WikiConsumer(conn, wiki=None)
    ts = 1753000000000
    page = [
        {"root_id": "r", "path": f"/f{i}", "op": "create",
         "is_dir": False, "detected_at": ts, "seq": 100 + i}
        for i in range(3)
    ]
    consumer._ingest(page)
    since, seq = get_wiki_cursor(conn)
    assert since == ts
    assert seq == 102


def test_cursor_migration_adds_last_seq(tmp_path):
    """An old DB (without the last_seq column) auto-migrates on open, defaulting to 0."""
    import sqlite3
    from parser.db import init_db

    p = tmp_path / "old.db"
    conn = sqlite3.connect(str(p))
    conn.execute("CREATE TABLE wiki_cursor (id INTEGER PRIMARY KEY CHECK(id = 1), "
                 "since_ms INTEGER NOT NULL DEFAULT 0, updated_at INTEGER NOT NULL)")
    conn.execute("INSERT INTO wiki_cursor(id, since_ms, updated_at) VALUES (1, 42, 0)")
    conn.commit()
    conn.close()

    conn = init_db(p)
    row = conn.execute("SELECT since_ms, last_seq FROM wiki_cursor WHERE id = 1").fetchone()
    assert row["since_ms"] == 42
    assert row["last_seq"] == 0


def test_consumer_enqueues_retire_root_at_top_priority(conn):
    wiki = FakeWiki([[{"id": "e1", "root_id": "gone", "path": "", "op": "root_removed",
                       "is_dir": 0, "detected_at": 100, "seq": 7}]])
    c = WikiConsumer(conn, wiki, poll_interval_s=0.01, poll_limit=100)

    async def runner():
        await c.start()
        for _ in range(50):
            if list_jobs(conn, status="pending", limit=10):
                break
            await asyncio.sleep(0.02)
        await c.stop()

    asyncio.run(runner())
    jobs = list_jobs(conn, status="pending", limit=10)
    assert len(jobs) == 1
    assert (jobs[0]["op"], jobs[0]["root_id"], jobs[0]["path"], jobs[0]["priority"]) == \
        ("retire_root", "gone", "", 50)
    assert get_wiki_cursor(conn) == (100, 7)
