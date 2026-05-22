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
    async def fetch_file_events(self, *, since_ms, limit):
        self.calls.append(since_ms)
        if not self.batches:
            return []
        return self.batches.pop(0)


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
    assert get_wiki_cursor(conn) == 200


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
            if get_wiki_cursor(conn) == 100:
                break
        await c.stop()

    asyncio.run(runner())
    assert list_jobs(conn, status="pending", limit=10) == []
    assert get_wiki_cursor(conn) == 100
