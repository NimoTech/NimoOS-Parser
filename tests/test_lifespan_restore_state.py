import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from parser.db import init_db
from parser.repo_state import set_paused, set_concurrency


def test_startup_restores_paused_and_concurrency(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "parser.db"
    conn = init_db(db_path)
    set_paused(conn, True)
    set_concurrency(conn, 4)
    conn.close()

    monkeypatch.setenv("PARSER_DATA_PATH", str(tmp_path))
    from parser.main import create_app, app_state
    app = create_app(skip_workers=True)
    with TestClient(app) as c:
        body = c.get("/v1/parser/control/state").json()
        assert body == {"paused": True, "concurrency": 4}


@pytest.mark.asyncio
async def test_startup_applies_state_to_worker_pool(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "parser.db"
    conn = init_db(db_path)
    set_paused(conn, True)
    set_concurrency(conn, 1)
    conn.close()

    monkeypatch.setenv("PARSER_DATA_PATH", str(tmp_path))
    from parser.main import create_app, app_state
    app = create_app(skip_workers=False)
    async with app.router.lifespan_context(app):
        pool = app_state.worker_pool
        # Parser's worker_pool needs Qdrant reachable; if the environment doesn't have it, pool is None, skip
        if pool is None:
            pytest.skip("worker_pool not started (qdrant unavailable in test env)")
        assert pool.concurrency == 1
        assert pool._run_event.is_set() is False  # paused
