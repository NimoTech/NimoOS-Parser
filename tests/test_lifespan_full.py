"""Tests for full-lifecycle startup (Task 27).

These tests run with skip_workers=False so the `_lifespan` actually wires up
sqlite + (optional) qdrant + (optional) wiki_client + consumer + worker_pool +
gc_task. Qdrant is not running in the sandbox; we assert the lifespan handles
this gracefully by leaving `app_state.qstore = None`.
"""

import os
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_app_state():
    # Reset module-level singleton between tests (each test gets clean state)
    from parser.main import app_state
    app_state.conn = None
    app_state.qstore = None
    app_state.settings = None
    app_state.consumer = None
    app_state.worker_pool = None
    app_state.wiki_client = None
    app_state.gc_task = None
    yield


def _set_env(monkeypatch, tmp_path: Path) -> None:
    # Point parser at writable tmp paths, unreachable Qdrant (so we exercise
    # the graceful-degradation path), no wiki discovery.
    monkeypatch.setenv("PARSER_DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setenv("PARSER_WIKI_DISCOVERY_PATH",
                       str(tmp_path / "wiki.url"))
    monkeypatch.setenv("PARSER_QDRANT_URL", "http://127.0.0.1:1")
    # Long enough that GC doesn't fire during the test.
    monkeypatch.setenv("PARSER_GC_INTERVAL_S", "3600")
    monkeypatch.delenv("PARSER_DISCOVERY_FILE", raising=False)
    monkeypatch.delenv("PARSER_BIND_ADDR", raising=False)


def test_full_lifecycle_inits_db_and_registers_models(tmp_path, monkeypatch):
    _set_env(monkeypatch, tmp_path)
    from parser.main import create_app, app_state
    app = create_app(skip_workers=False)
    with TestClient(app) as c:
        # DB file should exist after startup
        db_file = tmp_path / "data" / "parser.db"
        assert db_file.exists()
        assert app_state.conn is not None
        # Active models registered
        rows = app_state.conn.execute(
            "SELECT name, version, active FROM model_versions WHERE active = 1 "
            "ORDER BY name"
        ).fetchall()
        names = {r["name"] for r in rows}
        assert names == {"bge-m3", "bge-reranker-v2-m3"}
        # Service responds
        r = c.get("/healthz")
        assert r.status_code == 200
    # After shutdown, conn cleared
    assert app_state.conn is None


def test_full_lifecycle_degrades_without_qdrant(tmp_path, monkeypatch):
    _set_env(monkeypatch, tmp_path)
    from parser.main import create_app, app_state
    app = create_app(skip_workers=False)
    with TestClient(app) as c:
        # Qdrant unreachable → qstore stays None, worker_pool never started
        assert app_state.qstore is None
        assert app_state.worker_pool is None


def test_full_lifecycle_skips_consumer_without_wiki_url(tmp_path, monkeypatch):
    _set_env(monkeypatch, tmp_path)
    # No wiki.url file at the configured path
    from parser.main import create_app, app_state
    app = create_app(skip_workers=False)
    with TestClient(app) as c:
        assert app_state.wiki_client is None
        assert app_state.consumer is None


def test_full_lifecycle_reads_wiki_url(tmp_path, monkeypatch):
    _set_env(monkeypatch, tmp_path)
    wiki_url_file = tmp_path / "wiki.url"
    wiki_url_file.write_text("http://127.0.0.1:8080\n")
    from parser.main import create_app, app_state
    app = create_app(skip_workers=False)
    with TestClient(app) as c:
        assert app_state.wiki_client is not None
        # Note: worker_pool stays None because qstore is None
        # (Qdrant unreachable in this test); consumer is independent
        assert app_state.consumer is not None


def test_skip_workers_does_not_init_anything(tmp_path, monkeypatch):
    _set_env(monkeypatch, tmp_path)
    from parser.main import create_app, app_state
    app = create_app(skip_workers=True)
    with TestClient(app) as c:
        # No DB, no qstore, no consumer, no pool — skip_workers means skip all
        assert app_state.conn is None
        assert app_state.qstore is None
        assert app_state.consumer is None
        assert app_state.worker_pool is None
        # DB file NOT created
        assert not (tmp_path / "data" / "parser.db").exists()
