import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def reindex_ctx(tmp_path):
    from parser.main import app_state, create_app
    from parser.db import init_db

    class FakeQstore:
        def __init__(self):
            self.tombstoned = []
        def tombstone_file(self, *, file_id, tombstoned_at):
            self.tombstoned.append(file_id)

    conn = init_db(tmp_path / "p.db")
    prev_conn = app_state.conn
    prev_qstore = app_state.qstore
    app_state.conn = conn
    app_state.qstore = FakeQstore()
    app = create_app()
    try:
        with TestClient(app) as c:
            yield c, conn
    finally:
        app_state.conn = prev_conn
        app_state.qstore = prev_qstore


def _seed(conn, *, fid, root="r1", path=None, mime="text/plain"):
    from parser.repo_records import upsert_file_record, upsert_file_path
    if path is None:
        path = f"/p/{fid}"
    upsert_file_record(
        conn, file_id=fid, sha256_full="sha-" + fid, size=10, mime=mime,
        modalities_done={}, parser_version="parser/0.2.0", indexed_at=100,
    )
    upsert_file_path(conn, root_id=root, path=path, file_id=fid, mtime_ms=0)


def test_reindex_file_ids_mode_happy_path(reindex_ctx):
    client, conn = reindex_ctx
    _seed(conn, fid="a")
    _seed(conn, fid="b")
    r = client.post("/v1/parser/files/reindex",
                    json={"file_ids": ["a", "b"], "reason": "test"})
    assert r.status_code == 200
    body = r.json()
    assert body["tombstoned"] == 2
    assert body["queued"] == 2
    assert len(body["job_ids"]) == 2
    assert body["skipped"] == []


def test_reindex_returns_skipped_for_missing_id(reindex_ctx):
    client, conn = reindex_ctx
    r = client.post("/v1/parser/files/reindex", json={"file_ids": ["ghost"]})
    assert r.status_code == 200
    assert r.json()["skipped"] == [{"file_id": "ghost", "reason": "not_found"}]


def test_reindex_rejects_empty_file_ids(reindex_ctx):
    client, _ = reindex_ctx
    r = client.post("/v1/parser/files/reindex", json={"file_ids": []})
    assert r.status_code == 400


def test_reindex_rejects_both_modes(reindex_ctx):
    client, _ = reindex_ctx
    r = client.post("/v1/parser/files/reindex",
                    json={"file_ids": ["a"], "filter": {"root_id": "r1"}})
    assert r.status_code == 400


def test_reindex_rejects_neither_mode(reindex_ctx):
    client, _ = reindex_ctx
    r = client.post("/v1/parser/files/reindex", json={"reason": "test"})
    assert r.status_code == 400


def test_reindex_filter_mode_root_id(reindex_ctx):
    client, conn = reindex_ctx
    _seed(conn, fid="a", root="r1")
    _seed(conn, fid="b", root="r1")
    _seed(conn, fid="c", root="r2")
    r = client.post("/v1/parser/files/reindex",
                    json={"filter": {"root_id": "r1"}, "reason": "rebuild root r1"})
    assert r.status_code == 200
    assert r.json()["tombstoned"] == 2


def test_reindex_too_many_file_ids_400(reindex_ctx):
    client, _ = reindex_ctx
    r = client.post("/v1/parser/files/reindex",
                    json={"file_ids": [f"f{i}" for i in range(501)]})
    assert r.status_code == 400
