"""notes 路由三端点:embed 与 qstore 全部打桩,验证契约与隔离参数。"""
import pytest
from fastapi.testclient import TestClient


class _FakeQStore:
    def __init__(self):
        self.upserted = []
        self.deleted = []

    def upsert_notes(self, points):
        self.upserted.extend(points)

    def query_notes(self, user_id, dense, limit=10, statuses=None):
        self.query_args = (user_id, limit, statuses)
        return [{"note_id": "n1", "chunk_no": 0, "text": "t",
                 "type": "note", "status": "curated", "updated_at": 1,
                 "score": 0.8}]

    def delete_note(self, user_id, note_id):
        self.deleted.append((user_id, note_id))


@pytest.fixture()
def client(monkeypatch):
    from parser.routes import notes as notes_routes
    from parser import main as parser_main
    fake = _FakeQStore()
    monkeypatch.setattr(parser_main.app_state, "qstore", fake, raising=False)
    monkeypatch.setattr(
        notes_routes, "_embed_batch",
        lambda texts: [{"dense": [0.0] * 4,
                        "sparse": {"indices": [1], "values": [1.0]}}
                       for _ in texts])
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(notes_routes.router)
    return TestClient(app), fake


def test_upsert_deletes_stale_then_inserts(client):
    c, fake = client
    r = c.post("/v1/parser/notes/upsert", json={
        "user_id": "1", "note_id": "n1", "note_type": "note",
        "status": "draft", "created_by": "agent", "updated_at": 5,
        "chunks": [{"chunk_no": 0, "text": "hello"},
                   {"chunk_no": 1, "text": "world"}],
    })
    assert r.status_code == 200 and r.json()["upserted"] == 2
    assert fake.deleted == [("1", "n1")]          # 先删旧
    assert len(fake.upserted) == 2
    pl = fake.upserted[0]["payload"]
    assert pl["user_id"] == "1" and pl["note_id"] == "n1"
    assert pl["type"] == "note" and pl["status"] == "draft"


def test_upsert_requires_user_id(client):
    c, _ = client
    r = c.post("/v1/parser/notes/upsert", json={
        "user_id": "", "note_id": "n1", "note_type": "note",
        "status": "draft", "created_by": "agent", "updated_at": 5,
        "chunks": []})
    assert r.status_code == 400


def test_query_passes_user_and_statuses(client):
    c, fake = client
    r = c.post("/v1/parser/notes/query", json={
        "user_id": "1", "query": "hello", "top_k": 3,
        "statuses": ["curated"]})
    assert r.status_code == 200
    assert fake.query_args == ("1", 3, ["curated"])
    assert r.json()["hits"][0]["note_id"] == "n1"


def test_delete(client):
    c, fake = client
    r = c.post("/v1/parser/notes/delete",
               json={"user_id": "1", "note_id": "n1"})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert ("1", "n1") in fake.deleted


def test_upsert_point_ids_are_user_scoped(client):
    c, fake = client
    body = {"note_id": "n1", "note_type": "note", "status": "draft",
            "created_by": "agent", "updated_at": 5,
            "chunks": [{"chunk_no": 0, "text": "hello"}]}
    c.post("/v1/parser/notes/upsert", json={**body, "user_id": "1"})
    c.post("/v1/parser/notes/upsert", json={**body, "user_id": "2"})
    ids = [p["id"] for p in fake.upserted]
    assert len(ids) == 2 and ids[0] != ids[1]   # 不同用户同 note_id 绝不共享 point id
