import parser.routes.agent_memory as am
from parser.main import app_state


class FakeQStore:
    def __init__(self):
        self.upserted = None
        self.queried = None
        self.hits = [{"text": "t", "session_id": "s1", "chunk_no": 0,
                      "created_at": 1, "score": 0.9}]

    def upsert_agent_memory(self, points):
        self.upserted = list(points)

    def query_agent_memory(self, user_id, dense, limit=5):
        self.queried = {"user_id": user_id, "dense": dense, "limit": limit}
        return self.hits


def _fake_qstore(monkeypatch):
    fake = FakeQStore()
    monkeypatch.setattr(app_state, "qstore", fake, raising=False)
    monkeypatch.setattr(am, "_embed_dense_batch",
                        lambda texts: [[0.1] * 1024 for _ in texts])
    return fake


def test_upsert_indexes_chunks(client, monkeypatch):
    fake = _fake_qstore(monkeypatch)
    r = client.post("/v1/parser/agent-memory/upsert", json={
        "user_id": "u1", "session_id": "s1",
        "chunks": [{"chunk_no": 0, "text": "hi", "created_at": 1}]})
    assert r.status_code == 200 and r.json() == {"upserted": 1}
    p = fake.upserted[0]
    assert p["payload"] == {"user_id": "u1", "session_id": "s1",
                            "chunk_no": 0, "text": "hi", "created_at": 1}
    assert len(p["dense"]) == 1024 and isinstance(p["id"], str)


def test_upsert_id_is_deterministic(client, monkeypatch):
    fake = _fake_qstore(monkeypatch)
    body = {"user_id": "u1", "session_id": "s1",
            "chunks": [{"chunk_no": 0, "text": "hi", "created_at": 1}]}
    client.post("/v1/parser/agent-memory/upsert", json=body)
    id1 = fake.upserted[0]["id"]
    client.post("/v1/parser/agent-memory/upsert", json=body)
    id2 = fake.upserted[0]["id"]
    assert id1 == id2   # same (user,session,chunk) → same point id (idempotent)


def test_upsert_rejects_empty_user_id(client, monkeypatch):
    _fake_qstore(monkeypatch)
    r = client.post("/v1/parser/agent-memory/upsert", json={
        "user_id": "", "session_id": "s1",
        "chunks": [{"chunk_no": 0, "text": "hi", "created_at": 1}]})
    assert r.status_code == 400


def test_query_passes_user_id_and_returns_hits(client, monkeypatch):
    fake = _fake_qstore(monkeypatch)
    r = client.post("/v1/parser/agent-memory/query",
                    json={"user_id": "u1", "query": "career", "top_k": 3})
    assert r.status_code == 200
    assert r.json() == {"hits": fake.hits}
    assert fake.queried["user_id"] == "u1" and fake.queried["limit"] == 3


def test_query_rejects_empty_user_id(client, monkeypatch):
    _fake_qstore(monkeypatch)
    r = client.post("/v1/parser/agent-memory/query",
                    json={"user_id": "", "query": "x"})
    assert r.status_code == 400
