from parser.db import init_db
from parser.repo_models import register_model, set_active


def test_stats_returns_counts(client, monkeypatch, tmp_path):
    conn = init_db(tmp_path / "p.db")
    register_model(conn, name="bge-m3", version="v1", modality="text",
                   dim=1024, registered_at=100)

    class FakeQ:
        def count_vectors(self):
            return {"text": 42, "visual": 17}

    monkeypatch.setattr("parser.routes.stats.get_conn", lambda: conn)
    monkeypatch.setattr("parser.routes.stats.get_qstore", lambda: FakeQ())
    monkeypatch.setattr("parser.routes.stats.get_wiki_cursor_val",
                        lambda c: 9999)

    r = client.get("/v1/parser/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total_vectors_text"] == 42
    assert body["total_vectors_visual"] == 17
    assert body["last_cursor_ms"] == 9999
    assert any(m["name"] == "bge-m3" for m in body["models"])


def test_models_route(client, monkeypatch, tmp_path):
    conn = init_db(tmp_path / "p.db")
    register_model(conn, name="bge-m3", version="v1", modality="text",
                   dim=1024, registered_at=100)
    monkeypatch.setattr("parser.routes.models.get_conn", lambda: conn)
    r = client.get("/v1/parser/models")
    assert r.status_code == 200
    body = r.json()
    names = {m["name"] for m in body["models"]}
    assert "bge-m3" in names
