from parser.db import init_db
from parser.repo_jobs import enqueue_job
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


def test_stats_includes_rate_and_eta(client, monkeypatch, tmp_path):
    conn = init_db(tmp_path / "p.db")
    for i in range(6):
        enqueue_job(conn, root_id="r", path=f"/f{i}", op="index", now_ms=100)

    class FakeQ:
        def count_vectors(self):
            return {"text": 0, "visual": 0}

    class FakePool:
        def throughput(self):
            return {"done_last_10m": 20, "rate_per_min": 2.0}

    monkeypatch.setattr("parser.routes.stats.get_conn", lambda: conn)
    monkeypatch.setattr("parser.routes.stats.get_qstore", lambda: FakeQ())
    monkeypatch.setattr("parser.routes.stats.get_wiki_cursor_val",
                        lambda c: 0)
    monkeypatch.setattr("parser.routes.stats.get_pool", lambda: FakePool())

    r = client.get("/v1/parser/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["done_last_10m"] == 20
    assert body["rate_per_min"] == 2.0
    assert body["eta_s"] == int(6 * 60 / 2.0)


def test_stats_eta_null_when_no_rate_or_no_pending(client, monkeypatch, tmp_path):
    conn = init_db(tmp_path / "p.db")

    class FakeQ:
        def count_vectors(self):
            return {"text": 0, "visual": 0}

    monkeypatch.setattr("parser.routes.stats.get_conn", lambda: conn)
    monkeypatch.setattr("parser.routes.stats.get_qstore", lambda: FakeQ())
    monkeypatch.setattr("parser.routes.stats.get_wiki_cursor_val",
                        lambda c: 0)

    # rate == 0, pending > 0 -> eta_s is None
    enqueue_job(conn, root_id="r", path="/f0", op="index", now_ms=100)

    class ZeroRatePool:
        def throughput(self):
            return {"done_last_10m": 0, "rate_per_min": 0.0}

    monkeypatch.setattr("parser.routes.stats.get_pool", lambda: ZeroRatePool())
    r = client.get("/v1/parser/stats")
    assert r.status_code == 200
    assert r.json()["eta_s"] is None

    # rate > 0, pending == 0 -> eta_s is None
    conn2 = init_db(tmp_path / "p2.db")
    monkeypatch.setattr("parser.routes.stats.get_conn", lambda: conn2)

    class PositiveRatePool:
        def throughput(self):
            return {"done_last_10m": 5, "rate_per_min": 3.0}

    monkeypatch.setattr("parser.routes.stats.get_pool", lambda: PositiveRatePool())
    r = client.get("/v1/parser/stats")
    assert r.status_code == 200
    assert r.json()["eta_s"] is None
