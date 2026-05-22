from fastapi.testclient import TestClient


def test_embed_text_via_bge_m3(client, monkeypatch):
    fake_out = [{"dense": [0.1] * 1024,
                 "sparse": {"indices": [1, 2], "values": [0.5, 0.3]}}]

    class FakeBGE:
        version = "bge-m3/v1"
        dim = 1024
        def embed_text(self, texts):
            return fake_out

    from parser.routes import embed as embed_route
    monkeypatch.setattr(embed_route, "get_bge_m3", lambda: FakeBGE())

    r = client.post("/v1/parser/embed", json={
        "model": "bge-m3", "input_type": "text", "text": "hello",
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["dense"]) == 1024
    assert body["sparse"]["indices"] == [1, 2]
    assert body["dim"] == 1024
    assert body["model_version"] == "bge-m3/v1"


def test_embed_rejects_unknown_model(client):
    r = client.post("/v1/parser/embed", json={
        "model": "nope", "input_type": "text", "text": "x",
    })
    assert r.status_code == 400


def test_embed_rejects_missing_input(client):
    r = client.post("/v1/parser/embed", json={
        "model": "bge-m3", "input_type": "text",
    })
    assert r.status_code == 400
