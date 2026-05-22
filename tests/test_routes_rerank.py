def test_rerank_returns_scores(client, monkeypatch):
    class FakeRR:
        version = "bge-reranker-v2-m3/v1"
        def rerank(self, query, candidates):
            return [{"id": c["id"], "score": 1.0 - i * 0.1}
                    for i, c in enumerate(candidates)]

    from parser.routes import rerank as r
    monkeypatch.setattr(r, "get_reranker", lambda: FakeRR())

    resp = client.post("/v1/parser/rerank", json={
        "model": "bge-reranker-v2-m3",
        "query": "q",
        "candidates": [
            {"id": "a", "text": "alpha"},
            {"id": "b", "text": "beta"},
        ],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == "bge-reranker-v2-m3/v1"
    assert [s["id"] for s in body["scores"]] == ["a", "b"]
    assert body["scores"][0]["score"] == 1.0


def test_rerank_caps_candidates(client):
    candidates = [{"id": str(i), "text": "t"} for i in range(200)]
    resp = client.post("/v1/parser/rerank", json={
        "model": "bge-reranker-v2-m3",
        "query": "q",
        "candidates": candidates,
    })
    assert resp.status_code == 400


def test_rerank_unknown_model(client):
    resp = client.post("/v1/parser/rerank", json={
        "model": "nope", "query": "q", "candidates": [],
    })
    assert resp.status_code == 400
