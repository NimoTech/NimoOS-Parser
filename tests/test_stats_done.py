from parser.db import init_db


def test_stats_includes_done_count(client, monkeypatch, tmp_path):
    conn = init_db(tmp_path / "p.db")

    class FakeQ:
        def count_vectors(self):
            return {"text": 0, "visual": 0}

    monkeypatch.setattr("parser.routes.stats.get_conn", lambda: conn)
    monkeypatch.setattr("parser.routes.stats.get_qstore", lambda: FakeQ())
    monkeypatch.setattr("parser.routes.stats.get_wiki_cursor_val",
                        lambda c: 0)

    r = client.get("/v1/parser/stats")
    assert r.status_code == 200
    body = r.json()
    assert "done" in body["queue_depth"]
    assert isinstance(body["queue_depth"]["done"], int)
