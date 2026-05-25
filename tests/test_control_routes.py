def test_get_state_default(client):
    r = client.get("/v1/parser/control/state")
    assert r.status_code == 200
    body = r.json()
    assert body == {"paused": False, "concurrency": 2}


def test_pause_then_state_shows_paused(client):
    r = client.post("/v1/parser/control/pause")
    assert r.status_code == 200
    assert r.json() == {"paused": True}
    assert client.get("/v1/parser/control/state").json()["paused"] is True


def test_resume_clears_paused(client):
    client.post("/v1/parser/control/pause")
    r = client.post("/v1/parser/control/resume")
    assert r.status_code == 200
    assert r.json() == {"paused": False}
    # Verify via GET /state too
    assert client.get("/v1/parser/control/state").json()["paused"] is False


def test_set_concurrency_valid_values(client):
    for n in (1, 2, 4):
        r = client.post("/v1/parser/control/concurrency", json={"n": n})
        assert r.status_code == 200
        assert r.json() == {"concurrency": n}
        assert client.get("/v1/parser/control/state").json()["concurrency"] == n


def test_set_concurrency_invalid_returns_400(client):
    r = client.post("/v1/parser/control/concurrency", json={"n": 3})
    assert r.status_code == 400


def test_set_concurrency_missing_body_returns_422(client):
    r = client.post("/v1/parser/control/concurrency", json={})
    assert r.status_code == 422


def test_pause_is_idempotent(client):
    client.post("/v1/parser/control/pause")
    r = client.post("/v1/parser/control/pause")
    assert r.status_code == 200
    assert r.json() == {"paused": True}


def test_resume_is_idempotent(client):
    r = client.post("/v1/parser/control/resume")
    assert r.status_code == 200
    assert r.json() == {"paused": False}
    r = client.post("/v1/parser/control/resume")
    assert r.status_code == 200
