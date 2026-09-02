def test_get_state_default(client):
    r = client.get("/v1/parser/control/state")
    assert r.status_code == 200
    body = r.json()
    assert body["paused"] is False
    assert body["concurrency"] == 2
    assert body["device"] == "auto"
    # resolution depends on the real host's hardware (cuda / OpenVINO gpu / cpu)
    assert body["resolved_device"] in ("cuda", "gpu", "cpu")
    assert body["ocr_enabled"] is False


def test_set_ocr_toggles(client, tmp_path):
    from parser.main import app_state
    from parser.ocr_installer import FILE_NAMES, model_dir, set_models_dir
    from parser.repo_state import set_ocr_model

    # enabling OCR with no installed/active model is gated
    r = client.post("/v1/parser/control/ocr", json={"enabled": True})
    assert r.status_code == 409

    # fake-install + activate a model (same pattern as
    # test_routes_ocr_models.py's _fake_install)
    set_models_dir(tmp_path / "ocr")
    d = model_dir("ppocr-v4-mobile")
    d.mkdir(parents=True, exist_ok=True)
    for name in FILE_NAMES.values():
        (d / name).write_bytes(b"onnx")
    set_ocr_model(app_state.conn, "ppocr-v4-mobile")

    r = client.post("/v1/parser/control/ocr", json={"enabled": True})
    assert r.status_code == 200
    assert r.json() == {"ocr_enabled": True}
    assert client.get("/v1/parser/control/state").json()["ocr_enabled"] is True

    r = client.post("/v1/parser/control/ocr", json={"enabled": False})
    assert r.status_code == 200
    assert r.json() == {"ocr_enabled": False}


def test_set_ocr_missing_field_returns_422(client):
    r = client.post("/v1/parser/control/ocr", json={})
    assert r.status_code == 422


def test_set_device_valid(client):
    for device in ("cpu", "cuda", "auto"):
        r = client.post("/v1/parser/control/device", json={"device": device})
        assert r.status_code == 200
        body = r.json()
        assert body["device"] == device
        # resolution depends on the real host's hardware (cuda / OpenVINO gpu / cpu)
        assert body["resolved_device"] in ("cuda", "gpu", "cpu")
        # state endpoint reflects the change
        assert client.get("/v1/parser/control/state").json()["device"] == device


def test_set_device_invalid_returns_400(client):
    r = client.post("/v1/parser/control/device", json={"device": "tpu"})
    assert r.status_code == 400


def test_set_device_missing_field_returns_422(client):
    r = client.post("/v1/parser/control/device", json={})
    assert r.status_code == 422


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


def test_set_device_gpu_accepted(client, monkeypatch):
    import parser.hardware as hw
    monkeypatch.setattr(hw, "_has_nvidia_gpu", lambda: False)
    resp = client.post("/v1/parser/control/device", json={"device": "gpu"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["device"] == "gpu"
    assert body["resolved_device"] == "gpu"
