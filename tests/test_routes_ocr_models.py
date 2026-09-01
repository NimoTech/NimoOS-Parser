# tests/test_routes_ocr_models.py
"""Route-level state machine via FastAPI TestClient; installer fs under tmp."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from parser import ocr_installer
from parser.db import init_db
from parser.ocr_installer import FILE_NAMES, model_dir, set_models_dir
from parser.repo_state import get_state
from parser.routes import control, ocr_models


@pytest.fixture()
def client(tmp_path, monkeypatch):
    conn = init_db(tmp_path / "t.db")
    set_models_dir(tmp_path / "ocr")
    ocr_installer._progress.clear()
    from parser.main import app_state
    monkeypatch.setattr(app_state, "conn", conn)
    app = FastAPI()
    app.include_router(ocr_models.router)
    app.include_router(control.router)
    return TestClient(app)


def _fake_install(model_id):
    d = model_dir(model_id)
    d.mkdir(parents=True, exist_ok=True)
    for name in FILE_NAMES.values():
        (d / name).write_bytes(b"onnx")


def test_list_shows_catalog(client):
    body = client.get("/v1/parser/ocr/models").json()
    ids = [m["id"] for m in body["models"]]
    assert "ppocr-v4-mobile" in ids
    row = body["models"][0]
    assert row["installed"] is False and row["active"] is False


def test_install_unknown_404(client):
    assert client.post("/v1/parser/ocr/models/nope/install").status_code == 404


def test_activate_requires_installed(client):
    r = client.post("/v1/parser/ocr/models/ppocr-v4-mobile/activate")
    assert r.status_code == 409


def test_activate_and_delete_guards(client):
    _fake_install("ppocr-v4-mobile")
    r = client.post("/v1/parser/ocr/models/ppocr-v4-mobile/activate")
    assert r.status_code == 200
    from parser.main import app_state
    assert get_state(app_state.conn)["ocr_model"] == "ppocr-v4-mobile"
    # active model refuses deletion
    assert client.delete("/v1/parser/ocr/models/ppocr-v4-mobile").status_code == 409


def test_ocr_enable_gated_on_installed_model(client):
    r = client.post("/v1/parser/control/ocr", json={"enabled": True})
    assert r.status_code == 409
    _fake_install("ppocr-v4-mobile")
    client.post("/v1/parser/ocr/models/ppocr-v4-mobile/activate")
    r = client.post("/v1/parser/control/ocr", json={"enabled": True})
    assert r.status_code == 200


def test_list_unknown_profile_size_note_blank(client, monkeypatch):
    def fake_entries():
        return [{"id": "mystery", "name": "Mystery Model", "langs": "en",
                 "profile": "quantum", "recommended": False, "files": {}}]

    monkeypatch.setattr(ocr_models, "entries", fake_entries)
    r = client.get("/v1/parser/ocr/models")
    assert r.status_code == 200
    row = next(m for m in r.json()["models"] if m["id"] == "mystery")
    assert row["size_note"] == ""


def test_activate_rolls_back_on_failure(client, monkeypatch):
    _fake_install("ppocr-v4-mobile")

    def boom(conn, model_id):
        raise RuntimeError("boom")

    strict_client = TestClient(client.app, raise_server_exceptions=False)
    # Scoped to this `with` block so it doesn't also undo the `client`
    # fixture's app_state.conn patch (both share the outer `monkeypatch`).
    with monkeypatch.context() as m:
        m.setattr(ocr_models, "set_ocr_model", boom)
        r = strict_client.post("/v1/parser/ocr/models/ppocr-v4-mobile/activate")
    assert r.status_code >= 500
    from parser.main import app_state
    assert app_state.conn.in_transaction is False

    r2 = client.post("/v1/parser/ocr/models/ppocr-v4-mobile/activate")
    assert r2.status_code == 200
