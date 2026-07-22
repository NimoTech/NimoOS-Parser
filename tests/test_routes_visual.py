import json

import pytest


@pytest.fixture
def img(tmp_path, monkeypatch):
    d = tmp_path / "thumbs"
    d.mkdir()
    p = d / "a1-small.jpg"
    p.write_bytes(b"\xff\xd8fake")
    monkeypatch.setenv("PARSER_VISUAL_ALLOWED_DIRS", str(d))
    return p


def _body(p, **over):
    b = {"source": "photos", "asset_id": "a1", "image_path": str(p),
         "mime": "image/jpeg", "meta": {"taken_at": "2025-06-01"}}
    b.update(over)
    return b


def test_ingest_enqueues(client, img):
    r = client.post("/v1/parser/visual/ingest", json=_body(img))
    assert r.status_code == 202
    from parser.main import app_state
    row = app_state.conn.execute(
        "SELECT * FROM parse_jobs WHERE op='visual_ingest'").fetchone()
    assert row["root_id"] == "photos"
    assert row["path"] == str(img)
    assert row["priority"] == 200
    payload = json.loads(row["sub_modality"])
    assert payload["asset_id"] == "a1" and payload["mime"] == "image/jpeg"


def test_ingest_rejects_path_outside_allowlist(client, img, tmp_path):
    evil = tmp_path / "etc-passwd"
    evil.write_text("x")
    r = client.post("/v1/parser/visual/ingest",
                    json=_body(img, image_path=str(evil)))
    assert r.status_code == 400


def test_ingest_rejects_missing_file(client, img):
    r = client.post("/v1/parser/visual/ingest",
                    json=_body(img, image_path=str(img) + ".nope"))
    assert r.status_code == 400


def test_ingest_rejects_traversal(client, img):
    sneaky = str(img.parent) + "/../outside.jpg"
    r = client.post("/v1/parser/visual/ingest",
                    json=_body(img, image_path=sneaky))
    assert r.status_code == 400


def test_delete_without_pipeline_503(client):
    r = client.delete("/v1/parser/visual/assets/photos/a1")
    assert r.status_code == 503
