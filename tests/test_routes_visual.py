import json

import pytest


@pytest.fixture
def captions_ctx(tmp_path):
    # Following test_routes_files_reindex.py's FakeQstore injection convention:
    # build conn/qstore first, then create_app; skip_workers mode never
    # touches app_state.qstore, so the test stays fully in control throughout.
    from parser.main import app_state, create_app
    from parser.db import init_db
    from fastapi.testclient import TestClient

    class FakeQstore:
        def __init__(self):
            self.calls = []

        def scroll_captions(self, source, limit, offset):
            self.calls.append((source, limit, offset))
            if offset is None:
                return (
                    [{"file_id": "photos:a1", "text": "A dog.", "mtime_ms": 1}],
                    "cursor2",
                )
            if offset == "cursor2":
                return (
                    [{"file_id": "photos:a2", "text": "A cat.", "mtime_ms": 2}],
                    None,
                )
            return ([], None)

    conn = init_db(tmp_path / "captions.db")
    prev_conn = app_state.conn
    prev_qstore = app_state.qstore
    app_state.conn = conn
    fake = FakeQstore()
    app_state.qstore = fake
    app = create_app(skip_workers=True)
    try:
        with TestClient(app) as c:
            yield c, fake
    finally:
        app_state.conn = prev_conn
        app_state.qstore = prev_qstore
        conn.close()


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


def test_delete_qdrant_midflight_failure_503(client):
    # Qdrant went down transiently after startup: delete_asset raising an
    # exception should be wrapped as a 503 (retryable semantics), not
    # propagate as a raw 500.
    from parser.main import app_state

    class _BrokenPipeline:
        def delete_asset(self, *, source, asset_id):
            raise RuntimeError("qdrant connection refused")

    prev = getattr(app_state, "visual_pipeline", None)
    app_state.visual_pipeline = _BrokenPipeline()
    try:
        r = client.delete("/v1/parser/visual/assets/photos/a1")
        assert r.status_code == 503
        assert "retry" in r.json()["detail"]
    finally:
        app_state.visual_pipeline = prev


def test_captions_export_basic(captions_ctx):
    client_visual, fake = captions_ctx
    r1 = client_visual.get(
        "/v1/parser/visual/captions?source=photos&limit=512")
    assert r1.status_code == 200
    body = r1.json()
    assert body["items"] == [{"asset_id": "a1", "text": "A dog.", "mtime_ms": 1}]
    assert body["next_offset"] == "cursor2"

    r2 = client_visual.get(
        "/v1/parser/visual/captions?source=photos&offset=cursor2")
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["items"][0]["asset_id"] == "a2"
    assert body2["next_offset"] is None

    # the limit/offset scroll_captions receives passes through unchanged from the query params
    assert fake.calls[0] == ("photos", 512, None)
    assert fake.calls[1] == ("photos", 512, "cursor2")


def test_captions_export_strips_only_matching_prefix(captions_ctx):
    client_visual, fake = captions_ctx

    def scroll_captions(source, limit, offset):
        return (
            [
                {"file_id": "photos:a1", "text": "A dog.", "mtime_ms": 1},
                {"file_id": "wiki:xyz", "text": "irrelevant", "mtime_ms": 9},
            ],
            None,
        )

    fake.scroll_captions = scroll_captions
    r = client_visual.get("/v1/parser/visual/captions?source=photos")
    assert r.status_code == 200
    body = r.json()
    assert body["items"] == [{"asset_id": "a1", "text": "A dog.", "mtime_ms": 1}]


def test_captions_export_empty_offset_normalizes_to_none(captions_ctx):
    # A literal contract example, `offset=` (an explicit empty string) - FastAPI
    # binds it to "" rather than None, so the handler must normalize it,
    # otherwise passing it through to qdrant scroll fails to parse and gets wrapped as a 503.
    client_visual, fake = captions_ctx
    r = client_visual.get("/v1/parser/visual/captions?source=photos&offset=")
    assert r.status_code == 200
    body = r.json()
    assert body["items"] == [{"asset_id": "a1", "text": "A dog.", "mtime_ms": 1}]
    assert body["next_offset"] == "cursor2"
    assert fake.calls[-1] == ("photos", 512, None)


def test_captions_export_limit_clamped(captions_ctx):
    client_visual, fake = captions_ctx
    client_visual.get("/v1/parser/visual/captions?source=photos&limit=99999")
    assert fake.calls[-1][1] == 1024
    client_visual.get("/v1/parser/visual/captions?source=photos&limit=0")
    assert fake.calls[-1][1] == 1


def test_captions_export_qdrant_down_503(client, monkeypatch):
    # Don't rely on app_state.qstore's environment default (under the full
    # suite, an earlier-run test may have set it to non-None and left it
    # uncleaned) - explicitly monkeypatch it to None, auto-restored, to eliminate ordering dependence.
    from parser.main import app_state
    monkeypatch.setattr(app_state, "qstore", None)
    r = client.get("/v1/parser/visual/captions?source=photos")
    assert r.status_code == 503
