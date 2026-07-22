import sqlite3

import pytest

from parser.pipeline_visual import VisualPipeline, build_index_text, file_id_for


class FakeBackend:
    version = "qwen3-vl-4b-int4/prompt-v1"
    def caption(self, image_bytes):
        return "A cat on a sofa."


class FakeEmbedder:
    version = "bge-m3/v1"
    def embed_text(self, texts):
        return [{"dense": [0.0] * 1024,
                 "sparse": {"indices": [1], "values": [0.5]}} for _ in texts]


class RecordingQStore:
    def __init__(self):
        self.upserts, self.deletes = [], []
    def upsert_text_chunks(self, points):
        self.upserts.append(list(points))
    def delete_file(self, file_id):
        self.deletes.append(file_id)


@pytest.fixture
def pipe(tmp_path):
    conn = sqlite3.connect(":memory:")
    q = RecordingQStore()
    p = VisualPipeline(conn, qstore=q, embedder=FakeEmbedder(),
                       caption_backend=FakeBackend(),
                       parser_version="parser/test")
    img = tmp_path / "small.jpg"
    img.write_bytes(b"\xff\xd8fake")
    return p, q, str(img)


def test_ingest_payload(pipe):
    p, q, img = pipe
    p.ingest_asset(source="photos", asset_id="a1", image_path=img,
                   mime="image/jpeg",
                   meta={"taken_at": "2025-06-01", "place": "Tokyo, Japan"},
                   now_ms=123)
    assert q.deletes == ["photos:a1"], "先清旧块保证幂等"
    (points,) = q.upserts
    (pt,) = points
    pl = pt["payload"]
    assert pl["file_id"] == "photos:a1"
    assert pl["kind"] == "caption"
    assert pl["lang"] == "en"
    assert pl["mime"] == "image/jpeg"
    assert pl["root_ids"] == ["photos"]
    assert pl["source_model_version"] == "qwen3-vl-4b-int4/prompt-v1"
    assert pl["chunk_no"] == 0 and pl["tombstoned_at"] is None
    assert "A cat on a sofa." in pl["text"]
    assert "Taken: 2025-06-01, Tokyo, Japan" in pl["text"]


def test_ingest_deterministic_point_id(pipe):
    p, q, img = pipe
    for _ in range(2):
        p.ingest_asset(source="photos", asset_id="a1", image_path=img,
                       mime="image/jpeg", meta={}, now_ms=1)
    id1 = q.upserts[0][0]["id"]
    id2 = q.upserts[1][0]["id"]
    assert id1 == id2, "同资产重投喂 point id 必须稳定(覆盖式更新)"


def test_delete_asset(pipe):
    p, q, _ = pipe
    p.delete_asset(source="photos", asset_id="a9")
    assert q.deletes == ["photos:a9"]


def test_build_index_text_no_meta():
    assert build_index_text("A cat.", {}) == "A cat."


def test_ingest_missing_image_raises(pipe):
    p, _, _ = pipe
    with pytest.raises(FileNotFoundError):
        p.ingest_asset(source="photos", asset_id="a2",
                       image_path="/nonexistent.jpg", mime="image/jpeg",
                       meta={}, now_ms=1)
