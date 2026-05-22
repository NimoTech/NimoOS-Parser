import json
import os
import uuid
from pathlib import Path

import pytest

from parser.db import init_db
from parser.pipeline_text import TextPipeline


class FakeBGE:
    version = "bge-m3/v1"
    dim = 1024
    def embed_text(self, texts):
        return [{"dense": [0.1] * 1024,
                 "sparse": {"indices": [1], "values": [0.5]}} for _ in texts]


class FakeQdrant:
    def __init__(self):
        self.upserts = []
        self.deletes = []
        self.payload_sets = []
    def upsert_text_chunks(self, points):
        self.upserts.extend(points)
    def set_root_ids_for_file(self, collection, file_id, root_ids):
        self.payload_sets.append((collection, file_id, root_ids))
    def tombstone_file(self, file_id, tombstoned_at):
        self.payload_sets.append(("tombstone", file_id, tombstoned_at))
    def delete_file(self, file_id):
        self.deletes.append(file_id)
    text_collection = "text_chunks"
    visual_collection = "visual_chunks"


@pytest.fixture
def setup(tmp_path):
    conn = init_db(tmp_path / "p.db")
    qstore = FakeQdrant()
    bge = FakeBGE()
    return conn, qstore, bge, tmp_path


def test_pipeline_indexes_markdown_file(setup):
    conn, qstore, bge, tmp_path = setup
    p = tmp_path / "doc.md"
    p.write_text(
        "# Title\n\nIntro paragraph with enough words to pass min_tokens filter. " * 5,
        encoding="utf-8",
    )
    pipe = TextPipeline(conn, qstore=qstore, embedder=bge,
                        parser_version="parser/0.1.0")
    pipe.index_file(root_id="root1", path=str(p), now_ms=100)
    assert len(qstore.upserts) >= 1
    payload = qstore.upserts[0]["payload"]
    assert payload["kind"] == "body"
    assert payload["root_ids"] == ["root1"]
    assert payload["embed_model_version"] == "bge-m3/v1"
    assert "Title" in payload["text"]


def test_pipeline_writes_vector_count(setup):
    conn, qstore, bge, tmp_path = setup
    p = tmp_path / "doc.md"
    p.write_text(
        "# Title\n\nIntro paragraph with enough words to pass min_tokens filter. " * 5,
        encoding="utf-8",
    )
    pipe = TextPipeline(conn, qstore=qstore, embedder=bge,
                        parser_version="parser/0.1.0")
    pipe.index_file(root_id="root1", path=str(p), now_ms=100)
    expected = len(qstore.upserts)
    row = conn.execute(
        "SELECT vector_count FROM file_records WHERE tombstoned_at IS NULL"
    ).fetchone()
    assert row["vector_count"] == expected
    assert expected > 0


def test_pipeline_appends_root_for_known_file(setup):
    conn, qstore, bge, tmp_path = setup
    p = tmp_path / "doc.md"
    p.write_text("# T\n\n" + ("para " * 200), encoding="utf-8")
    pipe = TextPipeline(conn, qstore=qstore, embedder=bge,
                        parser_version="parser/0.1.0")
    pipe.index_file(root_id="root1", path=str(p), now_ms=100)
    n_upserts = len(qstore.upserts)
    pipe.index_file(root_id="root2", path=str(p), now_ms=200)
    # second index call for same content but different root: NO new embed work
    assert len(qstore.upserts) == n_upserts
    # but a payload_set should have happened on text_chunks
    assert any(c == "text_chunks" for c, _, _ in qstore.payload_sets)


def test_pipeline_orphans_old_on_modify(setup):
    conn, qstore, bge, tmp_path = setup
    p = tmp_path / "doc.md"
    p.write_text("# T\n\n" + ("para " * 200), encoding="utf-8")
    pipe = TextPipeline(conn, qstore=qstore, embedder=bge,
                        parser_version="parser/0.1.0")
    pipe.index_file(root_id="root1", path=str(p), now_ms=100)
    p.write_text("# T\n\n" + ("DIFFERENT " * 200), encoding="utf-8")
    pipe.index_file(root_id="root1", path=str(p), now_ms=200)
    # old file_id must have been tombstoned (not deleted)
    tombstone_ops = [op for op in qstore.payload_sets if op[0] == "tombstone"]
    assert len(tombstone_ops) == 1
