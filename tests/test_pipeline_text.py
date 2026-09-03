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
    def set_root_ids_for_file(self, file_id, root_ids):
        # New API: patches BOTH text + visual collections symmetric to
        # tombstone_file. Record both updates so tests can verify coverage.
        self.payload_sets.append(("text_chunks", file_id, root_ids))
        self.payload_sets.append(("visual_chunks", file_id, root_ids))
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
    # I3: set_root_ids_for_file must hit BOTH collections (text + visual)
    # so visual_chunks for this file_id stay reachable from the new root.
    cols_touched = {c for c, _, _ in qstore.payload_sets if c in (
        "text_chunks", "visual_chunks")}
    assert cols_touched == {"text_chunks", "visual_chunks"}


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


def test_pipeline_payload_carries_section_and_parent_id(setup):
    """Search merges sibling chunks back into their section (item C); the
    payload must therefore say which section a chunk belongs to and give a
    parent id that is identical for every chunk of that section and stable
    across re-indexes of the same file."""
    conn, qstore, bge, tmp_path = setup
    body = "\n\n".join("paragraph %d %s" % (i, "word " * 60) for i in range(30))
    p = tmp_path / "doc.md"
    p.write_text("lead paragraph before any heading, long enough to pass the min_tokens gate the pipeline applies to every chunk. " * 2 + "\n\n# Big\n\n" + body,
                 encoding="utf-8")
    pipe = TextPipeline(conn, qstore=qstore, embedder=bge, parser_version="parser/0.3.0")
    pipe.index_file(root_id="root1", path=str(p), now_ms=100)
    payloads = [u["payload"] for u in qstore.upserts]
    assert len(payloads) > 2
    for pl in payloads:
        assert "parent_id" in pl and "section" in pl and "section_no" in pl, pl.keys()
    lead = [pl for pl in payloads if pl["section"] == ""]
    big = [pl for pl in payloads if pl["section"] == "Big"]
    assert len(lead) == 1 and len(big) > 1
    assert len({pl["parent_id"] for pl in big}) == 1, "all chunks of one section share a parent"
    assert lead[0]["parent_id"] != big[0]["parent_id"]
    fid = payloads[0]["file_id"]
    assert big[0]["parent_id"] == str(uuid.uuid5(uuid.NAMESPACE_OID, f"{fid}:section:{big[0]['section_no']}"))
