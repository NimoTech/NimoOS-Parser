import os
import time
from pathlib import Path

import pytest


def test_payload_includes_mtime_ms(tmp_path):
    """Verify pipeline_text writes mtime_ms into Qdrant payload."""
    from parser import pipeline_text
    from parser.db import init_db
    from parser import repo_allowlist

    conn = init_db(tmp_path / "test.db")
    # Ensure .md is enabled
    assert repo_allowlist.is_extension_enabled(conn, ".md")

    f = tmp_path / "hello.md"
    f.write_text("# hello\nworld " * 30, encoding="utf-8")
    mtime_ms = int(os.path.getmtime(f) * 1000)

    captured_points = []

    class StubQStore:
        def tombstone_file(self, **kw): pass
        def upsert_text_chunks(self, points):
            captured_points.extend(points)
        def delete_file(self, **kw): pass
        def set_root_ids_for_file(self, **kw): pass

    class StubEmbedder:
        version = "test-embedder"
        def embed_text(self, texts):
            return [{"dense": [0.0] * 1024,
                     "sparse": {"indices": [], "values": []}} for _ in texts]

    # Seed file_records first (FK required by file_paths)
    now_ms = int(time.time() * 1000)
    conn.execute(
        "INSERT INTO file_records(file_id, sha256_full, size, mime, "
        "modalities_done, parser_version, indexed_at) "
        "VALUES ('fid-1', 'sha-1', 0, 'text/markdown', '{}', 'test-parser', ?)",
        (now_ms,),
    )
    # Seed file_paths so _collect_root_ids works
    conn.execute(
        "INSERT INTO file_paths(root_id, path, file_id, mtime_ms) "
        "VALUES ('r1', ?, 'fid-1', ?)",
        (str(f), now_ms),
    )

    pipe = pipeline_text.TextPipeline(
        conn=conn, qstore=StubQStore(),
        embedder=StubEmbedder(), parser_version="test-parser",
    )
    pipe._run_full(root_id="r1", path=str(f),
                   file_id="fid-1", sha256_full="sha-1",
                   now_ms=int(time.time() * 1000))

    assert captured_points, "expected chunks upserted"
    for p in captured_points:
        assert "mtime_ms" in p["payload"]
        # Within 1 second tolerance for filesystem mtime granularity
        assert abs(p["payload"]["mtime_ms"] - mtime_ms) < 1000
