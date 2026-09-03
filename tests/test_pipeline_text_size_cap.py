"""Audit P4: no per-file size cap — a multi-GB .log/.csv was read whole into
RAM and embedded, pinning a worker for so long that its lease expired and a
second worker re-picked the same file (P5). Files above the cap are recorded
(so they stop being re-queued) but their content is skipped."""
from parser.db import init_db
from parser.pipeline_text import TextPipeline
from parser.repo_records import get_file_record, list_paths_for_file
from tests.test_pipeline_text import FakeBGE, FakeQdrant


def test_oversized_file_is_recorded_but_not_embedded(tmp_path):
    conn = init_db(tmp_path / "p.db")
    qstore, bge = FakeQdrant(), FakeBGE()
    big = tmp_path / "huge.log"
    big.write_text("line of log text that repeats\n" * 2000, encoding="utf-8")  # ~58 KB
    pipe = TextPipeline(conn, qstore=qstore, embedder=bge, parser_version="parser/0.3.0",
                        max_file_bytes=10_000)
    pipe.index_file(root_id="r", path=str(big), now_ms=100)
    assert qstore.upserts == [], "content above the cap must not be embedded"
    rec = conn.execute("select * from file_records").fetchone()
    assert rec is not None and rec["parser_version"] == "parser/0.3.0"
    assert rec["mime"].startswith("application/x-too-large")


def test_file_under_cap_is_indexed_normally(tmp_path):
    conn = init_db(tmp_path / "p.db")
    qstore, bge = FakeQdrant(), FakeBGE()
    small = tmp_path / "ok.md"
    small.write_text("# T\n\n" + "enough words to pass the minimum token gate for one chunk. " * 3, encoding="utf-8")
    pipe = TextPipeline(conn, qstore=qstore, embedder=bge, parser_version="parser/0.3.0",
                        max_file_bytes=10_000)
    pipe.index_file(root_id="r", path=str(small), now_ms=100)
    assert len(qstore.upserts) >= 1
