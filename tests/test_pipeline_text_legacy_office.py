"""Tests for the legacy-office branch of TextPipeline._run_full.

These tests monkeypatch convert_legacy and DoclingExtractor.load — we do NOT
need real LibreOffice or real docling here. The integration story (real
soffice end-to-end) is covered separately in test_legacy_office_extractor.
"""
from pathlib import Path

import pytest

from parser.db import init_db
from parser.pipeline_text import TextPipeline


class FakeBGE:
    version = "bge-m3/v1"
    dim = 1024
    def embed_text(self, texts):
        return [{"dense": [0.1] * 1024,
                 "sparse": {"indices": [1], "values": [0.5]}}
                for _ in texts]


class FakeQdrant:
    def __init__(self):
        self.upserts = []
    def upsert_text_chunks(self, points): self.upserts.extend(points)
    def set_root_ids_for_file(self, **kw): pass
    def tombstone_file(self, **kw): pass
    def delete_file(self, **kw): pass
    text_collection = "text_chunks"
    visual_collection = "visual_chunks"


class FakeDoclingExtractor:
    """Replacement for DoclingExtractor.load(...) — returns self, and
    to_markdown returns a long enough markdown string to survive
    chunk_markdown's min_tokens=20 filter."""
    version = "docling/v1"
    @classmethod
    def load(cls, *, ocr=False, model_dir=None, use_gpu=False): return cls()
    def to_markdown(self, source):
        return ("# Converted Document\n\n"
                "This is the markdown produced by docling from the .docx that "
                "LibreOffice generated from the original .doc. " * 10)


@pytest.fixture
def setup(tmp_path):
    conn = init_db(tmp_path / "p.db")
    qstore = FakeQdrant()
    bge = FakeBGE()
    return conn, qstore, bge, tmp_path


def _patch_convert_and_docling(monkeypatch, tmp_path):
    """Make convert_legacy return a stub .docx path, DoclingExtractor.load
    return a FakeDoclingExtractor, and shutil.rmtree a no-op so we can
    inspect what was passed in."""
    fake_outdir = tmp_path / "fake-lo-out"
    fake_outdir.mkdir()
    fake_docx = fake_outdir / "converted.docx"
    fake_docx.write_bytes(b"PK\x03\x04 fake docx")

    seen = {"convert_calls": [], "to_markdown_calls": []}

    def fake_convert_legacy(src):
        seen["convert_calls"].append(src)
        return fake_docx

    monkeypatch.setattr(
        "parser.legacy_office_extractor.convert_legacy",
        fake_convert_legacy,
    )
    monkeypatch.setattr(
        "parser.docling_extractor.DoclingExtractor.load",
        classmethod(lambda cls, *, ocr=False, model_dir=None, use_gpu=False:
                    FakeDoclingExtractor()),
    )
    # Intercept the cleanup so the fixture survives for inspection.
    import parser.pipeline_text as pt
    monkeypatch.setattr(pt.shutil, "rmtree", lambda *a, **kw: None)
    return seen, fake_docx


def test_doc_file_is_converted_then_indexed(setup, monkeypatch):
    conn, qstore, bge, tmp_path = setup
    seen, fake_docx = _patch_convert_and_docling(monkeypatch, tmp_path)

    doc = tmp_path / "report.doc"
    doc.write_bytes(b"\xd0\xcf\x11\xe0 fake OLE")  # not parsed

    pipe = TextPipeline(conn, qstore=qstore, embedder=bge,
                        parser_version="parser/0.1.0")
    pipe.index_file(root_id="root1", path=str(doc), now_ms=100)

    assert seen["convert_calls"] == [str(doc)]
    assert len(qstore.upserts) >= 1
    payload = qstore.upserts[0]["payload"]
    assert "libreoffice-docling" in payload["mime"]
    assert "Converted Document" in payload["text"] or \
           "markdown produced by docling" in payload["text"]
    # No replacement char garbage:
    assert "�" not in payload["text"]


def test_doc_conversion_failure_falls_back_to_skip(setup, monkeypatch):
    """If convert_legacy raises, the pipeline must NOT decode bytes as
    UTF-8 (the old bug). It must record the file with empty chunks and a
    legacy-office mime, exactly like the pre-conversion skip branch."""
    conn, qstore, bge, tmp_path = setup

    def boom(src):
        raise RuntimeError("soffice exploded")

    monkeypatch.setattr(
        "parser.legacy_office_extractor.convert_legacy", boom,
    )

    doc = tmp_path / "broken.doc"
    doc.write_bytes(b"\xd0\xcf\x11\xe0 fake OLE")

    pipe = TextPipeline(conn, qstore=qstore, embedder=bge,
                        parser_version="parser/0.1.0")
    pipe.index_file(root_id="root1", path=str(doc), now_ms=100)

    # No chunks indexed.
    assert qstore.upserts == []
    # But file_record exists with legacy-office mime.
    row = conn.execute(
        "SELECT mime FROM file_records WHERE tombstoned_at IS NULL"
    ).fetchone()
    assert row is not None
    assert row["mime"].startswith("application/legacy-office/")


def test_ppt_and_xls_also_routed_through_convert(setup, monkeypatch):
    conn, qstore, bge, tmp_path = setup
    seen, _ = _patch_convert_and_docling(monkeypatch, tmp_path)

    for name in ("deck.ppt", "sheet.xls", "old.wps"):
        p = tmp_path / name
        p.write_bytes(b"\xd0\xcf\x11\xe0 fake OLE " + name.encode())
        pipe = TextPipeline(conn, qstore=qstore, embedder=bge,
                            parser_version="parser/0.1.0")
        pipe.index_file(root_id="root1", path=str(p), now_ms=100)

    assert [Path(c).suffix for c in seen["convert_calls"]] == [
        ".ppt", ".xls", ".wps",
    ]
