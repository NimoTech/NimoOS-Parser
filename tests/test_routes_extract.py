import os
from pathlib import Path

import pytest
from parser.routes import extract as extract_mod
from parser.docling_extractor import DoclingExtractor as _DoclingExtractorClass

FIXTURES = Path(__file__).parent / "fixtures" / "extract"

# Captured at module-import time, before the autouse fixture below patches
# `DoclingExtractor.load` — lets the corrupt-file tests opt back into the
# real docling converter to exercise a genuine ConversionError.
_REAL_DOCLING_LOAD = _DoclingExtractorClass.__dict__["load"]


class _FakeDocling:
    @classmethod
    def load(cls, *, ocr=False):
        return cls()

    def to_markdown(self, source, *, page_range=None, page_count_out=None):
        return "# Extracted\n\n" + ("body text " * 50)


@pytest.fixture(autouse=True)
def _mock_docling(monkeypatch):
    monkeypatch.setattr(
        "parser.docling_extractor.DoclingExtractor.load",
        classmethod(lambda cls, *, ocr=False: _FakeDocling()),
    )


def test_extract_pdf_returns_markdown(client, monkeypatch, tmp_path):
    monkeypatch.setattr(extract_mod, "EXTRACT_ROOTS", (str(tmp_path),))
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    r = client.post("/v1/parser/extract", json={"path": str(f)})
    assert r.status_code == 200
    body = r.json()
    assert "Extracted" in body["markdown"]
    assert body["truncated"] is False
    assert body["ocr"] is False


def test_extract_truncates_to_max_chars(client, monkeypatch, tmp_path):
    monkeypatch.setattr(extract_mod, "EXTRACT_ROOTS", (str(tmp_path),))
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    r = client.post("/v1/parser/extract", json={"path": str(f), "max_chars": 10})
    assert r.status_code == 200
    body = r.json()
    assert len(body["markdown"]) == 10
    assert body["truncated"] is True


def test_extract_rejects_path_outside_roots(client, monkeypatch, tmp_path):
    monkeypatch.setattr(extract_mod, "EXTRACT_ROOTS", (str(tmp_path / "allowed"),))
    (tmp_path / "allowed").mkdir()
    outside = tmp_path / "secret.pdf"
    outside.write_bytes(b"x")
    r = client.post("/v1/parser/extract", json={"path": str(outside)})
    assert r.status_code == 403


def test_extract_missing_file_404(client, monkeypatch, tmp_path):
    monkeypatch.setattr(extract_mod, "EXTRACT_ROOTS", (str(tmp_path),))
    r = client.post("/v1/parser/extract", json={"path": str(tmp_path / "nope.pdf")})
    assert r.status_code == 404


def test_extract_plain_text_file(client, monkeypatch, tmp_path):
    monkeypatch.setattr(extract_mod, "EXTRACT_ROOTS", (str(tmp_path),))
    f = tmp_path / "notes.txt"
    f.write_text("hello plain")
    r = client.post("/v1/parser/extract", json={"path": str(f)})
    assert r.status_code == 200
    assert r.json()["markdown"] == "hello plain"


def test_extract_plain_markdown_file_still_200(client, monkeypatch, tmp_path):
    # Untouched path: plain .md never goes near docling, so the 422 handling
    # added for conversion failures must not affect it.
    monkeypatch.setattr(extract_mod, "EXTRACT_ROOTS", (str(tmp_path),))
    f = tmp_path / "notes.md"
    f.write_text("# hello markdown")
    r = client.post("/v1/parser/extract", json={"path": str(f)})
    assert r.status_code == 200
    assert r.json()["markdown"] == "# hello markdown"


def test_extract_corrupt_pptx_returns_422(client, monkeypatch, tmp_path):
    # Opt back into the real docling converter (undoing the autouse mock)
    # so we exercise a genuine ConversionError on unparseable input.
    monkeypatch.setattr(extract_mod, "EXTRACT_ROOTS", (str(tmp_path),))
    monkeypatch.setattr(_DoclingExtractorClass, "load", _REAL_DOCLING_LOAD)
    f = tmp_path / "corrupt.pptx"
    f.write_bytes(b"this is not a real pptx file, just garbage bytes" * 5)
    r = client.post("/v1/parser/extract", json={"path": str(f)})
    assert r.status_code == 422
    assert r.json()["detail"].startswith("extraction failed")


def test_extract_corrupt_docx_returns_422(client, monkeypatch, tmp_path):
    monkeypatch.setattr(extract_mod, "EXTRACT_ROOTS", (str(tmp_path),))
    monkeypatch.setattr(_DoclingExtractorClass, "load", _REAL_DOCLING_LOAD)
    f = tmp_path / "corrupt.docx"
    f.write_bytes(b"this is not a real docx file, just garbage bytes" * 5)
    r = client.post("/v1/parser/extract", json={"path": str(f)})
    assert r.status_code == 422
    assert r.json()["detail"].startswith("extraction failed")


# --- max_pages (P2) -------------------------------------------------------
#
# Fixture `tests/fixtures/extract/multi_page.pdf` is a real 5-page PDF (built
# via python-docx + `soffice --headless --convert-to pdf`, each page holding
# a unique "PAGE N MARKER" string) — needed because docling's `page_range`
# behavior can only be verified against a real multi-page document, not the
# module's `_FakeDocling` mock. `multi_page.docx` is the pre-conversion,
# equally-multi-"page" (page-break-separated) source, used to pin docling's
# observed behavior of silently ignoring `page_range` on non-PDF backends.
#
# Experiment findings (verified locally against the installed docling, see
# .superpowers-r4-p1-report.md P2 section for the full transcript):
#   - PDF: `convert(path, page_range=(1, 1))` on the 5-page fixture returns
#     only page 1's content; `result.input.page_count` still reports 5 (the
#     source total) regardless of page_range — this is what the route uses
#     to compute `truncated` exactly rather than by heuristic.
#   - DOCX: `convert(path, page_range=(1, 1))` returns ALL 5 markers — no
#     error, but no effect either (DOCX has no backend page concept pre­
#     render). `input.page_count` is 0 for docx. Hence the route only
#     forwards page_range when ext == ".pdf".
#   - .md: never reaches docling at all (plain-read branch), confirmed by
#     `test_extract_plain_markdown_file_still_200` above — untouched by any
#     of this.

def test_extract_pdf_max_pages_caps_and_marks_truncated(client, monkeypatch, tmp_path):
    monkeypatch.setattr(extract_mod, "EXTRACT_ROOTS", (str(tmp_path),))
    monkeypatch.setattr(_DoclingExtractorClass, "load", _REAL_DOCLING_LOAD)
    f = tmp_path / "multi.pdf"
    f.write_bytes((FIXTURES / "multi_page.pdf").read_bytes())

    r_full = client.post("/v1/parser/extract", json={"path": str(f)})
    assert r_full.status_code == 200
    full_body = r_full.json()
    assert "PAGE 1 MARKER" in full_body["markdown"]
    assert "PAGE 5 MARKER" in full_body["markdown"]
    assert full_body["truncated"] is False

    r_capped = client.post("/v1/parser/extract",
                            json={"path": str(f), "max_pages": 1})
    assert r_capped.status_code == 200
    capped_body = r_capped.json()
    assert "PAGE 1 MARKER" in capped_body["markdown"]
    assert "PAGE 5 MARKER" not in capped_body["markdown"]
    assert len(capped_body["markdown"]) < len(full_body["markdown"])
    assert capped_body["truncated"] is True


def test_extract_pdf_no_max_pages_behavior_unchanged(client, monkeypatch, tmp_path):
    # Sanity companion to the mocked test_extract_pdf_returns_markdown above,
    # but against the real converter: omitting max_pages must still return
    # the full document, untouched by the new code path.
    monkeypatch.setattr(extract_mod, "EXTRACT_ROOTS", (str(tmp_path),))
    monkeypatch.setattr(_DoclingExtractorClass, "load", _REAL_DOCLING_LOAD)
    f = tmp_path / "multi.pdf"
    f.write_bytes((FIXTURES / "multi_page.pdf").read_bytes())
    r = client.post("/v1/parser/extract", json={"path": str(f)})
    assert r.status_code == 200
    body = r.json()
    for n in range(1, 6):
        assert f"PAGE {n} MARKER" in body["markdown"]
    assert body["truncated"] is False


def test_extract_docx_with_max_pages_is_noop_no_crash(client, monkeypatch, tmp_path):
    # Pinning the gated (PDF-only) behavior: max_pages on a .docx must not
    # error, and — per the observed docling behavior above — has no cropping
    # effect (all pages still come back, truncated stays char-only).
    monkeypatch.setattr(extract_mod, "EXTRACT_ROOTS", (str(tmp_path),))
    monkeypatch.setattr(_DoclingExtractorClass, "load", _REAL_DOCLING_LOAD)
    f = tmp_path / "multi.docx"
    f.write_bytes((FIXTURES / "multi_page.docx").read_bytes())
    r = client.post("/v1/parser/extract",
                     json={"path": str(f), "max_pages": 1})
    assert r.status_code == 200
    body = r.json()
    for n in range(1, 6):
        assert f"PAGE {n} MARKER" in body["markdown"]
    assert body["truncated"] is False


def test_extract_max_pages_zero_rejected_422(client, monkeypatch, tmp_path):
    monkeypatch.setattr(extract_mod, "EXTRACT_ROOTS", (str(tmp_path),))
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    r = client.post("/v1/parser/extract",
                     json={"path": str(f), "max_pages": 0})
    assert r.status_code == 422
