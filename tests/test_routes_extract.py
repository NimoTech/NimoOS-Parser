import os
import pytest
from parser.routes import extract as extract_mod
from parser.docling_extractor import DoclingExtractor as _DoclingExtractorClass

# Captured at module-import time, before the autouse fixture below patches
# `DoclingExtractor.load` — lets the corrupt-file tests opt back into the
# real docling converter to exercise a genuine ConversionError.
_REAL_DOCLING_LOAD = _DoclingExtractorClass.__dict__["load"]


class _FakeDocling:
    @classmethod
    def load(cls, *, ocr=False):
        return cls()

    def to_markdown(self, source):
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
