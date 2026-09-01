# tests/test_ocr_catalog.py
"""Catalog resolves every entry against rapidocr's bundled registry (no network)."""
from parser.ocr_catalog import entries, get_entry


def test_entries_resolve_urls_and_hashes():
    items = entries()
    ids = [e["id"] for e in items]
    assert ids == ["ppocr-v4-mobile", "ppocr-v4-server", "ppocr-v5-mobile",
                   "ppocr-v5-server", "ppocr-v6-small"]
    for e in items:
        assert set(e["files"]) == {"det", "rec", "cls"}
        for f in e["files"].values():
            assert f["url"].startswith("https://")
            assert len(f["sha256"]) == 64


def test_get_entry():
    assert get_entry("ppocr-v4-mobile")["recommended"] is True
    assert get_entry("nope") is None
