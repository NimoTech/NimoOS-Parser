import pytest
from parser.routes import render as render_mod
from parser.routes import extract as extract_mod


@pytest.fixture(autouse=True)
def _fake_render(monkeypatch):
    # Avoid docling/torch: stub the actual rasterization.
    def fake(path, start, end, scale):
        return [{"page": p, "png_b64": f"PNG{p}"} for p in range(start, end + 1)]
    monkeypatch.setattr(render_mod, "_render_pdf_pages", fake)


def test_render_returns_pages(client, monkeypatch, tmp_path):
    monkeypatch.setattr(extract_mod, "EXTRACT_ROOTS", (str(tmp_path),))
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    r = client.post("/v1/parser/render/pages",
                    json={"path": str(f), "page_start": 1, "page_end": 2})
    assert r.status_code == 200
    body = r.json()
    assert [p["page"] for p in body["pages"]] == [1, 2]
    assert body["pages"][0]["png_b64"] == "PNG1"


def test_render_caps_page_count(client, monkeypatch, tmp_path):
    monkeypatch.setattr(extract_mod, "EXTRACT_ROOTS", (str(tmp_path),))
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    r = client.post("/v1/parser/render/pages",
                    json={"path": str(f), "page_start": 1, "page_end": 100})
    assert r.status_code == 200
    # capped to MAX_PAGES_PER_CALL pages starting at page_start
    assert len(r.json()["pages"]) == render_mod.MAX_PAGES_PER_CALL


def test_render_rejects_non_pdf(client, monkeypatch, tmp_path):
    monkeypatch.setattr(extract_mod, "EXTRACT_ROOTS", (str(tmp_path),))
    f = tmp_path / "doc.txt"
    f.write_text("hi")
    r = client.post("/v1/parser/render/pages", json={"path": str(f)})
    assert r.status_code == 400


def test_render_rejects_path_outside_roots(client, monkeypatch, tmp_path):
    monkeypatch.setattr(extract_mod, "EXTRACT_ROOTS", (str(tmp_path / "ok"),))
    (tmp_path / "ok").mkdir()
    outside = tmp_path / "x.pdf"
    outside.write_bytes(b"x")
    r = client.post("/v1/parser/render/pages", json={"path": str(outside)})
    assert r.status_code == 403
