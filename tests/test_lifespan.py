import os

from fastapi.testclient import TestClient


def test_discovery_file_written_and_cleaned(tmp_path, monkeypatch):
    f = tmp_path / "parser.url"
    monkeypatch.setenv("PARSER_DISCOVERY_FILE", str(f))
    monkeypatch.setenv("PARSER_BIND_ADDR", "127.0.0.1:9999")
    from parser.main import create_app
    app = create_app(skip_workers=True)
    with TestClient(app) as c:
        assert f.exists()
        assert "http://127.0.0.1:9999" in f.read_text()
    assert not f.exists()


def test_no_discovery_env_no_file(tmp_path, monkeypatch):
    f = tmp_path / "parser.url"
    monkeypatch.delenv("PARSER_DISCOVERY_FILE", raising=False)
    monkeypatch.delenv("PARSER_BIND_ADDR", raising=False)
    from parser.main import create_app
    app = create_app(skip_workers=True)
    with TestClient(app) as c:
        r = c.get("/healthz")
        assert r.status_code == 200
    assert not f.exists()
