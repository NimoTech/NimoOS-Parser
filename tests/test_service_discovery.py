from pathlib import Path

from parser.service_discovery import write_url, remove_url


def test_write_and_remove_url(tmp_path):
    f = tmp_path / "parser.url"
    write_url(f, "127.0.0.1:8283")
    assert f.read_text() == "http://127.0.0.1:8283\n"
    remove_url(f)
    assert not f.exists()


def test_remove_url_idempotent(tmp_path):
    f = tmp_path / "nope.url"
    remove_url(f)  # must not throw


def test_write_url_preserves_http_prefix(tmp_path):
    f = tmp_path / "parser.url"
    write_url(f, "http://192.168.1.5:9000")
    assert f.read_text() == "http://192.168.1.5:9000\n"


def test_write_url_creates_parent_dir(tmp_path):
    f = tmp_path / "subdir" / "parser.url"
    write_url(f, "127.0.0.1:8283")
    assert f.exists()
