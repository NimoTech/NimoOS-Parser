# tests/test_ocr_installer.py
"""Installer state machine with a fake downloader (no network, no threads
needed for determinism: _install_worker is called synchronously)."""
import hashlib

import pytest

from parser import ocr_installer
from parser.ocr_installer import (
    FILE_NAMES, is_installed, model_dir, remove, set_models_dir, snapshot,
)


@pytest.fixture(autouse=True)
def _dirs(tmp_path):
    set_models_dir(tmp_path / "ocr")
    ocr_installer._progress.clear()
    yield


def _entry(payload: bytes):
    sha = hashlib.sha256(payload).hexdigest()
    return {"id": "m1", "files": {t: {"url": f"https://x/{t}", "sha256": sha}
                                  for t in ("det", "rec", "cls")}}


def test_worker_installs_and_verifies(monkeypatch):
    payload = b"model-bytes"
    monkeypatch.setattr(ocr_installer, "_fetch",
                        lambda url, dest, cb: dest.write_bytes(payload))
    ocr_installer._install_worker(_entry(payload))
    assert is_installed("m1")
    for name in FILE_NAMES.values():
        assert (model_dir("m1") / name).read_bytes() == payload
    assert "m1" not in snapshot()  # finished installs carry no progress row


def test_worker_rejects_bad_hash(monkeypatch):
    entry = _entry(b"expected")
    monkeypatch.setattr(ocr_installer, "_fetch",
                        lambda url, dest, cb: dest.write_bytes(b"tampered"))
    ocr_installer._install_worker(entry)
    assert not is_installed("m1")
    assert snapshot()["m1"]["status"] == "error"


def test_remove(monkeypatch):
    payload = b"x"
    monkeypatch.setattr(ocr_installer, "_fetch",
                        lambda url, dest, cb: dest.write_bytes(payload))
    ocr_installer._install_worker(_entry(payload))
    remove("m1")
    assert not is_installed("m1")
