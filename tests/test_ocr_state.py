from parser import ocr_installer, ocr_state
from parser.db import init_db
from parser.ocr_installer import FILE_NAMES, model_dir, set_models_dir
from parser.repo_state import get_state, set_ocr, set_ocr_model


def _install(model_id):
    d = model_dir(model_id)
    d.mkdir(parents=True, exist_ok=True)
    for name in FILE_NAMES.values():
        (d / name).write_bytes(b"onnx")


def test_disabled_short_circuit(tmp_path, monkeypatch):
    conn = init_db(tmp_path / "t.db")
    set_models_dir(tmp_path / "ocr")
    assert ocr_state.resolve_ocr(conn) == (False, None, False)


def test_enabled_with_installed_model(tmp_path, monkeypatch):
    conn = init_db(tmp_path / "t.db")
    set_models_dir(tmp_path / "ocr")
    _install("ppocr-v4-mobile")
    set_ocr_model(conn, "ppocr-v4-mobile")
    set_ocr(conn, True)
    monkeypatch.setattr(ocr_state, "current_device", lambda c: "gpu")
    ocr, mdir, gpu = ocr_state.resolve_ocr(conn)
    assert ocr is True and gpu is True
    assert mdir == str(model_dir("ppocr-v4-mobile"))


def test_enabled_but_files_vanished_degrades(tmp_path, monkeypatch, caplog):
    conn = init_db(tmp_path / "t.db")
    set_models_dir(tmp_path / "ocr")
    set_ocr_model(conn, "ppocr-v4-mobile")   # state says active, disk empty
    set_ocr(conn, True)
    ocr, mdir, gpu = ocr_state.resolve_ocr(conn)
    assert ocr is False and mdir is None
