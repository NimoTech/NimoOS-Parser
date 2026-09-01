from parser.db import init_db
from parser.repo_state import get_state, set_ocr_model


def test_ocr_model_default_empty(tmp_path):
    conn = init_db(tmp_path / "t.db")
    assert get_state(conn)["ocr_model"] == ""


def test_set_and_clear_ocr_model(tmp_path):
    conn = init_db(tmp_path / "t.db")
    set_ocr_model(conn, "ppocr-v4-mobile")
    assert get_state(conn)["ocr_model"] == "ppocr-v4-mobile"
    set_ocr_model(conn, "")
    assert get_state(conn)["ocr_model"] == ""
