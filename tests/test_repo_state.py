import sqlite3
import tempfile
from pathlib import Path

import pytest

from parser.db import init_db
from parser.repo_state import get_state, set_paused, set_concurrency, set_device, set_ocr


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    return init_db(tmp_path / "parser.db")


def test_default_state_after_init(conn: sqlite3.Connection):
    st = get_state(conn)
    assert st["paused"] is False
    assert st["concurrency"] == 2
    assert st["device"] == "auto"
    assert st["ocr_enabled"] is False


def test_set_ocr_persists(conn: sqlite3.Connection):
    set_ocr(conn, True)
    assert get_state(conn)["ocr_enabled"] is True
    set_ocr(conn, False)
    assert get_state(conn)["ocr_enabled"] is False


@pytest.mark.parametrize("device", ["auto", "cuda", "cpu"])
def test_set_device_persists(conn: sqlite3.Connection, device: str):
    set_device(conn, device)
    assert get_state(conn)["device"] == device


@pytest.mark.parametrize("bad", ["tpu", "GPU", "", "xpu"])
def test_set_device_rejects_invalid(conn: sqlite3.Connection, bad: str):
    with pytest.raises(ValueError):
        set_device(conn, bad)


def test_set_paused_persists(conn: sqlite3.Connection):
    set_paused(conn, True)
    assert get_state(conn)["paused"] is True
    set_paused(conn, False)
    assert get_state(conn)["paused"] is False


def test_set_concurrency_persists(conn: sqlite3.Connection):
    set_concurrency(conn, 4)
    assert get_state(conn)["concurrency"] == 4
    set_concurrency(conn, 1)
    assert get_state(conn)["concurrency"] == 1


@pytest.mark.parametrize("bad", [0, 3, 5, 8, -1])
def test_set_concurrency_rejects_invalid_values(conn: sqlite3.Connection, bad: int):
    with pytest.raises(ValueError):
        set_concurrency(conn, bad)
