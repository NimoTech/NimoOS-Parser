import sqlite3
import tempfile
from pathlib import Path

import pytest

from parser.db import init_db
from parser.repo_state import get_state, set_paused, set_concurrency


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    return init_db(tmp_path / "parser.db")


def test_default_state_after_init(conn: sqlite3.Connection):
    st = get_state(conn)
    assert st["paused"] is False
    assert st["concurrency"] == 2


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
