import pytest

from parser.db import init_db
from parser.repo_state import (get_cursor_gap, get_verify_last, set_cursor_gap,
                               set_verify_last)


@pytest.fixture
def conn(tmp_path):
    return init_db(tmp_path / "p.db")


def test_cursor_gap_roundtrip_and_clear(conn):
    assert get_cursor_gap(conn) is None
    gap = {"detected_at": 5, "since_ms": 1, "last_seq": 2, "archive_cutoff_ms": 9}
    set_cursor_gap(conn, gap)
    assert get_cursor_gap(conn) == gap
    set_cursor_gap(conn, None)
    assert get_cursor_gap(conn) is None


def test_verify_last_roundtrip(conn):
    assert get_verify_last(conn) is None
    res = {"trigger": "manual", "started_at": 1, "finished_at": None, "ok": True,
           "roots": [], "retired_roots": [], "error": None}
    set_verify_last(conn, res)
    assert get_verify_last(conn) == res
