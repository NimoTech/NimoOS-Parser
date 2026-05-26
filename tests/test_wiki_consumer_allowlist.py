import tempfile
from pathlib import Path

import pytest

from parser.db import init_db
from parser.wiki_consumer import _op_for_event


@pytest.fixture
def conn():
    with tempfile.TemporaryDirectory() as d:
        c = init_db(Path(d) / "test.db")
        yield c
        c.close()


def test_allowed_ext_returns_index(conn):
    ev = {"op": "create", "path": "/x.md", "is_dir": False}
    assert _op_for_event(ev, conn) == "index"


def test_disallowed_ext_returns_none(conn):
    for path in ("/x.MOV", "/x.mp4", "/x.jpg", "/x.zip", "/x.immich", "/x.sql.gz"):
        ev = {"op": "create", "path": path, "is_dir": False}
        assert _op_for_event(ev, conn) is None, f"should skip {path}"


def test_delete_event_passes_through_for_any_ext(conn):
    # delete 任何文件都要 forward 到 parser (用于清向量),allowlist 不查
    ev = {"op": "delete", "path": "/x.MOV", "is_dir": False}
    assert _op_for_event(ev, conn) == "delete"


def test_unknown_ext_skipped(conn):
    ev = {"op": "create", "path": "/x.weirdext", "is_dir": False}
    assert _op_for_event(ev, conn) is None


def test_no_extension_skipped(conn):
    ev = {"op": "create", "path": "/Makefile", "is_dir": False}
    assert _op_for_event(ev, conn) is None
