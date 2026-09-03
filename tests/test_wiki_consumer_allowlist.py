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
    # delete for any file must be forwarded to parser (to clean up vectors); the allowlist isn't checked
    ev = {"op": "delete", "path": "/x.MOV", "is_dir": False}
    assert _op_for_event(ev, conn) == "delete"


def test_unknown_ext_skipped(conn):
    ev = {"op": "create", "path": "/x.weirdext", "is_dir": False}
    assert _op_for_event(ev, conn) is None


def test_no_extension_skipped(conn):
    ev = {"op": "create", "path": "/Makefile", "is_dir": False}
    assert _op_for_event(ev, conn) is None


def test_index_event_under_container_dir_is_skipped(conn):
    # .md is an allowed extension, but nothing under .system_data is indexable
    ev = {"op": "create", "root_id": "r1",
          "path": "/DATA/.system_data/home/nimo/.claude/cache/changelog.md"}
    assert _op_for_event(ev, conn) is None


def test_delete_event_under_container_dir_still_passes(conn):
    # deletes must still flow so stale vectors get cleaned up
    ev = {"op": "delete", "root_id": "r1",
          "path": "/DATA/.system_data/home/nimo/.claude.json"}
    assert _op_for_event(ev, conn) == "delete"
