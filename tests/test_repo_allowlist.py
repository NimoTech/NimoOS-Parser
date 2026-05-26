import sqlite3
import tempfile
from pathlib import Path

import pytest

from parser.db import init_db
from parser import repo_allowlist as ra


@pytest.fixture
def conn():
    with tempfile.TemporaryDirectory() as d:
        c = init_db(Path(d) / "test.db")
        yield c
        c.close()


def test_extensions_seed_on_first_init(conn):
    rows = ra.list_extensions(conn)
    exts = {r["ext"] for r in rows}
    # seed from wiki_consumer.TEXT_EXT_ALLOWLIST must be present
    assert ".pdf" in exts
    assert ".md" in exts
    assert ".py" in exts
    for r in rows:
        assert r["enabled"] == 1
        assert r["source"] == "default"


def test_set_extension_enabled_toggle(conn):
    ra.set_extension_enabled(conn, ".pdf", False)
    assert ra.is_extension_enabled(conn, ".pdf") is False
    ra.set_extension_enabled(conn, ".pdf", True)
    assert ra.is_extension_enabled(conn, ".pdf") is True


def test_unknown_extension_is_disabled(conn):
    assert ra.is_extension_enabled(conn, ".xyz") is False


def test_folder_rules_crud(conn):
    rid = ra.add_folder_rule(conn, root_id="r1",
                              path_glob="/Downloads/*", action="deny")
    rules = ra.list_folder_rules(conn)
    assert len(rules) == 1
    assert rules[0]["id"] == rid
    assert rules[0]["action"] == "deny"
    ra.delete_folder_rule(conn, rid)
    assert ra.list_folder_rules(conn) == []


def test_folder_rule_invalid_action_rejected(conn):
    with pytest.raises(ValueError):
        ra.add_folder_rule(conn, root_id="r1",
                            path_glob="/x/*", action="maybe")


def test_is_path_indexable_extension_disabled(conn):
    ra.set_extension_enabled(conn, ".pdf", False)
    assert ra.is_path_indexable(conn, root_id="r1",
                                 path="/x/y.pdf") is False
    assert ra.is_path_indexable(conn, root_id="r1",
                                 path="/x/y.md") is True


def test_is_path_indexable_folder_deny_wins(conn):
    ra.add_folder_rule(conn, root_id="r1",
                       path_glob="/Downloads/*", action="deny")
    assert ra.is_path_indexable(conn, root_id="r1",
                                 path="/Downloads/a.pdf") is False
    assert ra.is_path_indexable(conn, root_id="r1",
                                 path="/Wiki/a.pdf") is True


def test_is_path_indexable_folder_allow_under_default_deny(conn):
    # explicit allow has no effect when default is already allow,
    # but verifies allow rule does not falsely deny
    ra.add_folder_rule(conn, root_id="r1",
                       path_glob="/Wiki/*", action="allow")
    assert ra.is_path_indexable(conn, root_id="r1",
                                 path="/Wiki/a.pdf") is True


def test_folder_rule_only_matches_own_root(conn):
    ra.add_folder_rule(conn, root_id="r1",
                       path_glob="/Downloads/*", action="deny")
    assert ra.is_path_indexable(conn, root_id="r2",
                                 path="/Downloads/a.pdf") is True
