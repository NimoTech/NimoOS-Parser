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


def test_wiki_consumer_uses_db_not_constant(conn):
    """Disabling .pdf in DB should make consumer skip a .pdf event,
    even though .pdf is in the legacy TEXT_EXT_ALLOWLIST constant."""
    from parser.wiki_consumer import _op_for_event
    ra.set_extension_enabled(conn, ".pdf", False)
    ev = {"op": "create", "path": "/x/y.pdf", "root_id": "r1", "is_dir": False}
    assert _op_for_event(ev, conn) is None

    # Sanity: with .pdf re-enabled, the event becomes "index"
    ra.set_extension_enabled(conn, ".pdf", True)
    assert _op_for_event(ev, conn) == "index"


def test_pipeline_text_run_full_skipped_via_db(conn, tmp_path):
    """When DB disables .pdf, pipeline_text._run_full should early-return
    without writing anything to the qstore."""
    from parser import pipeline_text

    # Create a tiny real file so getsize() works
    f = tmp_path / "y.pdf"
    f.write_bytes(b"%PDF-fake\n")

    ra.set_extension_enabled(conn, ".pdf", False)

    upserts = []

    class StubQStore:
        def tombstone_file(self, **kw): pass
        def upsert_text_chunks(self, points): upserts.extend(points)
        def delete_file(self, **kw): pass
        def set_root_ids_for_file(self, **kw): pass

    class StubEmbedder:
        version = "test-embed"
        def embed_text(self, texts): return []  # would crash if called

    pipe = pipeline_text.TextPipeline(
        conn=conn, qstore=StubQStore(),
        embedder=StubEmbedder(), parser_version="test",
    )
    # Should early-return because .pdf is disabled in DB
    pipe._run_full(root_id="r1", path=str(f), file_id="fid",
                   sha256_full="abc", now_ms=1)
    assert upserts == []  # nothing was embedded or upserted
