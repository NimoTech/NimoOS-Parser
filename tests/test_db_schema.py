import sqlite3
from pathlib import Path

from parser.db import init_db, SCHEMA_SQL


def test_init_db_creates_tables(tmp_path):
    conn = init_db(tmp_path / "p.db")
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert tables >= {
        "file_records", "file_paths", "parse_jobs",
        "model_versions", "wiki_cursor"
    }


def test_init_db_sets_pragmas(tmp_path):
    conn = init_db(tmp_path / "p.db")
    journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert journal.lower() == "wal"
    busy = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    assert busy >= 5000
    fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert fk == 1


def test_init_db_idempotent(tmp_path):
    init_db(tmp_path / "p.db")
    init_db(tmp_path / "p.db")  # second call must not throw


def test_wiki_cursor_singleton(tmp_path):
    conn = init_db(tmp_path / "p.db")
    rows = conn.execute("SELECT id, since_ms FROM wiki_cursor").fetchall()
    assert [tuple(r) for r in rows] == [(1, 0)]


def test_row_factory_enables_dict_access(tmp_path):
    conn = init_db(tmp_path / "p.db")
    row = conn.execute("SELECT id, since_ms FROM wiki_cursor").fetchone()
    assert row["id"] == 1
    assert row["since_ms"] == 0


def test_init_db_adds_last_error_column():
    """file_records should have a last_error TEXT column, readable/writable, defaulting to NULL."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        from parser.db import init_db
        conn = init_db(Path(td) / "p.db")
        cols = {r["name"] for r in conn.execute(
            "PRAGMA table_info(file_records)"
        ).fetchall()}
        assert "last_error" in cols
        # default NULL writable
        conn.execute(
            "INSERT INTO file_records "
            "(file_id, sha256_full, size, mime, modalities_done, "
            " parser_version, indexed_at) "
            "VALUES ('f1','sha',0,'text/plain','{}','parser/0.2.0',0)"
        )
        row = conn.execute(
            "SELECT last_error FROM file_records WHERE file_id='f1'"
        ).fetchone()
        assert row["last_error"] is None


def test_init_db_adds_three_indexes():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        from parser.db import init_db
        conn = init_db(Path(td) / "p.db")
        idx = {r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()}
        assert "idx_file_records_indexed_at" in idx
        assert "idx_file_records_mime" in idx
        assert "idx_file_records_last_error" in idx


def test_init_db_migration_is_idempotent():
    """A second init_db on the same DB should not raise (column already exists / index already exists)."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        from parser.db import init_db
        p = Path(td) / "p.db"
        init_db(p)
        init_db(p)  # second call, the key point - re-adding an ALTER column must not raise
