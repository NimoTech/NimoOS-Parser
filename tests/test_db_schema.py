import sqlite3

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
    assert rows == [(1, 0)]
