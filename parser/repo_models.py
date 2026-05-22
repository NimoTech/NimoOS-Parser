import sqlite3
from typing import Optional


def register_model(
    conn: sqlite3.Connection, *, name: str, version: str, modality: str,
    dim: Optional[int], registered_at: int,
) -> None:
    conn.execute(
        """
        INSERT INTO model_versions
          (name, version, modality, dim, active, registered_at)
        VALUES (?, ?, ?, ?, 1, ?)
        ON CONFLICT(name, version) DO NOTHING
        """,
        (name, version, modality, dim, registered_at),
    )


def set_active(conn: sqlite3.Connection, name: str, version: str) -> None:
    conn.execute(
        "UPDATE model_versions SET active = 0 WHERE name = ? AND version != ?",
        (name, version),
    )
    conn.execute(
        "UPDATE model_versions SET active = 1 WHERE name = ? AND version = ?",
        (name, version),
    )


def get_active_models(conn: sqlite3.Connection) -> dict:
    rows = conn.execute(
        "SELECT name, version, modality, dim FROM model_versions WHERE active = 1"
    ).fetchall()
    return {r["name"]: {"version": r["version"], "modality": r["modality"], "dim": r["dim"]} for r in rows}


def get_wiki_cursor(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT since_ms FROM wiki_cursor WHERE id = 1").fetchone()
    return row["since_ms"]


def set_wiki_cursor(conn: sqlite3.Connection, since_ms: int, now_ms: int) -> None:
    conn.execute(
        "UPDATE wiki_cursor SET since_ms = ?, updated_at = ? WHERE id = 1",
        (since_ms, now_ms),
    )
