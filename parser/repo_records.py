import json
import sqlite3
from typing import Optional


def upsert_file_record(
    conn: sqlite3.Connection, *, file_id: str, sha256_full: str, size: int,
    mime: str, modalities_done: dict, parser_version: str, indexed_at: int,
) -> None:
    conn.execute(
        """
        INSERT INTO file_records
          (file_id, sha256_full, size, mime, modalities_done, parser_version,
           indexed_at, tombstoned_at, vector_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 0)
        ON CONFLICT(file_id) DO UPDATE SET
          size = excluded.size,
          mime = excluded.mime,
          modalities_done = excluded.modalities_done,
          parser_version = excluded.parser_version,
          indexed_at = excluded.indexed_at,
          tombstoned_at = NULL
        """,
        (file_id, sha256_full, size, mime, json.dumps(modalities_done),
         parser_version, indexed_at),
    )


def get_file_record(conn: sqlite3.Connection, file_id: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM file_records WHERE file_id = ?", (file_id,)
    ).fetchone()


def upsert_file_path(
    conn: sqlite3.Connection, root_id: str, path: str, file_id: str,
    mtime_ms: int,
) -> None:
    conn.execute(
        """
        INSERT INTO file_paths(root_id, path, file_id, mtime_ms)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(root_id, path) DO UPDATE SET
          file_id = excluded.file_id,
          mtime_ms = excluded.mtime_ms
        """,
        (root_id, path, file_id, mtime_ms),
    )


def remove_file_path(conn: sqlite3.Connection, root_id: str, path: str) -> None:
    conn.execute("DELETE FROM file_paths WHERE root_id = ? AND path = ?",
                 (root_id, path))


def list_paths_for_file(conn: sqlite3.Connection, file_id: str) -> list:
    return conn.execute(
        "SELECT root_id, path, mtime_ms FROM file_paths WHERE file_id = ?",
        (file_id,),
    ).fetchall()


def count_paths_for_file_in_root(
    conn: sqlite3.Connection, file_id: str, root_id: str,
) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM file_paths WHERE file_id = ? AND root_id = ?",
        (file_id, root_id),
    ).fetchone()
    return row[0]


def list_paths_under_root(conn: sqlite3.Connection, root_id: str) -> list:
    return conn.execute(
        "SELECT root_id, path, file_id FROM file_paths WHERE root_id = ?",
        (root_id,),
    ).fetchall()


def set_tombstone(conn: sqlite3.Connection, file_id: str, at_ms: int) -> None:
    conn.execute(
        "UPDATE file_records SET tombstoned_at = ? WHERE file_id = ?",
        (at_ms, file_id),
    )


def clear_tombstone(conn: sqlite3.Connection, file_id: str) -> None:
    conn.execute(
        "UPDATE file_records SET tombstoned_at = NULL WHERE file_id = ?",
        (file_id,),
    )


def list_tombstoned_older_than(conn: sqlite3.Connection, before_ms: int) -> list:
    return conn.execute(
        """
        SELECT file_id FROM file_records
        WHERE tombstoned_at IS NOT NULL AND tombstoned_at < ?
        """,
        (before_ms,),
    ).fetchall()


def delete_file_record(conn: sqlite3.Connection, file_id: str) -> None:
    conn.execute("DELETE FROM file_paths WHERE file_id = ?", (file_id,))
    conn.execute("DELETE FROM file_records WHERE file_id = ?", (file_id,))


def set_last_error(
    conn: sqlite3.Connection, *, root_id: str, path: str,
    error: Optional[str],
) -> None:
    """Update file_records.last_error for the file at (root_id, path).

    No-op if (root_id, path) has no file_paths row — this happens when a job
    fails before its file_record is created (e.g. sha256 stage). Those
    job-level failures stay visible only through parse_jobs.last_error and
    /v1/parser/jobs; they don't surface in the file list.
    """
    row = conn.execute(
        "SELECT file_id FROM file_paths WHERE root_id = ? AND path = ?",
        (root_id, path),
    ).fetchone()
    if row is None:
        return
    conn.execute(
        "UPDATE file_records SET last_error = ? WHERE file_id = ?",
        (error, row["file_id"]),
    )
