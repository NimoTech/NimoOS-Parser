import sqlite3
from typing import Optional


def enqueue_job(
    conn: sqlite3.Connection, *, root_id: str, path: str, op: str,
    priority: int = 100, sub_modality: Optional[str] = None, now_ms: int,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO parse_jobs
          (root_id, path, op, sub_modality, priority, attempts,
           last_error, locked_until, created_at, picked_at, done_at)
        VALUES (?, ?, ?, ?, ?, 0, NULL, NULL, ?, NULL, NULL)
        """,
        (root_id, path, op, sub_modality, priority, now_ms),
    )
    return cur.lastrowid


def dequeue_job(
    conn: sqlite3.Connection, *, lease_s: int, now_ms: int,
) -> Optional[sqlite3.Row]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            """
            SELECT * FROM parse_jobs
            WHERE done_at IS NULL
              AND (locked_until IS NULL OR locked_until < ?)
            ORDER BY priority ASC, id ASC
            LIMIT 1
            """,
            (now_ms,),
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        conn.execute(
            """
            UPDATE parse_jobs
            SET locked_until = ?, picked_at = COALESCE(picked_at, ?),
                attempts = attempts + 1
            WHERE id = ?
            """,
            (now_ms + lease_s * 1000, now_ms, row["id"]),
        )
        conn.execute("COMMIT")
        return conn.execute(
            "SELECT * FROM parse_jobs WHERE id = ?", (row["id"],)
        ).fetchone()
    except Exception:
        conn.execute("ROLLBACK")
        raise


def complete_job(conn: sqlite3.Connection, job_id: int, now_ms: int) -> None:
    conn.execute(
        "UPDATE parse_jobs SET done_at = ?, locked_until = NULL WHERE id = ?",
        (now_ms, job_id),
    )


def fail_job(
    conn: sqlite3.Connection, *, job_id: int, error: str, now_ms: int,
    max_attempts: int = 5,
) -> None:
    row = conn.execute(
        "SELECT attempts FROM parse_jobs WHERE id = ?", (job_id,)
    ).fetchone()
    attempts = (row["attempts"] if row else 0) + 1
    if attempts >= max_attempts:
        conn.execute(
            """
            UPDATE parse_jobs
            SET done_at = ?, locked_until = NULL, last_error = ?,
                attempts = attempts + 1
            WHERE id = ?
            """,
            (now_ms, error, job_id),
        )
    else:
        conn.execute(
            """
            UPDATE parse_jobs
            SET last_error = ?, locked_until = NULL, attempts = attempts + 1
            WHERE id = ?
            """,
            (error, job_id),
        )


def list_jobs(
    conn: sqlite3.Connection, *, status: str, limit: int,
) -> list:
    if status == "pending":
        q = "SELECT * FROM parse_jobs WHERE done_at IS NULL AND locked_until IS NULL"
    elif status == "running":
        q = "SELECT * FROM parse_jobs WHERE done_at IS NULL AND locked_until IS NOT NULL"
    elif status == "failed":
        q = "SELECT * FROM parse_jobs WHERE done_at IS NOT NULL AND last_error IS NOT NULL"
    else:
        raise ValueError(f"unknown status: {status}")
    return conn.execute(f"{q} ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


def retry_failed_jobs(
    conn: sqlite3.Connection, *, file_ids: Optional[list], now_ms: int,
) -> int:
    # re-enqueue: clear done_at, reset attempts/error, reset lease
    # file_ids param reserved for §B; for MVP retry all failed
    cur = conn.execute(
        """
        UPDATE parse_jobs
        SET done_at = NULL, attempts = 0, last_error = NULL,
            locked_until = NULL, picked_at = NULL
        WHERE done_at IS NOT NULL AND last_error IS NOT NULL
        """
    )
    return cur.rowcount
