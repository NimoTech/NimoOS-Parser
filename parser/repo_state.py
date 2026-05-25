import sqlite3
import time


def get_state(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        "SELECT paused, concurrency FROM parser_state WHERE id = 1"
    ).fetchone()
    return {"paused": bool(row[0]), "concurrency": int(row[1])}


def set_paused(conn: sqlite3.Connection, paused: bool) -> None:
    now_ms = int(time.time() * 1000)
    conn.execute(
        "UPDATE parser_state SET paused = ?, updated_at = ? WHERE id = 1",
        (1 if paused else 0, now_ms),
    )
    conn.commit()


def set_concurrency(conn: sqlite3.Connection, n: int) -> None:
    if n not in (1, 2, 4):
        raise ValueError(f"concurrency must be 1, 2, or 4; got {n}")
    now_ms = int(time.time() * 1000)
    conn.execute(
        "UPDATE parser_state SET concurrency = ?, updated_at = ? WHERE id = 1",
        (n, now_ms),
    )
    conn.commit()
