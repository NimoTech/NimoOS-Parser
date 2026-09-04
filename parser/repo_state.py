import json
import sqlite3
import time


_ALLOWED_DEVICES = ("auto", "cuda", "gpu", "cpu")


def get_state(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        "SELECT paused, concurrency, device, ocr_enabled, ocr_model FROM parser_state WHERE id = 1"
    ).fetchone()
    if row is None:
        raise RuntimeError(
            "parser_state row missing — run init_db first to seed the singleton"
        )
    return {
        "paused": bool(row[0]),
        "concurrency": int(row[1]),
        "device": str(row[2]),
        "ocr_enabled": bool(row[3]),
        "ocr_model": str(row[4]),
    }


def set_device(conn: sqlite3.Connection, device: str) -> None:
    if device not in _ALLOWED_DEVICES:
        raise ValueError(
            f"device must be one of {_ALLOWED_DEVICES}; got {device!r}"
        )
    now_ms = int(time.time() * 1000)
    conn.execute(
        "UPDATE parser_state SET device = ?, updated_at = ? WHERE id = 1",
        (device, now_ms),
    )
    conn.commit()


def set_ocr(conn: sqlite3.Connection, enabled: bool) -> None:
    now_ms = int(time.time() * 1000)
    conn.execute(
        "UPDATE parser_state SET ocr_enabled = ?, updated_at = ? WHERE id = 1",
        (1 if enabled else 0, now_ms),
    )
    conn.commit()


def set_ocr_model(conn: sqlite3.Connection, model_id: str) -> None:
    now_ms = int(time.time() * 1000)
    conn.execute(
        "UPDATE parser_state SET ocr_model = ?, updated_at = ? WHERE id = 1",
        (model_id, now_ms),
    )
    conn.commit()


def set_paused(conn: sqlite3.Connection, paused: bool) -> None:
    now_ms = int(time.time() * 1000)
    conn.execute(
        "UPDATE parser_state SET paused = ?, updated_at = ? WHERE id = 1",
        (1 if paused else 0, now_ms),
    )
    conn.commit()


def set_concurrency(conn: sqlite3.Connection, n: int) -> None:
    # 1 / 2 / 4 are the three UI-exposed worker-pool sizing presets (power-saving/balanced/max-performance)
    if n not in (1, 2, 4):
        raise ValueError(f"concurrency must be 1, 2, or 4; got {n}")
    now_ms = int(time.time() * 1000)
    conn.execute(
        "UPDATE parser_state SET concurrency = ?, updated_at = ? WHERE id = 1",
        (n, now_ms),
    )
    conn.commit()


def _get_json(conn: sqlite3.Connection, col: str):
    row = conn.execute(f"SELECT {col} FROM parser_state WHERE id = 1").fetchone()
    if row is None or row[0] is None:
        return None
    return json.loads(row[0])


def _set_json(conn: sqlite3.Connection, col: str, value) -> None:
    conn.execute(
        f"UPDATE parser_state SET {col} = ?, updated_at = ? WHERE id = 1",
        (None if value is None else json.dumps(value), int(time.time() * 1000)),
    )


def get_cursor_gap(conn: sqlite3.Connection):
    """Last detected Wiki archive-horizon gap, or None when the cursor is
    inside the horizon."""
    return _get_json(conn, "cursor_gap")


def set_cursor_gap(conn: sqlite3.Connection, gap) -> None:
    _set_json(conn, "cursor_gap", gap)


def get_verify_last(conn: sqlite3.Connection):
    return _get_json(conn, "verify_last")


def set_verify_last(conn: sqlite3.Connection, result) -> None:
    _set_json(conn, "verify_last", result)
