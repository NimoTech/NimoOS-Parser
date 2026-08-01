import sqlite3
import time


_ALLOWED_DEVICES = ("auto", "cuda", "cpu")


def get_state(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        "SELECT paused, concurrency, device, ocr_enabled FROM parser_state WHERE id = 1"
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
