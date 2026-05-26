"""Allowlist persistence + path matching.

The legacy constant TEXT_EXT_ALLOWLIST in wiki_consumer.py remains as a
*seed* (db.init_db copies it on first run) and as a sanity fallback. The
authoritative source is now the DB.
"""
import fnmatch
import posixpath
import sqlite3
import time
import uuid


def list_extensions(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(conn.execute(
        "SELECT ext, enabled, source FROM allowlist_extensions ORDER BY ext"
    ))


def is_extension_enabled(conn: sqlite3.Connection, ext: str) -> bool:
    row = conn.execute(
        "SELECT enabled FROM allowlist_extensions WHERE ext = ?", (ext.lower(),)
    ).fetchone()
    return bool(row and row["enabled"])


def set_extension_enabled(conn: sqlite3.Connection, ext: str, enabled: bool) -> None:
    now_ms = int(time.time() * 1000)
    # upsert: support adding custom extensions too
    conn.execute(
        "INSERT INTO allowlist_extensions(ext, enabled, source, updated_at) "
        "VALUES (?, ?, 'custom', ?) "
        "ON CONFLICT(ext) DO UPDATE SET enabled = excluded.enabled, "
        "updated_at = excluded.updated_at",
        (ext.lower(), 1 if enabled else 0, now_ms),
    )


def list_folder_rules(conn: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT id, root_id, path_glob, action, created_at "
        "FROM allowlist_folders ORDER BY root_id, path_glob"
    )]


def add_folder_rule(conn: sqlite3.Connection, *, root_id: str,
                    path_glob: str, action: str) -> str:
    if action not in ("allow", "deny"):
        raise ValueError(f"action must be 'allow' or 'deny', got {action!r}")
    rule_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO allowlist_folders(id, root_id, path_glob, action, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (rule_id, root_id, path_glob, action, int(time.time() * 1000)),
    )
    return rule_id


def delete_folder_rule(conn: sqlite3.Connection, rule_id: str) -> bool:
    cur = conn.execute("DELETE FROM allowlist_folders WHERE id = ?", (rule_id,))
    return cur.rowcount > 0


def is_path_indexable(conn: sqlite3.Connection, *, root_id: str,
                      path: str) -> bool:
    """Single source of truth for "should this (root_id, path) be indexed".

    Priority: explicit deny > explicit allow > extension check.
    Folder rules only apply within their own root_id.
    """
    ext = posixpath.splitext(path)[1].lower()

    # Folder rules first — deny has the highest priority
    rules = conn.execute(
        "SELECT path_glob, action FROM allowlist_folders WHERE root_id = ?",
        (root_id,),
    ).fetchall()
    matched_allow = False
    for r in rules:
        if fnmatch.fnmatchcase(path, r["path_glob"]):
            if r["action"] == "deny":
                return False
            matched_allow = True

    # Extension check (skip if explicit allow rule covered this path)
    if matched_allow:
        return True
    return is_extension_enabled(conn, ext)
