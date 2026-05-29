"""Business logic for the file list + filter endpoints.

The HTTP layer (parser.routes.files) translates query strings into kwargs and
hands them here. Splitting this out keeps SQL out of the route module and
makes the logic unit-testable without a TestClient.
"""
import json
import sqlite3
from typing import Optional

MAX_LIMIT = 500
DEFAULT_LIMIT = 100

_SORT_COLUMNS = {
    "indexed_at": "r.indexed_at",
    "size": "r.size",
    "vector_count": "r.vector_count",
    "path": "r.file_id",  # path is multi-valued; sort by file_id as a stable proxy
}


def _escape_like(s: str) -> str:
    """Escape % and _ so user input is treated as a literal prefix."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _build_where(
    *, root_id, path_prefix, mime_prefix, has_error, tombstoned,
) -> tuple[str, list]:
    """Build the WHERE clause for the file_records-anchored query.
    Returns (where_sql, params) — joinable into the outer SELECT.
    """
    clauses, params = [], []

    # tombstoned filter
    if tombstoned == "alive":
        clauses.append("r.tombstoned_at IS NULL")
    elif tombstoned == "tombstoned":
        clauses.append("r.tombstoned_at IS NOT NULL")
    elif tombstoned == "all":
        pass
    else:
        raise ValueError(f"invalid tombstoned: {tombstoned!r}")

    if has_error:
        clauses.append("r.last_error IS NOT NULL")

    if mime_prefix:
        clauses.append("r.mime LIKE ? ESCAPE '\\'")
        params.append(_escape_like(mime_prefix) + "%")

    if root_id or path_prefix:
        # EXISTS subquery to avoid join-row explosion. Uses file_paths PK.
        exists_clauses, exists_params = [], []
        if root_id:
            exists_clauses.append("p.root_id = ?")
            exists_params.append(root_id)
        if path_prefix:
            exists_clauses.append("p.path LIKE ? ESCAPE '\\'")
            exists_params.append(_escape_like(path_prefix) + "%")
        clauses.append(
            "EXISTS (SELECT 1 FROM file_paths p "
            "        WHERE p.file_id = r.file_id "
            "          AND " + " AND ".join(exists_clauses) + ")"
        )
        params.extend(exists_params)

    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    return where, params


def _fetch_paths(conn, file_ids):
    if not file_ids:
        return {}
    ph = ",".join("?" * len(file_ids))
    rows = conn.execute(
        f"SELECT root_id, path, file_id, mtime_ms FROM file_paths "
        f"WHERE file_id IN ({ph})",
        file_ids,
    ).fetchall()
    by_id = {}
    for r in rows:
        by_id.setdefault(r["file_id"], []).append({
            "root_id": r["root_id"], "path": r["path"],
            "mtime_ms": r["mtime_ms"],
        })
    return by_id


def _fetch_indexing_file_ids(conn, file_ids):
    """Return set of file_ids that currently have an open parse_job
    (done_at IS NULL) against any of their paths."""
    if not file_ids:
        return set()
    ph = ",".join("?" * len(file_ids))
    rows = conn.execute(
        f"""
        SELECT DISTINCT fp.file_id
        FROM file_paths fp
        JOIN parse_jobs pj
          ON pj.root_id = fp.root_id AND pj.path = fp.path
        WHERE fp.file_id IN ({ph})
          AND pj.done_at IS NULL
        """,
        file_ids,
    ).fetchall()
    return {r["file_id"] for r in rows}


def compute_status(rec_row, *, indexing_ids: set) -> str:
    """tombstoned > indexing > error > ok."""
    if rec_row["tombstoned_at"] is not None:
        return "tombstoned"
    if rec_row["file_id"] in indexing_ids:
        return "indexing"
    if rec_row["last_error"] is not None:
        return "error"
    return "ok"


def list_files(
    conn: sqlite3.Connection, *,
    root_id: Optional[str] = None,
    path_prefix: Optional[str] = None,
    mime_prefix: Optional[str] = None,
    has_error: bool = False,
    tombstoned: str = "alive",
    sort: str = "indexed_at",
    order: str = "desc",
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
) -> dict:
    """Return paginated file list matching the filters.

    See nimo_os_docs/.../2026-05-28-knowledge-files-reindex-design.md §4.1.
    """
    if sort not in _SORT_COLUMNS:
        raise ValueError(f"invalid sort: {sort!r}")
    if order not in ("asc", "desc"):
        raise ValueError(f"invalid order: {order!r}")
    if limit < 1 or limit > MAX_LIMIT:
        raise ValueError(f"limit must be 1..{MAX_LIMIT}")
    if offset < 0:
        raise ValueError("offset must be >= 0")

    where, params = _build_where(
        root_id=root_id, path_prefix=path_prefix, mime_prefix=mime_prefix,
        has_error=has_error, tombstoned=tombstoned,
    )

    sort_col = _SORT_COLUMNS[sort]
    order_sql = "DESC" if order == "desc" else "ASC"

    total_row = conn.execute(
        f"SELECT COUNT(*) AS n FROM file_records r {where}", params,
    ).fetchone()
    total = total_row["n"]

    rows = conn.execute(
        f"""
        SELECT r.file_id, r.sha256_full, r.size, r.mime,
               r.modalities_done, r.parser_version, r.indexed_at,
               r.tombstoned_at, r.vector_count, r.last_error
        FROM file_records r
        {where}
        ORDER BY {sort_col} {order_sql}, r.file_id {order_sql}
        LIMIT ? OFFSET ?
        """,
        params + [limit, offset],
    ).fetchall()

    file_ids = [r["file_id"] for r in rows]
    paths_by_id = _fetch_paths(conn, file_ids)
    indexing_ids = _fetch_indexing_file_ids(conn, file_ids)

    files = []
    for r in rows:
        try:
            modalities = json.loads(r["modalities_done"])
        except Exception:
            modalities = {}
        files.append({
            "file_id": r["file_id"],
            "paths": paths_by_id.get(r["file_id"], []),
            "sha256_full": r["sha256_full"],
            "size": r["size"],
            "mime": r["mime"],
            "modalities_done": modalities,
            "parser_version": r["parser_version"],
            "indexed_at": r["indexed_at"],
            "tombstoned_at": r["tombstoned_at"],
            "vector_count": r["vector_count"],
            "last_error": r["last_error"],
            "status": compute_status(r, indexing_ids=indexing_ids),
        })

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "files": files,
    }
