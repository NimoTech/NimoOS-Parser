"""Force-reindex business logic.

A single transactional unit:
  1. Resolve target file_id set (explicit list OR filter)
  2. For each:
     - skip if not_found / already_tombstoned / no_paths_orphan
     - else: set_tombstone (SQLite) + enqueue one reindex job per (root,path)
  3. After commit: best-effort Qdrant tombstone (failures logged, not raised)

Spec: internal spec — knowledge files reindex (2026-05-28) §4.2
"""
import logging
import sqlite3
from typing import Optional

from parser.repo_jobs import enqueue_job
from parser.repo_records import (
    get_file_record, list_paths_for_file, set_tombstone,
)
from parser.service_files import (
    MAX_REINDEX_BY_FILTER, count_file_ids_by_filter, select_file_ids_by_filter,
)

log = logging.getLogger("parser.service_reindex")

MAX_REINDEX_FILE_IDS = 500
REINDEX_PRIORITY = 1000
# Version-drift re-index runs behind everything else: live file events (100)
# and explicit user reindex requests (1000) must never wait for it.
DRIFT_REINDEX_PRIORITY = 2000


def enqueue_version_drift(
    conn: sqlite3.Connection, *, parser_version: str, now_ms: int,
) -> int:
    """Queue a low-priority `reindex` job for every live file whose stored
    parser_version differs from the running one.

    This is what makes "bump PARSER_VERSION" sufficient: the version check
    in IdentityResolver only fires when a file is next touched, so without
    this sweep untouched files kept serving old-schema chunks indefinitely.
    Idempotent — a path that already has an open job is skipped, so a
    restart mid-sweep does not double-queue. Returns the number of jobs
    enqueued. No tombstoning: index_file resolves the drift itself and
    replaces the file's vectors atomically.
    """
    rows = conn.execute(
        """
        SELECT fp.root_id, fp.path
        FROM file_records fr
        JOIN file_paths fp ON fp.file_id = fr.file_id
        WHERE fr.parser_version != ? AND fr.tombstoned_at IS NULL
        ORDER BY fp.root_id, fp.path
        """,
        (parser_version,),
    ).fetchall()
    queued = 0
    for r in rows:
        open_job = conn.execute(
            "SELECT 1 FROM parse_jobs WHERE root_id = ? AND path = ? "
            "AND done_at IS NULL LIMIT 1",
            (r["root_id"], r["path"]),
        ).fetchone()
        if open_job is not None:
            continue
        enqueue_job(
            conn, root_id=r["root_id"], path=r["path"], op="reindex",
            priority=DRIFT_REINDEX_PRIORITY, now_ms=now_ms,
        )
        queued += 1
    if queued:
        log.info("parser_version drift: queued %d re-index jobs (now %s)",
                 queued, parser_version)
    return queued


def reindex_files(
    conn: sqlite3.Connection, *, qstore, file_ids: Optional[list],
    filter: Optional[dict], reason: Optional[str], now_ms: int,
) -> dict:
    """Force-reindex either an explicit file_ids list (mode A) or all files
    matching a filter (mode B). Exactly one must be given.

    Returns {queued, tombstoned, job_ids, skipped}. Raises ValueError on
    user input errors (caller maps these to 400).
    """
    if (file_ids is None) == (filter is None):
        raise ValueError(
            "must specify exactly one of file_ids or filter (not both, not neither)"
        )

    if file_ids is not None:
        if len(file_ids) < 1 or len(file_ids) > MAX_REINDEX_FILE_IDS:
            raise ValueError(
                f"too many file_ids (max {MAX_REINDEX_FILE_IDS})"
            )
        candidate_ids = list(file_ids)
    else:
        # filter mode — count first so we can 400 before enqueuing anything
        n = count_file_ids_by_filter(conn, **filter)
        if n > MAX_REINDEX_BY_FILTER:
            raise ValueError(
                f"filter matches {n} files (> {MAX_REINDEX_BY_FILTER}); "
                "narrow it or raise max_reindex_by_filter"
            )
        candidate_ids = select_file_ids_by_filter(
            conn, **filter, limit=MAX_REINDEX_BY_FILTER,
        )

    log.info(
        "reindex requested: reason=%s candidates=%d",
        reason, len(candidate_ids),
    )

    skipped = []
    tombstoned_ids = []
    job_ids = []

    # Single SQLite transaction for tombstone + enqueue. If anything raises,
    # SQLite rolls back; Qdrant call below never runs.
    with conn:
        for fid in candidate_ids:
            rec = get_file_record(conn, fid)
            if rec is None:
                skipped.append({"file_id": fid, "reason": "not_found"})
                continue
            if rec["tombstoned_at"] is not None:
                skipped.append(
                    {"file_id": fid, "reason": "already_tombstoned"}
                )
                continue
            paths = list_paths_for_file(conn, fid)
            if not paths:
                log.warning(
                    "orphan file_record with no paths: %s — skipping", fid,
                )
                skipped.append(
                    {"file_id": fid, "reason": "no_paths_orphan"}
                )
                continue
            set_tombstone(conn, file_id=fid, at_ms=now_ms)
            for p in paths:
                jid = enqueue_job(
                    conn, root_id=p["root_id"], path=p["path"],
                    op="reindex", priority=REINDEX_PRIORITY, now_ms=now_ms,
                )
                job_ids.append(jid)
            tombstoned_ids.append(fid)

    # Best-effort Qdrant tombstone (outside transaction; failures don't undo)
    for fid in tombstoned_ids:
        try:
            qstore.tombstone_file(file_id=fid, tombstoned_at=now_ms)
        except Exception as e:
            log.warning("qdrant tombstone failed for %s: %s", fid, e)

    return {
        "queued": len(job_ids),
        "tombstoned": len(tombstoned_ids),
        "job_ids": job_ids,
        "skipped": skipped,
    }
