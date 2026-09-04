"""Retire every record under a Wiki root in one pass.

Triggered by Wiki's `root_removed` event (one row per deleted root, replacing
the per-file delete fan-out) and by verify when Parser holds a root_id Wiki no
longer lists. Semantics per file mirror pipeline_text.delete_path: a file that
still has paths in other roots keeps its vectors with a narrowed root_ids; a
file left with no path is tombstoned (vectors hidden now, deleted by gc after
the grace period).
"""
import logging
import sqlite3

from parser.repo_records import list_paths_for_file, set_tombstone

log = logging.getLogger("parser.service_retire")


def retire_root(conn: sqlite3.Connection, qstore, *, root_id: str, now_ms: int) -> dict:
    file_ids = [r[0] for r in conn.execute(
        "SELECT DISTINCT file_id FROM file_paths WHERE root_id = ? ORDER BY file_id",
        (root_id,))]
    out = {"root_id": root_id, "files_seen": len(file_ids), "tombstoned": 0,
           "rehomed": 0, "jobs_dropped": 0}
    if file_ids and qstore is None:
        raise RuntimeError("qdrant unavailable: cannot retire root %s" % root_id)

    # Update Qdrant BEFORE touching SQLite for each file, so a mid-loop
    # qstore failure leaves that file (and every file not yet processed)
    # untouched in file_paths/file_records — a retry of retire_root simply
    # redoes the files that never got past the qstore call. (The connection
    # is opened with isolation_level=None/autocommit, so each conn.execute
    # commits immediately; there is no transaction to roll back on failure.)
    for fid in file_ids:
        remaining = sorted(
            {r["root_id"] for r in list_paths_for_file(conn, fid)} - {root_id}
        )
        if remaining:
            qstore.set_root_ids_for_file(file_id=fid, root_ids=remaining)
            conn.execute(
                "DELETE FROM file_paths WHERE root_id = ? AND file_id = ?",
                (root_id, fid))
            out["rehomed"] += 1
        else:
            qstore.tombstone_file(file_id=fid, tombstoned_at=now_ms)
            # Tombstone BEFORE dropping the path: the reverse order means a
            # failure of the second statement leaves the file with no path
            # (nothing to find it by on a retry) and no tombstoned_at
            # (nothing for gc to collect) — a permanently orphaned record.
            set_tombstone(conn, file_id=fid, at_ms=now_ms)
            conn.execute(
                "DELETE FROM file_paths WHERE root_id = ? AND file_id = ?",
                (root_id, fid))
            out["tombstoned"] += 1

    # Dropping a row for a job that is already leased by a worker does not
    # stop that worker: it can still finish and re-create file_records /
    # file_paths for this root afterwards. Those strays are transient — the
    # next verify sees a root Wiki no longer lists and retires them again.
    cur = conn.execute(
        "DELETE FROM parse_jobs WHERE root_id = ? AND done_at IS NULL AND op != 'retire_root'",
        (root_id,))
    out["jobs_dropped"] = cur.rowcount if cur.rowcount is not None else 0
    log.info("retired root %s: %s", root_id, out)
    return out
