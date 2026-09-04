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
        "SELECT DISTINCT file_id FROM file_paths WHERE root_id = ?", (root_id,))]
    out = {"root_id": root_id, "files_seen": len(file_ids), "tombstoned": 0,
           "rehomed": 0, "jobs_dropped": 0}
    if file_ids and qstore is None:
        raise RuntimeError("qdrant unavailable: cannot retire root %s" % root_id)

    conn.execute("DELETE FROM file_paths WHERE root_id = ?", (root_id,))
    for fid in file_ids:
        remaining = sorted({r["root_id"] for r in list_paths_for_file(conn, fid)})
        if remaining:
            qstore.set_root_ids_for_file(file_id=fid, root_ids=remaining)
            out["rehomed"] += 1
        else:
            set_tombstone(conn, file_id=fid, at_ms=now_ms)
            qstore.tombstone_file(file_id=fid, tombstoned_at=now_ms)
            out["tombstoned"] += 1

    cur = conn.execute(
        "DELETE FROM parse_jobs WHERE root_id = ? AND done_at IS NULL AND op != 'retire_root'",
        (root_id,))
    out["jobs_dropped"] = cur.rowcount if cur.rowcount is not None else 0
    log.info("retired root %s: %s", root_id, out)
    return out
