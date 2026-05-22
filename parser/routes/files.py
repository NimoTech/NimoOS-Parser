from typing import Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/v1/parser", tags=["files"])

MAX_BATCH = 200


def get_conn():
    from parser.main import app_state
    return app_state.conn


@router.get("/_internal/files")
async def get_files(
    file_ids: str = Query(..., description="comma-separated file_ids"),
) -> dict:
    ids = [s.strip() for s in file_ids.split(",") if s.strip()]
    if not ids:
        return {"files": [], "missing": []}
    if len(ids) > MAX_BATCH:
        raise HTTPException(400, f"too many file_ids (max {MAX_BATCH})")
    conn = get_conn()
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT file_id, sha256_full, size, mime, modalities_done, "
        f"  parser_version, indexed_at, tombstoned_at "
        f"FROM file_records WHERE file_id IN ({placeholders})",
        ids,
    ).fetchall()
    records = {r["file_id"]: dict(r) for r in rows}
    # Now grab all paths for these file_ids
    path_rows = conn.execute(
        f"SELECT root_id, path, file_id, mtime_ms FROM file_paths "
        f"WHERE file_id IN ({placeholders})",
        ids,
    ).fetchall()
    paths_by_id: dict[str, list] = {}
    for pr in path_rows:
        paths_by_id.setdefault(pr["file_id"], []).append({
            "root_id": pr["root_id"], "path": pr["path"], "mtime_ms": pr["mtime_ms"],
        })
    files = []
    found_ids = set()
    for fid in ids:
        if fid not in records:
            continue
        found_ids.add(fid)
        rec = records[fid]
        import json as _json
        try:
            modalities = _json.loads(rec["modalities_done"])
        except Exception:
            modalities = {}
        files.append({
            "file_id": fid,
            "paths": paths_by_id.get(fid, []),
            "mime": rec["mime"],
            "modalities_done": modalities,
            "parser_version": rec["parser_version"],
            "indexed_at": rec["indexed_at"],
            "tombstoned_at": rec["tombstoned_at"],
        })
    missing = [fid for fid in ids if fid not in found_ids]
    return {"files": files, "missing": missing}
