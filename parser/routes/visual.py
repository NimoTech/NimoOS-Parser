"""Visual ingest routes. The audio namespace is reserved but not registered
this cycle, see spec §2.1."""
import json
import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from parser.repo_jobs import enqueue_job

log = logging.getLogger("parser.routes.visual")

router = APIRouter(prefix="/v1/parser/visual", tags=["visual"])

VISUAL_JOB_PRIORITY = 200  # document jobs default to 100: documents take priority, photo backfill scans come last


class IngestReq(BaseModel):
    source: str = Field(min_length=1, pattern=r"^[a-z0-9_-]+$")
    asset_id: str = Field(min_length=1, max_length=128)
    image_path: str
    mime: str = "image/jpeg"
    meta: dict = Field(default_factory=dict)


def _allowed_dirs() -> list[Path]:
    from parser.main import app_state
    if app_state.settings is not None:
        raw = app_state.settings.visual_allowed_dirs
    else:  # test fixtures that skip settings read env/defaults, same resolution path as production
        from parser.config import load_settings
        raw = load_settings().visual_allowed_dirs
    return [Path(x.strip()) for x in raw.split(",") if x.strip()]


def _validate_path(image_path: str) -> Path:
    p = Path(image_path).resolve()  # resolve() collapses ../ traversal
    if not any(p.is_relative_to(d) for d in _allowed_dirs()):
        raise HTTPException(400, "image_path outside allowed directories")
    if not p.is_file():
        raise HTTPException(400, "image_path not found or not a file")
    return p


@router.post("/ingest", status_code=202)
def ingest(req: IngestReq):
    from parser.main import app_state
    p = _validate_path(req.image_path)
    job_id = enqueue_job(
        app_state.conn, root_id=req.source, path=str(p), op="visual_ingest",
        priority=VISUAL_JOB_PRIORITY,
        sub_modality=json.dumps(
            {"asset_id": req.asset_id, "mime": req.mime, "meta": req.meta},
            ensure_ascii=False),
        now_ms=int(time.time() * 1000),
    )
    return {"job_id": job_id}


@router.get("/captions")
def export_captions(
    source: str = Query(min_length=1, pattern=r"^[a-z0-9_-]+$"),
    limit: int = 512,
    offset: str | None = None,
):
    """Bulk-export captions for a given source (scroll pagination cursor).
    Used by Photos' periodic diff pull (the foundation for Smart Moments
    curation); backfill and incremental share the same path: the first full
    pull is the backfill, after that Photos does its own incremental check
    against mtime_ms."""
    from parser.main import app_state
    if app_state.qstore is None:
        raise HTTPException(503, "qdrant unavailable (qstore not ready)")
    limit = max(1, min(limit, 1024))
    offset = offset or None  # in the contract `offset=` (empty string) means the first page; normalize to None before passing down
    prefix = f"{source}:"
    try:
        points, next_offset = app_state.qstore.scroll_captions(
            source, limit, offset)
    except Exception as exc:
        # Same error semantics as delete_asset: transient outages are left to the
        # caller (Photos) to retry after getting a 503.
        raise HTTPException(503, f"qdrant unavailable, retry later: {exc}")
    items = []
    for p in points:
        file_id = p.get("file_id", "")
        if not file_id.startswith(prefix):
            continue
        items.append({
            "asset_id": file_id[len(prefix):],
            "text": p.get("text", ""),
            "mtime_ms": p.get("mtime_ms"),
        })
    return {
        "items": items,
        "next_offset": None if next_offset is None else str(next_offset),
    }


@router.delete("/assets/{source}/{asset_id}")
def delete_asset(source: str, asset_id: str):
    from parser.main import app_state
    if getattr(app_state, "visual_pipeline", None) is None:
        raise HTTPException(503, "visual pipeline unavailable (qdrant down?)")
    try:
        app_state.visual_pipeline.delete_asset(source=source, asset_id=asset_id)
    except Exception as exc:
        # Qdrant went down transiently after startup: same error semantics as
        # the "pipeline not wired" branch above. Delete doesn't go through the
        # job queue so there's no automatic retry; left to the caller (Photos)
        # to retry after getting a 503.
        raise HTTPException(503, f"qdrant unavailable, retry later: {exc}")
    return {"deleted": True}
