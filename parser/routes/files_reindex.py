"""POST /v1/parser/files/reindex — force-reindex selected files.

Body shape: exactly one of `file_ids` (list, ≤500) or `filter` (object, same
keys as GET /v1/parser/files filters); plus optional `reason` (echoed to
service log only). See spec §4.2.
"""
import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/v1/parser", tags=["files"])


def get_conn():
    from parser.main import app_state
    return app_state.conn


def get_qstore():
    from parser.main import app_state
    return app_state.qstore


class ReindexFilter(BaseModel):
    root_id: Optional[str] = None
    path_prefix: Optional[str] = None
    mime_prefix: Optional[str] = None
    has_error: bool = False
    tombstoned: str = "alive"


class ReindexRequest(BaseModel):
    file_ids: Optional[list[str]] = None
    filter: Optional[ReindexFilter] = None
    reason: Optional[str] = None


@router.post("/files/reindex")
async def reindex(req: ReindexRequest) -> dict:
    from parser.service_reindex import reindex_files
    now_ms = int(time.time() * 1000)
    try:
        return reindex_files(
            get_conn(),
            qstore=get_qstore(),
            file_ids=req.file_ids,
            filter=req.filter.model_dump() if req.filter else None,
            reason=req.reason,
            now_ms=now_ms,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
