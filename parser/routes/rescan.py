import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from parser.repo_jobs import enqueue_job
from parser.repo_records import list_paths_under_root

router = APIRouter(prefix="/v1/parser", tags=["rescan"])


def get_conn():
    from parser.main import app_state
    return app_state.conn


class RescanRequest(BaseModel):
    root_id: str
    op: str


@router.post("/rescan")
async def rescan(req: RescanRequest) -> dict:
    if req.op not in ("reindex", "verify"):
        raise HTTPException(400, "op must be 'reindex' or 'verify'")
    conn = get_conn()
    rows = list_paths_under_root(conn, req.root_id)
    now = int(time.time() * 1000)
    op = "reindex" if req.op == "reindex" else "index"
    for r in rows:
        enqueue_job(conn, root_id=r["root_id"], path=r["path"], op=op,
                    priority=500, now_ms=now)
    return {"queued": len(rows)}
