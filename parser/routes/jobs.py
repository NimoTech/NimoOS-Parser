import time
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from parser.repo_jobs import list_jobs as _list_jobs, retry_failed_jobs

router = APIRouter(prefix="/v1/parser", tags=["jobs"])


def get_conn():
    from parser.main import app_state
    return app_state.conn


class RetryRequest(BaseModel):
    file_ids: Optional[list[str]] = None


@router.get("/jobs")
async def list_jobs(
    status: str = Query("pending", pattern="^(pending|running|failed)$"),
    limit: int = Query(50, ge=1, le=500),
) -> dict:
    conn = get_conn()
    rows = _list_jobs(conn, status=status, limit=limit)
    return {"jobs": [dict(r) for r in rows]}


@router.post("/jobs/retry")
async def retry(req: RetryRequest) -> dict:
    conn = get_conn()
    n = retry_failed_jobs(conn, file_ids=req.file_ids,
                          now_ms=int(time.time() * 1000))
    return {"retried": n}
