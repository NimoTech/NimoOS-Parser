import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from parser.repo_jobs import (
    list_jobs as _list_jobs, retry_failed_jobs,
    delete_job, clear_failed_jobs, JobNotFound, JobRunning,
)

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


@router.delete("/jobs/{job_id}", status_code=204)
async def cancel_pending_job(job_id: int):
    try:
        delete_job(get_conn(), job_id, now_ms=int(time.time() * 1000))
    except JobNotFound:
        raise HTTPException(404, f"job {job_id} not found")
    except JobRunning:
        raise HTTPException(409, "cannot cancel a running job")
    return None


@router.post("/jobs/clear-failed")
async def post_clear_failed() -> dict:
    n = clear_failed_jobs(get_conn())
    return {"cleared": n}
