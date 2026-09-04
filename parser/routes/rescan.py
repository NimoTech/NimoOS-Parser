import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from parser.repo_allowlist import is_path_indexable
from parser.repo_jobs import enqueue_job
from parser.repo_records import list_paths_under_root

router = APIRouter(prefix="/v1/parser", tags=["rescan"])


def get_conn():
    from parser.main import app_state
    return app_state.conn


def get_verify_runner():
    from parser.main import app_state
    return getattr(app_state, "verify_runner", None)


class RescanRequest(BaseModel):
    op: str
    root_id: Optional[str] = None


@router.post("/rescan", status_code=200)
async def rescan(req: RescanRequest, response: Response) -> dict:
    """Force-rescan.

    `op=reindex` (root_id required) — enqueue a reindex job per known path.
    `op=verify`  (root_id optional) — reconcile Parser's ledger against Wiki's
      file_index in the background (service_verify); 202 on start, 409 while
      a verify is already running. Result lands in GET /stats `verify_last`.
    """
    if req.op == "reindex":
        if not req.root_id:
            raise HTTPException(400, "root_id is required for op='reindex'")
        conn = get_conn()
        rows = list_paths_under_root(conn, req.root_id)
        now = int(time.time() * 1000)
        queued = 0
        for r in rows:
            if not is_path_indexable(conn, root_id=r["root_id"], path=r["path"]):
                continue
            enqueue_job(conn, root_id=r["root_id"], path=r["path"],
                        op="reindex", priority=500, now_ms=now)
            queued += 1
        return {"queued": queued}
    if req.op == "verify":
        runner = get_verify_runner()
        if runner is None:
            raise HTTPException(503, "verify unavailable (wiki or qdrant not wired)")
        if not runner.start(root_ids=[req.root_id] if req.root_id else None, trigger="manual"):
            raise HTTPException(409, "a verify is already running")
        response.status_code = 202
        return {"started": True, "trigger": "manual"}
    raise HTTPException(400, "op must be 'reindex' or 'verify'")
