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
    """Force-rescan a Root.

    `op=reindex` — enqueue a reindex job per known path under this root.
    `op=verify` — NOT YET IMPLEMENTED. Future: walk the root, sha256 each
      file, compare with file_records, enqueue reindex only for drifted ones.
      Returning 501 here prevents the silent downgrade-to-index footgun.
    """
    if req.op == "reindex":
        conn = get_conn()
        rows = list_paths_under_root(conn, req.root_id)
        now = int(time.time() * 1000)
        for r in rows:
            enqueue_job(conn, root_id=r["root_id"], path=r["path"],
                        op="reindex", priority=500, now_ms=now)
        return {"queued": len(rows)}
    if req.op == "verify":
        raise HTTPException(
            501, "op='verify' is not implemented; use op='reindex' for now"
        )
    raise HTTPException(400, "op must be 'reindex' or 'verify'")
