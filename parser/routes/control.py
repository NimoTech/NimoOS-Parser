from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from parser.repo_state import get_state, set_paused, set_concurrency

router = APIRouter(prefix="/v1/parser/control", tags=["control"])


class ConcurrencyBody(BaseModel):
    n: int = Field(..., description="worker concurrency (1, 2, or 4)")


def _conn():
    from parser.main import app_state
    return app_state.conn


def _pool():
    from parser.main import app_state
    return app_state.worker_pool


@router.get("/state")
async def get_control_state() -> dict:
    return get_state(_conn())


@router.post("/pause")
async def pause() -> dict:
    set_paused(_conn(), True)
    pool = _pool()
    if pool is not None:
        await pool.pause()
    return {"paused": True}


@router.post("/resume")
async def resume() -> dict:
    set_paused(_conn(), False)
    pool = _pool()
    if pool is not None:
        await pool.resume()
    return {"paused": False}


@router.post("/concurrency")
async def set_pool_concurrency(body: ConcurrencyBody) -> dict:
    if body.n not in (1, 2, 4):
        raise HTTPException(status_code=400, detail="n must be 1, 2, or 4")
    set_concurrency(_conn(), body.n)
    pool = _pool()
    if pool is not None:
        await pool.set_concurrency(body.n)
    return {"concurrency": body.n}
