import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/v1/parser", tags=["rerank"])

MAX_CANDIDATES = 128


class Candidate(BaseModel):
    id: str
    text: str


class RerankRequest(BaseModel):
    model: str
    query: str
    candidates: list[Candidate]
    top_k: int | None = None


class Score(BaseModel):
    id: str
    score: float


class RerankResponse(BaseModel):
    scores: list[Score]
    model_version: str
    took_ms: int


def get_reranker():
    from parser.main import app_state
    from parser.text_backend import get_reranker as _get_reranker
    return _get_reranker(app_state.conn)


# Deliberately a plain `def`, not `async def`: reranking is CPU-bound and
# synchronous (~26s for 20 real chunks on CPU). As an `async def` it ran inline
# on the event loop and froze the whole service for its duration - every other
# endpoint (stats, healthz, control/state, and the path expansion Search issues
# right after) timed out. FastAPI runs a sync handler in its threadpool instead,
# so the loop stays responsive. BGEReranker serialises the inference itself.
@router.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    if req.model != "bge-reranker-v2-m3":
        raise HTTPException(400, f"unknown reranker: {req.model}")
    if len(req.candidates) > MAX_CANDIDATES:
        raise HTTPException(400, f"too many candidates (max {MAX_CANDIDATES})")
    rr = get_reranker()
    t0 = time.time()
    raw = rr.rerank(req.query, [c.model_dump() for c in req.candidates])
    raw.sort(key=lambda s: -s["score"])
    if req.top_k is not None:
        raw = raw[: req.top_k]
    return RerankResponse(
        scores=[Score(**s) for s in raw],
        model_version=rr.version,
        took_ms=int((time.time() - t0) * 1000),
    )
