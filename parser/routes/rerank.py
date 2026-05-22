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
    from parser.model_reranker import BGEReranker
    return BGEReranker.load()


@router.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest) -> RerankResponse:
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
