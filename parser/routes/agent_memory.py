import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/v1/parser", tags=["agent-memory"])

# Fixed namespace → deterministic point ids (idempotent re-index).
_NS = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


class _Chunk(BaseModel):
    chunk_no: int
    text: str
    created_at: int


class UpsertRequest(BaseModel):
    user_id: str
    session_id: str
    chunks: list[_Chunk]


class QueryRequest(BaseModel):
    user_id: str
    query: str
    top_k: int = Field(5, ge=1, le=50)


def _embed_dense_batch(texts: list[str]) -> list[list[float]]:
    """Dense BGE-M3 embeddings for a batch of texts in ONE encode call
    (seam for tests to monkeypatch). Avoids per-chunk sync loops blocking the
    event loop / OOM on large batches."""
    from parser.routes.embed import get_bge_m3
    if not texts:
        return []
    return [r["dense"] for r in get_bge_m3().embed_text(texts)]


@router.post("/agent-memory/upsert")
async def agent_memory_upsert(req: UpsertRequest) -> dict:
    if not req.user_id:
        raise HTTPException(status_code=400, detail="user_id required")
    from parser.main import app_state
    chunks = [c for c in req.chunks if c.text.strip()]
    if not chunks:
        return {"upserted": 0}
    denses = _embed_dense_batch([c.text for c in chunks])   # one batched encode
    points = []
    for c, dense in zip(chunks, denses):
        pid = str(uuid.uuid5(_NS, f"{req.user_id}:{req.session_id}:{c.chunk_no}"))
        points.append({
            "id": pid,
            "dense": dense,
            "payload": {"user_id": req.user_id, "session_id": req.session_id,
                        "chunk_no": c.chunk_no, "text": c.text,
                        "created_at": c.created_at},
        })
    app_state.qstore.upsert_agent_memory(points)
    return {"upserted": len(points)}


@router.post("/agent-memory/query")
async def agent_memory_query(req: QueryRequest) -> dict:
    if not req.user_id:
        raise HTTPException(status_code=400, detail="user_id required")
    from parser.main import app_state
    dense = _embed_dense_batch([req.query])[0]
    hits = app_state.qstore.query_agent_memory(req.user_id, dense,
                                               limit=req.top_k)
    return {"hits": hits}
