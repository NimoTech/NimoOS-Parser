"""Knowledge-notes indexing endpoints. Mirrors agent_memory (sync bypass of
the job queue) but adds bm25 sparse vectors and delete-before-insert so an
edited note that shrinks never leaves stale chunks behind."""
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/v1/parser", tags=["knowledge-notes"])

# Fixed namespace → deterministic point ids (idempotent re-index).
_NS = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


class _Chunk(BaseModel):
    chunk_no: int
    text: str


class NotesUpsertRequest(BaseModel):
    user_id: str
    note_id: str
    note_type: str = "note"
    status: str = "draft"
    created_by: str = "human"
    updated_at: int = 0
    chunks: list[_Chunk]


class NotesQueryRequest(BaseModel):
    user_id: str
    query: str
    top_k: int = Field(10, ge=1, le=50)
    statuses: list[str] | None = None


class NotesDeleteRequest(BaseModel):
    user_id: str
    note_id: str


def _embed_batch(texts: list[str]) -> list[dict]:
    """Dense+sparse BGE-M3 embeddings in ONE encode call (test seam)."""
    from parser.routes.embed import get_bge_m3
    if not texts:
        return []
    return list(get_bge_m3().embed_text(texts))


@router.post("/notes/upsert")
async def notes_upsert(req: NotesUpsertRequest) -> dict:
    if not req.user_id or not req.note_id:
        raise HTTPException(status_code=400,
                            detail="user_id and note_id required")
    from parser.main import app_state
    # Delete-before-insert: edits may shrink the chunk count.
    app_state.qstore.delete_note(req.user_id, req.note_id)
    chunks = [c for c in req.chunks if c.text.strip()]
    if not chunks:
        return {"upserted": 0}
    embs = _embed_batch([c.text for c in chunks])
    points = []
    for c, emb in zip(chunks, embs):
        pid = str(uuid.uuid5(_NS, f"note:{req.user_id}:{req.note_id}:{c.chunk_no}"))
        points.append({
            "id": pid,
            "dense": emb["dense"],
            "sparse": emb["sparse"],
            "payload": {"user_id": req.user_id, "note_id": req.note_id,
                        "chunk_no": c.chunk_no, "text": c.text,
                        "type": req.note_type, "status": req.status,
                        "created_by": req.created_by,
                        "updated_at": req.updated_at},
        })
    app_state.qstore.upsert_notes(points)
    return {"upserted": len(points)}


@router.post("/notes/query")
async def notes_query(req: NotesQueryRequest) -> dict:
    if not req.user_id:
        raise HTTPException(status_code=400, detail="user_id required")
    from parser.main import app_state
    dense = _embed_batch([req.query])[0]["dense"]
    hits = app_state.qstore.query_notes(req.user_id, dense,
                                        limit=req.top_k,
                                        statuses=req.statuses)
    return {"hits": hits}


@router.post("/notes/delete")
async def notes_delete(req: NotesDeleteRequest) -> dict:
    if not req.user_id or not req.note_id:
        raise HTTPException(status_code=400,
                            detail="user_id and note_id required")
    from parser.main import app_state
    app_state.qstore.delete_note(req.user_id, req.note_id)
    return {"ok": True}
