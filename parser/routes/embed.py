from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/v1/parser", tags=["embed"])


class EmbedRequest(BaseModel):
    model: str
    input_type: str
    text: str | None = None
    image_b64: str | None = None


class EmbedResponse(BaseModel):
    dense: list[float] | None = None
    sparse: dict | None = None
    dim: int
    model_version: str


def get_bge_m3():
    from parser.main import app_state
    from parser.text_backend import get_embedder
    return get_embedder(app_state.conn)


# Deliberately a plain `def`, not `async def`: embedding runs the tokenizer +
# OpenVINO/torch inference synchronously (same reasoning as routes/rerank.py -
# see the comment there). As an `async def` it ran inline on the event loop
# and froze the whole service for the call's duration; FastAPI runs a sync
# handler in its threadpool instead, so the loop stays responsive.
@router.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    if req.model == "bge-m3":
        if req.input_type != "text":
            raise HTTPException(400, "bge-m3 only supports input_type=text")
        if not req.text:
            raise HTTPException(400, "text required for input_type=text")
        m = get_bge_m3()
        out = m.embed_text([req.text])[0]
        return EmbedResponse(
            dense=out["dense"], sparse=out["sparse"],
            dim=m.dim, model_version=m.version,
        )
    raise HTTPException(400, f"unknown model: {req.model}")
