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
    from parser.device import current_device
    from parser.main import app_state
    from parser.model_bge_m3 import BGEM3
    return BGEM3.load(device=current_device(app_state.conn))


@router.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
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
