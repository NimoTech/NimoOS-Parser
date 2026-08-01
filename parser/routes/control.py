from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from parser.hardware import resolve_device
from parser.repo_state import get_state, set_paused, set_concurrency, set_device, set_ocr

router = APIRouter(prefix="/v1/parser/control", tags=["control"])


class ConcurrencyBody(BaseModel):
    n: int = Field(..., description="worker concurrency (1, 2, or 4)")


class DeviceBody(BaseModel):
    device: str = Field(..., description="auto | cuda | cpu")


class OcrBody(BaseModel):
    enabled: bool = Field(..., description="enable RapidOCR for scanned PDFs")


def _conn():
    from parser.main import app_state
    return app_state.conn


def _pool():
    from parser.main import app_state
    return app_state.worker_pool


@router.get("/state")
async def get_control_state() -> dict:
    s = get_state(_conn())
    # Expose what `auto` actually resolves to right now, so the UI can show
    # "Auto (cuda)" or "Auto (cpu)" without re-implementing the detection.
    s["resolved_device"] = resolve_device(s["device"])
    return s


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


@router.post("/ocr")
async def set_pool_ocr(body: OcrBody) -> dict:
    """Toggle OCR (RapidOCR) for docling-converted PDFs in the indexing
    pipeline. Unloads the cached extractor so the next ingest reloads
    with the new OCR setting.
    """
    set_ocr(_conn(), body.enabled)
    from parser.docling_extractor import DoclingExtractor
    DoclingExtractor.unload()
    return {"ocr_enabled": body.enabled}


@router.post("/device")
async def set_pool_device(body: DeviceBody) -> dict:
    """Change inference device. Persists choice, then drops the cached
    model instances so the next embed/rerank request will reload on the
    new device. Reload itself is lazy (5-15s cold load on first request).
    """
    if body.device not in ("auto", "cuda", "cpu"):
        raise HTTPException(status_code=400, detail="device must be auto, cuda, or cpu")
    set_device(_conn(), body.device)
    # Drop cached models so they reload on the new device next call.
    from parser.model_bge_m3 import BGEM3
    from parser.model_reranker import BGEReranker
    BGEM3.unload()
    BGEReranker.unload()
    return {
        "device": body.device,
        "resolved_device": resolve_device(body.device),
    }
