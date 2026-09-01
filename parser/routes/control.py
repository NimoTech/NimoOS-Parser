from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from parser.hardware import resolve_device
from parser.repo_state import get_state, set_paused, set_concurrency, set_device, set_ocr
from parser.text_backend import gpu_is_broken

router = APIRouter(prefix="/v1/parser/control", tags=["control"])


class ConcurrencyBody(BaseModel):
    n: int = Field(..., description="worker concurrency (1, 2, or 4)")


class DeviceBody(BaseModel):
    device: str = Field(..., description="auto | cuda | gpu | cpu")


class OcrBody(BaseModel):
    enabled: bool = Field(..., description="enable RapidOCR for scanned PDFs")


def _conn():
    from parser.main import app_state
    return app_state.conn


def _pool():
    from parser.main import app_state
    return app_state.worker_pool


def _resolved_device(device_pref: str) -> str:
    """resolve_device(), downgraded to "cpu" if the text backend has
    already hit an OV load failure this process - otherwise this would
    keep reporting "gpu" after a fallback that silently degraded to CPU.
    """
    resolved = resolve_device(device_pref)
    if resolved == "gpu" and gpu_is_broken():
        return "cpu"
    return resolved


@router.get("/state")
async def get_control_state() -> dict:
    s = get_state(_conn())
    # Expose what `auto` actually resolves to right now, so the UI can show
    # "Auto (cuda)" or "Auto (cpu)" without re-implementing the detection.
    s["resolved_device"] = _resolved_device(s["device"])
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
    if body.enabled:
        from parser import ocr_installer
        from parser.repo_state import get_state as _gs
        active = _gs(_conn())["ocr_model"]
        if not active or not ocr_installer.is_installed(active):
            raise HTTPException(
                status_code=409,
                detail="no OCR model installed; install and activate one "
                       "under /v1/parser/ocr/models first")
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
    if body.device not in ("auto", "cuda", "gpu", "cpu"):
        raise HTTPException(status_code=400, detail="device must be auto, cuda, gpu, or cpu")
    set_device(_conn(), body.device)
    # Drop cached models so they reload on the new device next call.
    from parser.text_backend import unload_all
    unload_all()
    return {
        "device": body.device,
        "resolved_device": _resolved_device(body.device),
    }
