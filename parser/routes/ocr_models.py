import time

from fastapi import APIRouter, HTTPException

from parser import ocr_installer
from parser.ocr_catalog import REGISTRY_TAG, entries, get_entry
from parser.repo_models import register_model, set_active
from parser.repo_state import get_state, set_ocr_model

router = APIRouter(prefix="/v1/parser", tags=["ocr-models"])

_SIZE_NOTES = {"fast": "~20 MB", "accurate": "~200 MB", "balanced": "~100 MB"}


def get_conn():
    from parser.main import app_state
    return app_state.conn


@router.get("/ocr/models")
async def list_models() -> dict:
    active = get_state(get_conn())["ocr_model"]
    prog = ocr_installer.snapshot()
    out = []
    for e in entries():
        row = prog.get(e["id"], {})
        out.append({
            "id": e["id"], "name": e["name"], "langs": e["langs"],
            "profile": e["profile"], "recommended": e["recommended"],
            "size_note": _SIZE_NOTES[e["profile"]],
            "installed": ocr_installer.is_installed(e["id"]),
            "active": e["id"] == active,
            "status": row.get("status", "idle"),
            "progress_pct": row.get("progress_pct", 0),
            "error": row.get("error"),
        })
    return {"models": out}


@router.post("/ocr/models/{model_id}/install")
async def install_model(model_id: str) -> dict:
    result = ocr_installer.start_install(model_id)
    if result == "unknown":
        raise HTTPException(status_code=404, detail="unknown OCR model")
    return {"result": result}


@router.post("/ocr/models/{model_id}/activate")
async def activate_model(model_id: str) -> dict:
    if get_entry(model_id) is None:
        raise HTTPException(status_code=404, detail="unknown OCR model")
    if not ocr_installer.is_installed(model_id):
        raise HTTPException(status_code=409, detail="model not installed")
    conn = get_conn()
    register_model(conn, name=model_id, version=REGISTRY_TAG, modality="ocr",
                   dim=None, registered_at=int(time.time() * 1000))
    set_active(conn, model_id, REGISTRY_TAG)
    set_ocr_model(conn, model_id)
    from parser.docling_extractor import DoclingExtractor
    DoclingExtractor.invalidate()
    return {"ocr_model": model_id}


@router.delete("/ocr/models/{model_id}")
async def delete_model(model_id: str) -> dict:
    if get_entry(model_id) is None:
        raise HTTPException(status_code=404, detail="unknown OCR model")
    if get_state(get_conn())["ocr_model"] == model_id:
        raise HTTPException(status_code=409, detail="model is active")
    ocr_installer.remove(model_id)
    return {"removed": model_id}
