"""visual 入库路由。音频(audio)命名空间本期只预留不注册,见 spec §2.1。"""
import json
import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from parser.repo_jobs import enqueue_job

log = logging.getLogger("parser.routes.visual")

router = APIRouter(prefix="/v1/parser/visual", tags=["visual"])

VISUAL_JOB_PRIORITY = 200  # 文档 job 默认 100:文档优先,照片补扫殿后


class IngestReq(BaseModel):
    source: str = Field(min_length=1, pattern=r"^[a-z0-9_-]+$")
    asset_id: str = Field(min_length=1, max_length=128)
    image_path: str
    mime: str = "image/jpeg"
    meta: dict = Field(default_factory=dict)


def _allowed_dirs() -> list[Path]:
    from parser.main import app_state
    if app_state.settings is not None:
        raw = app_state.settings.visual_allowed_dirs
    else:  # 测试夹具跳过 settings 时读环境/默认值,与生产同一解析路径
        from parser.config import load_settings
        raw = load_settings().visual_allowed_dirs
    return [Path(x.strip()) for x in raw.split(",") if x.strip()]


def _validate_path(image_path: str) -> Path:
    p = Path(image_path).resolve()  # resolve 吃掉 ../ 穿越
    if not any(p.is_relative_to(d) for d in _allowed_dirs()):
        raise HTTPException(400, "image_path outside allowed directories")
    if not p.is_file():
        raise HTTPException(400, "image_path not found or not a file")
    return p


@router.post("/ingest", status_code=202)
def ingest(req: IngestReq):
    from parser.main import app_state
    p = _validate_path(req.image_path)
    job_id = enqueue_job(
        app_state.conn, root_id=req.source, path=str(p), op="visual_ingest",
        priority=VISUAL_JOB_PRIORITY,
        sub_modality=json.dumps(
            {"asset_id": req.asset_id, "mime": req.mime, "meta": req.meta},
            ensure_ascii=False),
        now_ms=int(time.time() * 1000),
    )
    return {"job_id": job_id}


@router.delete("/assets/{source}/{asset_id}")
def delete_asset(source: str, asset_id: str):
    from parser.main import app_state
    if getattr(app_state, "visual_pipeline", None) is None:
        raise HTTPException(503, "visual pipeline unavailable (qdrant down?)")
    app_state.visual_pipeline.delete_asset(source=source, asset_id=asset_id)
    return {"deleted": True}
