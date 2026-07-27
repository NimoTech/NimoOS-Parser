"""visual 入库路由。音频(audio)命名空间本期只预留不注册,见 spec §2.1。"""
import json
import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
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


@router.get("/captions")
def export_captions(
    source: str = Query(min_length=1, pattern=r"^[a-z0-9_-]+$"),
    limit: int = 512,
    offset: str | None = None,
):
    """批量导出某 source 下的 caption(scroll 分页游标)。
    供 Photos 周期 diff 拉取(智能时刻策展地基),存量增量同路径:首次全量拉
    完即为存量,之后按 mtime_ms 在 Photos 侧自行做增量判断。"""
    from parser.main import app_state
    if app_state.qstore is None:
        raise HTTPException(503, "qdrant unavailable (qstore not ready)")
    limit = max(1, min(limit, 1024))
    offset = offset or None  # 契约里 `offset=`(空字符串)等价首页,归一化成 None 再往下传
    prefix = f"{source}:"
    try:
        points, next_offset = app_state.qstore.scroll_captions(
            source, limit, offset)
    except Exception as exc:
        # 与 delete_asset 同一错误语义:瞬断交由调用方(Photos)收到 503 后重试。
        raise HTTPException(503, f"qdrant unavailable, retry later: {exc}")
    items = []
    for p in points:
        file_id = p.get("file_id", "")
        if not file_id.startswith(prefix):
            continue
        items.append({
            "asset_id": file_id[len(prefix):],
            "text": p.get("text", ""),
            "mtime_ms": p.get("mtime_ms"),
        })
    return {
        "items": items,
        "next_offset": None if next_offset is None else str(next_offset),
    }


@router.delete("/assets/{source}/{asset_id}")
def delete_asset(source: str, asset_id: str):
    from parser.main import app_state
    if getattr(app_state, "visual_pipeline", None) is None:
        raise HTTPException(503, "visual pipeline unavailable (qdrant down?)")
    try:
        app_state.visual_pipeline.delete_asset(source=source, asset_id=asset_id)
    except Exception as exc:
        # Qdrant 启动后中途瞬断:与上面"pipeline 未接线"分支同一错误语义,
        # 删除不走 job 队列没有自动重试,交由调用方(Photos)收到 503 后重试。
        raise HTTPException(503, f"qdrant unavailable, retry later: {exc}")
    return {"deleted": True}
