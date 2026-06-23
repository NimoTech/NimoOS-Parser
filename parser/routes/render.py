"""Render PDF pages to PNG via docling's native page-image rendering.

Used by the agent's view_document_page for scanned/layout/figure questions a
text extract can't answer. No poppler/pdf2image — docling rasterizes pages
when generate_page_images is on. Pure read; writes nothing. Security: the
caller (AI layer) authorizes per-user; this endpoint bounds reads to the data
roots via extract._safe_resolve.
"""
import base64
import io
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from parser.routes.extract import _safe_resolve

router = APIRouter(prefix="/v1/parser/render", tags=["render"])

MAX_PAGES_PER_CALL = 8


class RenderRequest(BaseModel):
    path: str
    page_start: int = Field(default=1, ge=1)
    page_end: int = Field(default=1, ge=1)
    scale: float = Field(default=2.0, ge=1.0, le=4.0)


def _render_pdf_pages(path: str, start: int, end: int, scale: float) -> list[dict]:
    # Separate converter with page-image rendering ON — do NOT reuse the cached
    # text DoclingExtractor singleton (which renders no page images).
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    opts = PdfPipelineOptions()
    opts.images_scale = scale
    opts.generate_page_images = True
    conv = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)},
    )
    result = conv.convert(path)
    out: list[dict] = []
    for page_no, page in result.document.pages.items():
        if page_no < start or page_no > end:
            continue
        pil = page.image.pil_image
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        out.append({"page": page_no,
                    "png_b64": base64.b64encode(buf.getvalue()).decode("ascii")})
    out.sort(key=lambda p: p["page"])
    return out


@router.post("/pages")
def render_pages(body: RenderRequest) -> dict:
    # sync def → FastAPI threadpools it; docling rendering won't block the loop.
    real = _safe_resolve(body.path)
    if os.path.splitext(real)[1].lower() != ".pdf":
        raise HTTPException(status_code=400, detail="render supports PDF only")
    start = body.page_start
    end = min(body.page_end, start + MAX_PAGES_PER_CALL - 1)
    if end < start:
        raise HTTPException(status_code=400, detail="page_end < page_start")
    try:
        pages = _render_pdf_pages(real, start, end, body.scale)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"render failed: {exc}")
    return {"path": real, "pages": pages}
