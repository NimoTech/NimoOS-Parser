"""On-demand docling extraction of a disk path → markdown.

Unlike the indexing pipeline this writes nothing to Qdrant/DB — it is a pure
read used by the agent's read_document(path=...) for files not yet indexed.
Security: the authoritative per-user check is the caller (AI layer's
visible_resources gate). This endpoint additionally bounds reads to the NAS
data roots so it can never read outside them regardless of caller.
"""
import os
import shutil
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/v1/parser/extract", tags=["extract"])

# Defense-in-depth: only files under these roots may be read. Module-level so
# tests can monkeypatch it. The NAS exposes user content under /DATA; external
# media mounts under /media and /mnt.
EXTRACT_ROOTS = ("/DATA", "/media", "/mnt")

EXTRACT_MAX_CHARS_DEFAULT = 40000


class ExtractRequest(BaseModel):
    path: str
    ocr: bool = False
    max_chars: int = Field(default=EXTRACT_MAX_CHARS_DEFAULT, ge=1)
    # Cap extraction to the document's first N pages — lets the distillation
    # worker bound both wall-clock time (a 100+ page PDF can blow the 120s
    # extract window on this CPU box) and the 96k-char worker budget without
    # wasting either on pages that would be truncated away anyway. Only
    # honored for PDFs (see `extract()` below) — docling silently ignores
    # `page_range` on other backends (verified: .docx conversion with
    # page_range=(1,1) still returns all pages, no error), so forwarding it
    # there would be a silent no-op, not a real cap.
    max_pages: Optional[int] = Field(default=None, ge=1)


def _safe_resolve(path: str) -> str:
    try:
        real = os.path.realpath(path)
    except (ValueError, OSError):
        raise HTTPException(status_code=400, detail="invalid path")
    if not any(real == r or real.startswith(r.rstrip(os.sep) + os.sep)
               for r in EXTRACT_ROOTS):
        raise HTTPException(status_code=403, detail="path outside allowed roots")
    if not os.path.isfile(real):
        raise HTTPException(status_code=404, detail="file not found")
    return real


@router.post("")
def extract(body: ExtractRequest) -> dict:
    # sync def → FastAPI runs it in a threadpool, so docling does not block the
    # event loop (and other Parser requests keep flowing).
    from parser.docling_extractor import (
        DoclingExtractor, is_docling_format, LEGACY_BINARY_OFFICE_EXTS,
    )

    real = _safe_resolve(body.path)
    ext = os.path.splitext(real)[1].lower()

    page_capped = False  # true only when we actually asked docling for a
    # page-restricted conversion (i.e. PDF + max_pages set) — distinct from
    # body.max_pages being set, since it's a no-op on non-PDF backends.
    total_pages: Optional[int] = None

    if is_docling_format(ext):
        from docling.exceptions import ConversionError
        page_range = None
        if body.max_pages is not None and ext == ".pdf":
            page_range = (1, body.max_pages)
            page_capped = True
        page_info: dict = {}
        try:
            markdown = DoclingExtractor.load(ocr=body.ocr).to_markdown(
                real, page_range=page_range, page_count_out=page_info)
        except (ConversionError, RuntimeError) as exc:
            # Corrupt/legacy files docling (or its LibreOffice fallback)
            # cannot parse are permanently broken — 4xx tells the caller
            # (distillation worker) not to retry.
            raise HTTPException(status_code=422,
                                detail=f"extraction failed: {exc}")
        total_pages = page_info.get("total_pages")
    elif ext in LEGACY_BINARY_OFFICE_EXTS:
        converted = None
        try:
            from parser.legacy_office_extractor import convert_legacy
            converted = convert_legacy(real)
            markdown = DoclingExtractor.load(ocr=body.ocr).to_markdown(str(converted))
        except HTTPException:
            raise
        except Exception as exc:  # legacy conversion / docling failure
            raise HTTPException(status_code=422,
                                detail=f"extraction failed: {exc}")
        finally:
            if converted is not None:
                shutil.rmtree(converted.parent, ignore_errors=True)
    else:
        # plain text / source / markdown — read directly.
        with open(real, "r", encoding="utf-8", errors="replace") as f:
            markdown = f.read()

    truncated = False
    if len(markdown) > body.max_chars:
        markdown = markdown[:body.max_chars]
        truncated = True

    if page_capped:
        if total_pages is not None:
            # docling's `input.page_count` reflects the source document's
            # real total (confirmed via experiment: it stays == the full
            # page count even when `page_range` restricts what's actually
            # converted), so this is an exact comparison, not a guess.
            if total_pages > body.max_pages:
                truncated = True
        else:
            # Docling didn't give us a page count for this backend/version —
            # approximate: assume the cap actually cut something whenever we
            # asked for one. Slightly pessimistic (may say truncated=True on
            # a document that happened to fit) but never silently hides a cut.
            truncated = True

    return {"path": real, "markdown": markdown, "truncated": truncated,
            "ocr": body.ocr}
