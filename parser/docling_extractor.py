"""Convert non-text formats (PDF, DOCX, PPTX, XLSX, HTML) to markdown via docling.

Pipeline:
  pdf/docx/pptx/xlsx/html → DocumentConverter → DoclingDocument → markdown
  → chunk_markdown (existing path)

The converter is loaded lazily and cached as a singleton, identical to the
BGEM3 model lifecycle. OCR is opt-in (default off) because it adds 5–10x
runtime and pulls extra models.
"""
from pathlib import Path
from typing import Optional

# Extensions that benefit from docling. .md/.txt/source code are NOT here —
# they go through the existing raw-read + chunk_text path which is faster.
# Legacy binary OLE office formats (.doc/.ppt/.xls/.wps) are NOT in this set
# either: docling can't read them. They are handled by pipeline_text via
# `parser.legacy_office_extractor.convert_legacy`, which shells out to
# `libreoffice --headless --convert-to <docx|pptx|xlsx>` and then re-enters
# this docling path with the converted file.
DOCLING_EXTS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html", ".htm",
}

# Canonical set of legacy OLE office extensions. Owned here; consumed by
# pipeline_text._run_full to route through legacy_office_extractor.
LEGACY_BINARY_OFFICE_EXTS = {".doc", ".ppt", ".xls", ".wps"}


class DoclingExtractor:
    """Singleton wrapping docling's DocumentConverter."""

    _instance: Optional["DoclingExtractor"] = None
    _ocr_enabled: bool = False
    version = "docling/v1"

    def __init__(self, converter, ocr: bool) -> None:
        self._converter = converter
        self._ocr = ocr

    @classmethod
    def load(cls, *, ocr: bool = False) -> "DoclingExtractor":
        if cls._instance is not None and cls._ocr_enabled == ocr:
            return cls._instance
        cls.unload()

        # Import only when first used: docling pulls torch + transformers,
        # which is heavy. Service startup should not pay this cost.
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions, RapidOcrOptions,
        )

        pdf_opts = PdfPipelineOptions()
        pdf_opts.do_ocr = ocr
        pdf_opts.do_table_structure = True
        if ocr:
            # RapidOCR with simplified Chinese + English. Native-Chinese
            # engine, ONNX-runtime so it doesn't bloat the torch GPU
            # context. force_full_page_ocr=False means docling first tries
            # native text extraction and only OCRs regions without text —
            # good for hybrid PDFs (native + scanned pages).
            pdf_opts.ocr_options = RapidOcrOptions(
                lang=["chinese_sim", "english"],
                force_full_page_ocr=False,
            )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
            },
        )
        cls._instance = cls(converter, ocr)
        cls._ocr_enabled = ocr
        return cls._instance

    @classmethod
    def unload(cls) -> None:
        cls._instance = None
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        from parser.memutil import trim_malloc
        trim_malloc()

    def to_markdown(self, source, *, page_range: Optional[tuple] = None,
                     page_count_out: Optional[dict] = None) -> str:
        """Convert a file path or bytes to markdown.

        Accepts either a `pathlib.Path` / str path, or a `DocumentStream`-like
        object for in-memory bytes (used by the sandbox endpoint to avoid
        writing user uploads to disk).

        `page_range`: pass-through to docling's `convert(page_range=...)`
        (1-indexed, inclusive `(first, last)`). Only the PDF backend honors
        it — callers are responsible for gating by extension (see
        `parser/routes/extract.py`), this method just forwards whatever it's
        given.

        `page_count_out`: optional caller-owned dict; if given, this fills in
        `page_count_out["total_pages"]` with the source document's *total*
        page count (`result.input.page_count`, populated regardless of
        `page_range`) so callers can tell whether a `page_range` cap actually
        cut anything. This is an out-param rather than a return-value change
        or instance attribute so existing callers' return type (`str`) is
        untouched and so nothing is stored on this singleton — it is shared
        across concurrent threadpool requests (see extract.py), and any
        instance-attribute would race.
        """
        kwargs = {}
        if page_range is not None:
            kwargs["page_range"] = page_range
        result = self._converter.convert(source, **kwargs)
        if page_count_out is not None:
            page_count_out["total_pages"] = getattr(
                result.input, "page_count", None)
        return result.document.export_to_markdown()


def is_docling_format(ext: str) -> bool:
    return ext.lower() in DOCLING_EXTS
