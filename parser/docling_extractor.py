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
DOCLING_EXTS = {
    ".pdf",
    ".docx", ".doc",
    ".pptx", ".ppt",
    ".xlsx", ".xls",
    ".html", ".htm",
}


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
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        pdf_opts = PdfPipelineOptions()
        pdf_opts.do_ocr = ocr
        pdf_opts.do_table_structure = True

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

    def to_markdown(self, source) -> str:
        """Convert a file path or bytes to markdown.

        Accepts either a `pathlib.Path` / str path, or a `DocumentStream`-like
        object for in-memory bytes (used by the sandbox endpoint to avoid
        writing user uploads to disk).
        """
        result = self._converter.convert(source)
        return result.document.export_to_markdown()


def is_docling_format(ext: str) -> bool:
    return ext.lower() in DOCLING_EXTS
