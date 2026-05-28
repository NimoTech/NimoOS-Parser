"""Convert legacy OLE office formats to modern Open XML via LibreOffice headless.

The converted file is then fed into the existing docling pipeline. LibreOffice
takes a global lock on its UserInstallation profile (`~/.config/libreoffice`)
by default — concurrent soffice processes deadlock on it. This module uses a
private profile per call (`-env:UserInstallation=file:///tmp/lo-prof-<rand>`)
plus a per-call output directory to avoid that, and additionally serializes
all soffice spawns through a module-level lock so multi-worker parser jobs
don't simultaneously fork four ~200MB processes.
"""
import logging
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

log = logging.getLogger("parser.legacy_office_extractor")

LO_TIMEOUT_SEC = 120

# Map source ext → output Open XML target. .wps is the Kingsoft WPS Writer
# format; in practice it's a Word 97-2003 OLE container that LibreOffice
# reads as .doc. Targeting docx lets the same downstream docling path handle
# it; rare WPS-private extensions will fail conversion and fall through to
# the empty-chunks branch in pipeline_text.
_TARGET = {
    ".doc": "docx", ".wps": "docx",
    ".ppt": "pptx",
    ".xls": "xlsx",
}

# Serialize soffice spawns: each is ~200MB RAM and ~3s startup. With the
# parser's default concurrency=4 we don't want four LibreOffice processes
# spinning up simultaneously. docling and other format paths keep their
# existing concurrency — this lock only gates soffice.
_LO_GATE = threading.Lock()


def is_legacy_binary_office(ext: str) -> bool:
    return ext.lower() in _TARGET


def convert_legacy(src: str) -> Path:
    """Stub — implemented in Task 2."""
    raise NotImplementedError
