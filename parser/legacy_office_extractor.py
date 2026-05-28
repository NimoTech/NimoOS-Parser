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
    """Convert `src` to a modern Open XML file via `libreoffice --headless`.

    Returns the path to a freshly produced file inside a private outdir; the
    CALLER must `shutil.rmtree(result.parent, ignore_errors=True)` once it
    has consumed the file (we cannot do it ourselves — the caller is still
    reading the file when convert_legacy returns).

    Raises on any failure (non-zero exit, timeout, soffice ran but produced
    no output). Caller is expected to log and fall back to "skip" semantics
    rather than retry.
    """
    ext = Path(src).suffix.lower()
    if ext not in _TARGET:
        raise ValueError(f"unsupported legacy office ext: {ext!r}")
    target = _TARGET[ext]
    outdir = Path(tempfile.mkdtemp(prefix="lo-out-"))
    profile = Path(tempfile.mkdtemp(prefix="lo-prof-"))
    try:
        with _LO_GATE:
            cp = subprocess.run(
                [
                    "soffice",
                    f"-env:UserInstallation=file://{profile}",
                    "--headless",
                    "--convert-to", target,
                    "--outdir", str(outdir),
                    src,
                ],
                check=True,
                timeout=LO_TIMEOUT_SEC,
                capture_output=True,
            )
        produced = next(outdir.glob(f"*.{target}"), None)
        if produced is None:
            stdout = cp.stdout.decode("utf-8", errors="replace")
            stderr = cp.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"libreoffice produced no .{target} for {src} "
                f"(stdout={stdout!r}, stderr={stderr!r})"
            )
        return produced
    except Exception:
        # On failure clean up the outdir we created. Profile is always cleaned
        # in the finally below. We deliberately do NOT swallow — caller logs.
        shutil.rmtree(outdir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(profile, ignore_errors=True)
