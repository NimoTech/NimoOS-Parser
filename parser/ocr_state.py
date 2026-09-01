"""Resolve the effective OCR configuration for a parse run.

Single place that turns persisted state (ocr_enabled + ocr_model + device
preference) into what DoclingExtractor.load actually needs. Missing model
files (disk tampering, half-restored backup) degrade to OCR-off for the run
instead of failing the file — the warning is the operator's breadcrumb.
"""
import logging

from parser import ocr_installer
from parser.device import current_device
from parser.repo_state import get_state

log = logging.getLogger("parser.ocr_state")


def resolve_ocr(conn) -> tuple[bool, str | None, bool]:
    state = get_state(conn)
    if not state.get("ocr_enabled", False):
        return (False, None, False)
    model_id = state.get("ocr_model", "")
    if not model_id or not ocr_installer.is_installed(model_id):
        log.warning("OCR enabled but model %r is not installed; parsing "
                    "without OCR for this run", model_id)
        return (False, None, False)
    use_gpu = current_device(conn) == "gpu"
    return (True, str(ocr_installer.model_dir(model_id)), use_gpu)
