"""Pick the text-model backend (OpenVINO GPU vs torch) for embed/rerank.

Single decision point consumed by routes/embed, routes/rerank, routes/test
and the indexing worker. Resolution: current_device(conn) == "gpu" -> the
OpenVINO classes; anything else -> the existing torch classes unchanged.
Any OV load failure logs a warning, marks that OV class broken for this
process (per class, not global - a broken embedder does not disable the
reranker's GPU path and vice versa; no per-request retry storm), unloads
the failed instance defensively, and falls back to torch on CPU, so a
missing IR or a broken driver degrades to exactly the pre-GPU behaviour.
"""
import logging

from parser.device import current_device
from parser.model_bge_m3 import BGEM3
from parser.model_bge_m3_ov import BGEM3OV
from parser.model_reranker import BGEReranker
from parser.model_reranker_ov import BGERerankerOV

log = logging.getLogger("parser.text_backend")

# Class names (BGEM3OV.__name__ / BGERerankerOV.__name__) whose OV load has
# failed at least once this process. Per-class so one broken model doesn't
# take the other's GPU path down with it.
_gpu_broken: set[str] = set()


def _load_with_fallback(ov_cls, torch_cls, conn):
    device = current_device(conn)
    if device == "gpu" and ov_cls.__name__ not in _gpu_broken:
        try:
            return ov_cls.load()
        except Exception:
            log.warning("OpenVINO GPU load failed for %s; falling back to "
                        "torch CPU for this process", ov_cls.__name__,
                        exc_info=True)
            _gpu_broken.add(ov_cls.__name__)
            try:
                ov_cls.unload()
            except Exception:
                log.warning("cleanup unload failed for %s after OV load "
                            "failure", ov_cls.__name__, exc_info=True)
            device = "cpu"
    elif device == "gpu":
        device = "cpu"
    return torch_cls.load(device=device)


def get_embedder(conn):
    return _load_with_fallback(BGEM3OV, BGEM3, conn)


def get_reranker(conn):
    return _load_with_fallback(BGERerankerOV, BGEReranker, conn)


def gpu_is_broken() -> bool:
    """True when any text OV class has been marked broken this process.

    /v1/parser/control/state has a single `resolved_device` field (not one
    per model), so callers that need to report GPU health downgrade that
    one field to "cpu" whenever this is True - see routes/control.py.
    """
    return bool(_gpu_broken)


def unload_all() -> None:
    """Drop every cached text model (device change / tests)."""
    global _gpu_broken
    for klass in (BGEM3, BGEM3OV, BGEReranker, BGERerankerOV):
        klass.unload()
    _gpu_broken = set()
