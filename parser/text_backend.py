"""Pick the text-model backend (OpenVINO GPU vs torch) for embed/rerank.

Single decision point consumed by routes/embed, routes/rerank, routes/test
and the indexing worker. Resolution: current_device(conn) == "gpu" -> the
OpenVINO classes; anything else -> the existing torch classes unchanged.
Any OV load failure logs a warning, marks the GPU broken for this process
(no per-request retry storm) and falls back to torch on CPU, so a missing
IR or a broken driver degrades to exactly the pre-GPU behaviour.
"""
import logging

from parser.device import current_device
from parser.model_bge_m3 import BGEM3
from parser.model_bge_m3_ov import BGEM3OV
from parser.model_reranker import BGEReranker
from parser.model_reranker_ov import BGERerankerOV

log = logging.getLogger("parser.text_backend")

_gpu_broken = False


def _load_with_fallback(ov_cls, torch_cls, conn):
    global _gpu_broken
    device = current_device(conn)
    if device == "gpu" and not _gpu_broken:
        try:
            return ov_cls.load()
        except Exception:
            log.warning("OpenVINO GPU load failed for %s; falling back to "
                        "torch CPU for this process", ov_cls.__name__,
                        exc_info=True)
            _gpu_broken = True
            device = "cpu"
    elif device == "gpu":
        device = "cpu"
    return torch_cls.load(device=device)


def get_embedder(conn):
    return _load_with_fallback(BGEM3OV, BGEM3, conn)


def get_reranker(conn):
    return _load_with_fallback(BGERerankerOV, BGEReranker, conn)


def unload_all() -> None:
    """Drop every cached text model (device change / tests)."""
    global _gpu_broken
    for klass in (BGEM3, BGEM3OV, BGEReranker, BGERerankerOV):
        klass.unload()
    _gpu_broken = False
