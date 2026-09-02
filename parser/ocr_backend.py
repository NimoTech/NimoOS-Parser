# parser/ocr_backend.py
"""Route rapidocr's onnxruntime sessions through the OpenVINO EP on GPU.

rapidocr v3 builds every det/cls/rec session via
ProviderConfig.get_ep_list(), which only knows CUDA/DML/CANN/CoreML — no
OpenVINO. Rather than forking rapidocr, patch that one method: when the
parser's resolved device is "gpu" AND this onnxruntime build ships the
OpenVINO EP, prepend it (device_type=GPU). ORT itself falls back to the CPU
EP when the GPU is unavailable at session build (verified in the
2026-09-01 spike), so a broken driver degrades instead of crashing — no
broken-mark bookkeeping needed here.
"""
import logging

log = logging.getLogger("parser.ocr_backend")

_gpu_wanted = False
_patched = False


def _ov_ep_available() -> bool:
    try:
        import onnxruntime as ort
        return "OpenVINOExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def set_gpu(enabled: bool) -> None:
    global _gpu_wanted, _patched
    if enabled and not _ov_ep_available():
        log.warning("OCR GPU requested but this onnxruntime build has no "
                    "OpenVINO EP; staying on CPU")
        enabled = False
    _gpu_wanted = enabled
    if _patched:
        return
    from rapidocr.inference_engine.onnxruntime.provider_config import (
        EP, ProviderConfig,
    )

    orig = ProviderConfig.get_ep_list

    def patched(self):
        if _gpu_wanted:
            return [("OpenVINOExecutionProvider", {"device_type": "GPU"}),
                    (EP.CPU_EP.value, self.cpu_ep_cfg())]
        return orig(self)

    ProviderConfig.get_ep_list = patched
    _patched = True
