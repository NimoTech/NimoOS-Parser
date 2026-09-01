# tests/test_ocr_backend.py
"""The shim only reorders EPs; ORT itself degrades OV->CPU at session build,
so there is deliberately no broken-mark machinery here (spec amendment #4)."""
from rapidocr.inference_engine.onnxruntime.provider_config import ProviderConfig

from parser import ocr_backend


def _cfg():
    # ProviderConfig reads attribute-style engine cfg; OmegaConf mirrors what
    # rapidocr passes at runtime.
    from omegaconf import OmegaConf
    return OmegaConf.create(
        {"use_cuda": False, "use_dml": False, "use_cann": False,
         "use_coreml": False, "cpu_ep_cfg": {}, "cuda_ep_cfg": {},
         "dml_ep_cfg": None, "cann_ep_cfg": {}, "coreml_ep_cfg": {}})


def test_gpu_on_injects_openvino_ep(monkeypatch):
    monkeypatch.setattr(ocr_backend, "_ov_ep_available", lambda: True)
    ocr_backend.set_gpu(True)
    try:
        eps = ProviderConfig(engine_cfg=_cfg()).get_ep_list()
        assert eps[0][0] == "OpenVINOExecutionProvider"
        assert eps[0][1] == {"device_type": "GPU"}
        assert eps[-1][0] == "CPUExecutionProvider"
    finally:
        ocr_backend.set_gpu(False)


def test_gpu_off_leaves_default(monkeypatch):
    monkeypatch.setattr(ocr_backend, "_ov_ep_available", lambda: True)
    ocr_backend.set_gpu(False)
    eps = ProviderConfig(engine_cfg=_cfg()).get_ep_list()
    assert all(name != "OpenVINOExecutionProvider" for name, _ in eps)


def test_gpu_on_without_ov_build_stays_default(monkeypatch):
    monkeypatch.setattr(ocr_backend, "_ov_ep_available", lambda: False)
    ocr_backend.set_gpu(True)
    try:
        eps = ProviderConfig(engine_cfg=_cfg()).get_ep_list()
        assert all(name != "OpenVINOExecutionProvider" for name, _ in eps)
    finally:
        ocr_backend.set_gpu(False)
