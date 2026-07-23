"""backendselect.py 的硬件探测/打分/工厂/回退包装器测试。

三个探测函数(_probe_openvino/_probe_nvidia/_probe_amd)全部 monkeypatch
替身,不依赖真实硬件/驱动;build_backend/select_caption_backend 只构造
后端实例(懒加载,构造阶段不 import openvino/llama_cpp),因此本机没装
这两个可选依赖也能跑通。
"""
import logging

import pytest

import parser.backendselect as bs


# ---------------------------------------------------------------------------
# probe_hardware + rank
# ---------------------------------------------------------------------------

def test_intel_dgpu_selected(monkeypatch):
    monkeypatch.setattr(bs, "_probe_openvino",
        lambda: [{"runtime": "openvino", "device": "GPU.1", "tier": 30, "label": "Arc B60"},
                 {"runtime": "openvino", "device": "CPU", "tier": 10, "label": "CPU"}])
    monkeypatch.setattr(bs, "_probe_nvidia", lambda: [])
    monkeypatch.setattr(bs, "_probe_amd", lambda: [])
    ranked = bs.rank(bs.probe_hardware())
    assert (ranked[0]["runtime"], ranked[0]["device"]) == ("openvino", "GPU.1")


def test_intel_arc_beats_nvidia_when_tied(monkeypatch):
    # Intel Arc dGPU(30) 与 NVIDIA dGPU(30) 并存 → 同 tier,openvino 优先
    monkeypatch.setattr(bs, "_probe_openvino",
        lambda: [{"runtime": "openvino", "device": "GPU.0", "tier": 30, "label": "Arc B60"},
                 {"runtime": "openvino", "device": "CPU", "tier": 10, "label": "CPU"}])
    monkeypatch.setattr(bs, "_probe_nvidia",
        lambda: [{"runtime": "llamacpp", "device": "cuda", "tier": 30, "label": "RTX 4090"}])
    monkeypatch.setattr(bs, "_probe_amd", lambda: [])
    ranked = bs.rank(bs.probe_hardware())
    assert (ranked[0]["runtime"], ranked[0]["device"]) == ("openvino", "GPU.0")


def test_amd_only_picks_vulkan(monkeypatch):
    # openvino 只探到 CPU(10),AMD 探到 gfx → llamacpp:vulkan(iGPU 20) 胜出
    monkeypatch.setattr(bs, "_probe_openvino",
        lambda: [{"runtime": "openvino", "device": "CPU", "tier": 10, "label": "CPU"}])
    monkeypatch.setattr(bs, "_probe_nvidia", lambda: [])
    monkeypatch.setattr(bs, "_probe_amd",
        lambda: [{"runtime": "llamacpp", "device": "vulkan", "tier": 20, "label": "AMD gfx1100"}])
    ranked = bs.rank(bs.probe_hardware())
    assert (ranked[0]["runtime"], ranked[0]["device"]) == ("llamacpp", "vulkan")


def test_no_gpu_falls_to_openvino_cpu(monkeypatch):
    monkeypatch.setattr(bs, "_probe_openvino",
        lambda: [{"runtime": "openvino", "device": "CPU", "tier": 10, "label": "CPU"}])
    monkeypatch.setattr(bs, "_probe_nvidia", lambda: [])
    monkeypatch.setattr(bs, "_probe_amd", lambda: [])
    ranked = bs.rank(bs.probe_hardware())
    assert (ranked[0]["runtime"], ranked[0]["device"]) == ("openvino", "CPU")


# ---------------------------------------------------------------------------
# select_caption_backend(显式指定 / 非法值回退)
# ---------------------------------------------------------------------------

def test_explicit_device_respected(tmp_path):
    b = bs.select_caption_backend(
        vlm_device="llamacpp:cpu",
        model_path=tmp_path,
        gguf_path=tmp_path / "m.gguf",
        mmproj_path=tmp_path / "mm.gguf",
        idle_ttl_s=60,
    )
    assert isinstance(b, bs.SelectingCaptionBackend)
    assert type(b._backend).__name__ == "LlamaCppCaptionBackend"
    assert b._backend.backend_tag == "cpu"
    assert b._backend.n_gpu_layers == 0


def test_illegal_device_falls_to_auto(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(bs, "_probe_openvino",
        lambda: [{"runtime": "openvino", "device": "CPU", "tier": 10, "label": "CPU"}])
    monkeypatch.setattr(bs, "_probe_nvidia", lambda: [])
    monkeypatch.setattr(bs, "_probe_amd", lambda: [])
    with caplog.at_level(logging.WARNING, logger="parser.backendselect"):
        b = bs.select_caption_backend(
            vlm_device="not-a-real-spec",
            model_path=tmp_path,
            gguf_path=tmp_path / "m.gguf",
            mmproj_path=tmp_path / "mm.gguf",
            idle_ttl_s=60,
        )
    assert isinstance(b, bs.SelectingCaptionBackend)
    assert type(b._backend).__name__ == "OpenVINOCaptionBackend"
    assert b._backend.device == "CPU"
    assert len(caplog.records) >= 1  # 非法值必须 Warn


# ---------------------------------------------------------------------------
# SelectingCaptionBackend 回退包装器
# ---------------------------------------------------------------------------

def test_selecting_backend_fallback(monkeypatch, tmp_path, caplog):
    """首选候选加载即抛 CaptionError → 降级到下一候选并重试成功。"""

    class _Bad:
        def caption(self, image_bytes):
            raise bs.CaptionError("simulated load failure")

        def unload(self):
            pass

        @property
        def is_loaded(self):
            return False

        version = "bad/backend"

    class _Good:
        def caption(self, image_bytes):
            return "a good caption"

        def unload(self):
            pass

        @property
        def is_loaded(self):
            return True

        version = "good/backend"

    built = []

    def fake_build_backend(spec, **kwargs):
        built.append(spec)
        return _Bad() if spec == "openvino:GPU.1" else _Good()

    monkeypatch.setattr(bs, "build_backend", fake_build_backend)

    ranked = [
        {"runtime": "openvino", "device": "GPU.1", "tier": 30, "label": "bad"},
        {"runtime": "llamacpp", "device": "cuda", "tier": 30, "label": "good"},
    ]
    with caplog.at_level(logging.WARNING, logger="parser.backendselect"):
        wrapper = bs.SelectingCaptionBackend(
            ranked,
            model_path=tmp_path,
            gguf_path=tmp_path / "m.gguf",
            mmproj_path=tmp_path / "mm.gguf",
            idle_ttl_s=60,
        )
        out = wrapper.caption(b"fake-bytes")

    assert out == "a good caption"
    assert built[0] == "openvino:GPU.1"
    assert built[1] == "llamacpp:cuda"
    assert len(caplog.records) >= 1  # 降级必须 Warn
    assert wrapper.version == "good/backend"
    assert wrapper.is_loaded is True
