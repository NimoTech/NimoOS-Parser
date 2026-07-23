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
        def __init__(self):
            self.unloaded = False

        def caption(self, image_bytes):
            raise bs.CaptionError("simulated load failure")

        def unload(self):
            self.unloaded = True

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
    bad_instances = []

    def fake_build_backend(spec, **kwargs):
        built.append(spec)
        if spec == "openvino:GPU.1":
            bad = _Bad()
            bad_instances.append(bad)
            return bad
        return _Good()

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
    assert bad_instances[0].unloaded is True  # _demote 必须先 unload 旧后端再重建


def test_selecting_backend_no_demote_on_inference_failure(monkeypatch, tmp_path, caplog):
    """已加载后端单次推理失败(坏图)不应触发降级,异常原样抛出。"""

    class _LoadsThenBadImage:
        """模拟 is_loaded False→True 变化:首次调用加载成功并返回结果,
        之后(已加载状态下)推理失败,抛出 CaptionError。"""

        def __init__(self):
            self._loaded = False
            self.unloaded = False
            self.call_count = 0

        @property
        def is_loaded(self):
            return self._loaded

        def caption(self, image_bytes):
            self.call_count += 1
            if not self._loaded:
                self._loaded = True
                return "first good caption"
            raise bs.CaptionError("simulated inference failure on bad image")

        def unload(self):
            self.unloaded = True
            self._loaded = False

        version = "loaded-then-bad-image/backend"

    stub = _LoadsThenBadImage()
    built = []

    def fake_build_backend(spec, **kwargs):
        built.append(spec)
        return stub

    monkeypatch.setattr(bs, "build_backend", fake_build_backend)

    # 链上还有下一个候选可供降级——如果错误地降级,测试也能捕获到。
    ranked = [
        {"runtime": "openvino", "device": "GPU.1", "tier": 30, "label": "arc"},
        {"runtime": "llamacpp", "device": "cuda", "tier": 30, "label": "other"},
    ]
    wrapper = bs.SelectingCaptionBackend(
        ranked,
        model_path=tmp_path,
        gguf_path=tmp_path / "m.gguf",
        mmproj_path=tmp_path / "mm.gguf",
        idle_ttl_s=60,
    )

    # 第一次调用:加载成功并返回结果。
    out = wrapper.caption(b"good-image")
    assert out == "first good caption"
    assert stub.is_loaded is True

    # 第二次调用:后端已加载,这次是单张坏图推理失败——不应降级。
    with caplog.at_level(logging.WARNING, logger="parser.backendselect"):
        with pytest.raises(bs.CaptionError):
            wrapper.caption(b"bad-image")

    assert built == ["openvino:GPU.1"]  # 只构建过一次,未曾降级重建
    assert stub.unloaded is False       # 未被 _demote 卸载
    assert wrapper._backend is stub     # 活跃后端未变
    assert wrapper._index == 0          # 候选链索引未变
    assert not any("降级" in r.getMessage() for r in caplog.records)  # 无降级 Warn


def test_selecting_backend_no_demote_on_cold_start_first_frame_failure(
        monkeypatch, tmp_path, caplog):
    """冷启动首帧场景:调用前 is_loaded 为 False(TTL 卸载后的窗口),但
    本次 caption() 调用内部先加载成功(模拟 `_load_pipe` 落地)、随后才对
    这张图推理失败(模拟 `_infer` 失败)——真实状态是"加载没问题,只是
    这张图坏",不能因为调用前采样的快照是 False 就误判成"加载失败"进而
    触发降级。判定必须在异常抛出后再读 is_loaded,而不是用调用前的快照。
    """

    class _LoadsThenFailsWithinSameCall:
        """同一次 caption() 调用内:先把内部 loaded 标志置 True(模拟
        `_load_pipe` 成功),再抛 CaptionError(模拟 `_infer` 失败)。"""

        def __init__(self):
            self._loaded = False
            self.unloaded = False

        @property
        def is_loaded(self):
            return self._loaded

        def caption(self, image_bytes):
            self._loaded = True  # 模拟 _load_pipe 在本次调用内已经成功落地
            # 紧接着模拟 _infer 对这张(坏)图推理失败
            raise bs.CaptionError("simulated infer failure right after cold load")

        def unload(self):
            self.unloaded = True
            self._loaded = False

        version = "cold-start-first-frame/backend"

    stub = _LoadsThenFailsWithinSameCall()
    built = []

    def fake_build_backend(spec, **kwargs):
        built.append(spec)
        return stub

    monkeypatch.setattr(bs, "build_backend", fake_build_backend)

    ranked = [
        {"runtime": "openvino", "device": "GPU.1", "tier": 30, "label": "arc"},
        {"runtime": "llamacpp", "device": "cuda", "tier": 30, "label": "other"},
    ]
    wrapper = bs.SelectingCaptionBackend(
        ranked,
        model_path=tmp_path,
        gguf_path=tmp_path / "m.gguf",
        mmproj_path=tmp_path / "mm.gguf",
        idle_ttl_s=60,
    )

    assert wrapper.is_loaded is False  # 调用前尚未加载——冷启动首帧的前提条件

    with caplog.at_level(logging.WARNING, logger="parser.backendselect"):
        with pytest.raises(bs.CaptionError):
            wrapper.caption(b"bad-image-right-after-cold-load")

    assert built == ["openvino:GPU.1"]  # 只构建过一次,未曾降级重建
    assert stub.unloaded is False       # 未被 _demote 卸载
    assert wrapper._backend is stub     # 活跃后端未变
    assert wrapper._index == 0          # 候选链索引未变
    assert not any("降级" in r.getMessage() for r in caplog.records)  # 无降级 Warn
