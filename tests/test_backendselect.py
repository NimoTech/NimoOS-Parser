"""Tests for backendselect.py's hardware probing/scoring/factory/fallback wrapper.

All three probe functions (_probe_openvino/_probe_nvidia/_probe_amd) are
monkeypatched with doubles, so this doesn't depend on real hardware/drivers;
build_backend/select_caption_backend only construct backend instances
(lazy-loaded, openvino/llama_cpp aren't imported at construction time), so
this still passes even without these two optional deps installed locally.
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
    # Intel Arc dGPU (30) and NVIDIA dGPU (30) both present -> same tier, openvino wins
    monkeypatch.setattr(bs, "_probe_openvino",
        lambda: [{"runtime": "openvino", "device": "GPU.0", "tier": 30, "label": "Arc B60"},
                 {"runtime": "openvino", "device": "CPU", "tier": 10, "label": "CPU"}])
    monkeypatch.setattr(bs, "_probe_nvidia",
        lambda: [{"runtime": "llamacpp", "device": "cuda", "tier": 30, "label": "RTX 4090"}])
    monkeypatch.setattr(bs, "_probe_amd", lambda: [])
    ranked = bs.rank(bs.probe_hardware())
    assert (ranked[0]["runtime"], ranked[0]["device"]) == ("openvino", "GPU.0")


def test_amd_only_picks_vulkan(monkeypatch):
    # openvino only detects CPU (10), AMD detects gfx -> llamacpp:vulkan (iGPU 20) wins
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
# select_caption_backend (explicit spec / fallback on invalid value)
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
    assert len(caplog.records) >= 1  # an invalid value must warn


# ---------------------------------------------------------------------------
# SelectingCaptionBackend fallback wrapper
# ---------------------------------------------------------------------------

def test_selecting_backend_fallback(monkeypatch, tmp_path, caplog):
    """Preferred candidate raises CaptionError on load -> demotes to the next candidate and retries successfully."""

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
    assert len(caplog.records) >= 1  # demotion must warn
    assert wrapper.version == "good/backend"
    assert wrapper.is_loaded is True
    assert bad_instances[0].unloaded is True  # _demote must unload the old backend before rebuilding


def test_selecting_backend_no_demote_on_inference_failure(monkeypatch, tmp_path, caplog):
    """A single inference failure (bad image) on an already-loaded backend should not trigger demotion; the exception is re-raised as-is."""

    class _LoadsThenBadImage:
        """Simulates is_loaded going False->True: the first call loads
        successfully and returns a result, then (once loaded) inference
        fails and raises CaptionError."""

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

    # There's still a next candidate in the chain available for demotion - if it wrongly demotes, the test will catch it.
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

    # First call: loads successfully and returns a result.
    out = wrapper.caption(b"good-image")
    assert out == "first good caption"
    assert stub.is_loaded is True

    # Second call: backend is already loaded, this time a single bad image fails inference - should not demote.
    with caplog.at_level(logging.WARNING, logger="parser.backendselect"):
        with pytest.raises(bs.CaptionError):
            wrapper.caption(b"bad-image")

    assert built == ["openvino:GPU.1"]  # only built once, never demoted/rebuilt
    assert stub.unloaded is False       # not unloaded by _demote
    assert wrapper._backend is stub     # active backend unchanged
    assert wrapper._index == 0          # candidate chain index unchanged
    assert not any("demoted" in r.getMessage() for r in caplog.records)  # no demotion warning


def test_selecting_backend_no_demote_on_cold_start_first_frame_failure(
        monkeypatch, tmp_path, caplog):
    """Cold-start first-frame scenario: is_loaded is False before the call
    (the window right after a TTL unload), but inside this caption() call it
    first loads successfully (simulating `_load_pipe` landing), and only
    afterward fails inference on this image (simulating `_infer` failing) -
    the real state is "loading was fine, just this image is bad"; it must
    not be misjudged as a "load failure" (and thus demoted) just because the
    snapshot taken before the call was False. The check must read is_loaded
    after the exception is raised, not from the pre-call snapshot.
    """

    class _LoadsThenFailsWithinSameCall:
        """Within a single caption() call: first flips the internal loaded
        flag to True (simulating `_load_pipe` succeeding), then raises
        CaptionError (simulating `_infer` failing)."""

        def __init__(self):
            self._loaded = False
            self.unloaded = False

        @property
        def is_loaded(self):
            return self._loaded

        def caption(self, image_bytes):
            self._loaded = True  # simulates _load_pipe having already succeeded within this call
            # then simulates _infer failing inference on this (bad) image
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

    assert wrapper.is_loaded is False  # not yet loaded before the call - the precondition for a cold-start first frame

    with caplog.at_level(logging.WARNING, logger="parser.backendselect"):
        with pytest.raises(bs.CaptionError):
            wrapper.caption(b"bad-image-right-after-cold-load")

    assert built == ["openvino:GPU.1"]  # only built once, never demoted/rebuilt
    assert stub.unloaded is False       # not unloaded by _demote
    assert wrapper._backend is stub     # active backend unchanged
    assert wrapper._index == 0          # candidate chain index unchanged
    assert not any("demoted" in r.getMessage() for r in caplog.records)  # no demotion warning


# ---------------------------------------------------------------------------
# weight-availability-aware selection (a fresh install ships only GGUF weights;
# the IR form only exists after an on-machine conversion)
# ---------------------------------------------------------------------------

def _fake_probes(monkeypatch, openvino=(), nvidia=(), amd=()):
    monkeypatch.setattr(bs, "_probe_openvino", lambda: list(openvino))
    monkeypatch.setattr(bs, "_probe_nvidia", lambda: list(nvidia))
    monkeypatch.setattr(bs, "_probe_amd", lambda: list(amd))


def _gguf_weights(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"g")
    mm = tmp_path / "mmproj.gguf"
    mm.write_bytes(b"m")
    return gguf, mm


def test_gguf_only_intel_machine_gets_llamacpp_cpu(monkeypatch, tmp_path):
    # An Intel machine probes to a pure-openvino chain — before the weight
    # check, every link needed IR weights and captioning was dead on a
    # GGUF-only disk (the whole point of this fix).
    _fake_probes(monkeypatch, openvino=[
        {"runtime": "openvino", "device": "GPU.0", "tier": 20, "label": "iGPU"},
        {"runtime": "openvino", "device": "CPU", "tier": 10, "label": "CPU"}])
    gguf, mm = _gguf_weights(tmp_path)
    sel = bs.select_caption_backend(
        vlm_device="auto", model_path=tmp_path / "ir-absent",
        gguf_path=gguf, mmproj_path=mm, idle_ttl_s=60)
    assert [bs._candidate_spec(c) for c in sel._ranked] == ["llamacpp:cpu"]
    assert type(sel._backend).__name__ == "LlamaCppCaptionBackend"


def test_gguf_only_nvidia_keeps_cuda_then_cpu_tail(monkeypatch, tmp_path):
    _fake_probes(monkeypatch,
        openvino=[{"runtime": "openvino", "device": "CPU", "tier": 10, "label": "CPU"}],
        nvidia=[{"runtime": "llamacpp", "device": "cuda", "tier": 30, "label": "NVIDIA"}])
    gguf, mm = _gguf_weights(tmp_path)
    sel = bs.select_caption_backend(
        vlm_device="auto", model_path=tmp_path / "ir-absent",
        gguf_path=gguf, mmproj_path=mm, idle_ttl_s=60)
    assert [bs._candidate_spec(c) for c in sel._ranked] == ["llamacpp:cuda", "llamacpp:cpu"]


def test_ir_only_machine_drops_gguf_candidates(monkeypatch, tmp_path):
    _fake_probes(monkeypatch,
        openvino=[{"runtime": "openvino", "device": "CPU", "tier": 10, "label": "CPU"}],
        amd=[{"runtime": "llamacpp", "device": "vulkan", "tier": 30, "label": "AMD"}])
    ir = tmp_path / "ir"
    ir.mkdir()
    (ir / "openvino_model.xml").write_text("x")
    sel = bs.select_caption_backend(
        vlm_device="auto", model_path=ir,
        gguf_path=tmp_path / "no.gguf", mmproj_path=tmp_path / "no-mm.gguf", idle_ttl_s=60)
    assert [bs._candidate_spec(c) for c in sel._ranked] == ["openvino:CPU"]


def test_no_weights_keeps_probe_chain_unchanged(monkeypatch, tmp_path):
    # Neither form on disk: selection must behave exactly as before this fix,
    # so the "no weights installed at all" error surfaces the same way.
    _fake_probes(monkeypatch, openvino=[
        {"runtime": "openvino", "device": "GPU.0", "tier": 20, "label": "iGPU"},
        {"runtime": "openvino", "device": "CPU", "tier": 10, "label": "CPU"}])
    sel = bs.select_caption_backend(
        vlm_device="auto", model_path=tmp_path / "ir-absent",
        gguf_path=tmp_path / "no.gguf", mmproj_path=tmp_path / "no-mm.gguf", idle_ttl_s=60)
    assert [bs._candidate_spec(c) for c in sel._ranked] == ["openvino:GPU.0", "openvino:CPU"]
