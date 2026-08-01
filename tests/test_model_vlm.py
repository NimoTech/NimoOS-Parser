import sys
import threading
import time
import types

import pytest

from parser.model_vlm import CaptionError, OpenVINOCaptionBackend, PROMPT_V1


class _FakePipe:
    """A stand-in inference pipeline: records call concurrency, delay is injectable."""

    def __init__(self):
        self.calls = 0
        self.concurrent = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()

    def generate(self, prompt, images=None, max_new_tokens=None):
        with self._lock:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        time.sleep(0.05)
        with self._lock:
            self.concurrent -= 1
            self.calls += 1
        return "A dog running on the beach."


def _backend(tmp_path, ttl=300):
    b = OpenVINOCaptionBackend(model_path=tmp_path, idle_ttl_s=ttl)
    fake = _FakePipe()
    b._load_pipe = lambda: fake          # inject the double, skip openvino_genai
    b._decode_image = lambda data: data  # skip PIL decoding
    return b, fake


def test_lazy_load_and_caption(tmp_path):
    b, fake = _backend(tmp_path)
    assert not b.is_loaded
    out = b.caption(b"fake-jpeg-bytes")
    assert out == "A dog running on the beach."
    assert b.is_loaded and fake.calls == 1


def test_single_concurrency(tmp_path):
    b, fake = _backend(tmp_path)
    threads = [threading.Thread(target=b.caption, args=(b"x",)) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert fake.calls == 4
    assert fake.max_concurrent == 1, "inference must be serialized by the lock"


def test_idle_ttl_unload(tmp_path):
    b, _ = _backend(tmp_path, ttl=1)
    b.caption(b"x")
    assert b.is_loaded
    b._sweep(now=time.monotonic() + 2)   # simulate a sweep 2 seconds later
    assert not b.is_loaded


def test_version_and_prompt(tmp_path):
    b, _ = _backend(tmp_path)
    assert "qwen3-vl-4b-int4" in b.version and "prompt-v1" in b.version
    assert "English" in PROMPT_V1 or "sentences" in PROMPT_V1


def test_load_failure_wrapped_as_caption_error_and_recoverable(tmp_path):
    b = OpenVINOCaptionBackend(model_path=tmp_path, idle_ttl_s=300)

    def _boom():
        raise RuntimeError("IR corrupted")

    b._load_pipe = _boom
    b._decode_image = lambda data: data

    with pytest.raises(CaptionError):
        b.caption(b"x")
    assert not b.is_loaded  # after a load failure, _pipe stays None so it can be retried

    fake = _FakePipe()
    b._load_pipe = lambda: fake
    out = b.caption(b"x")
    assert out == "A dog running on the beach."
    assert b.is_loaded and fake.calls == 1


def test_device_defaults_to_cpu(tmp_path):
    b = OpenVINOCaptionBackend(model_path=tmp_path)
    assert b.device == "CPU"
    assert "CPU" in b.version


def test_device_param_reflected_in_version(tmp_path):
    b = OpenVINOCaptionBackend(model_path=tmp_path, device="GPU.1")
    assert "GPU.1" in b.version


def test_load_pipe_passes_device_to_vlmpipeline(tmp_path, monkeypatch):
    """_load_pipe must pass self.device through unchanged to openvino_genai.VLMPipeline."""
    captured = {}

    class _FakeVLMPipeline:
        def __init__(self, model_path, device):
            captured["model_path"] = model_path
            captured["device"] = device

    fake_module = types.SimpleNamespace(VLMPipeline=_FakeVLMPipeline)
    monkeypatch.setitem(sys.modules, "openvino_genai", fake_module)

    b = OpenVINOCaptionBackend(model_path=tmp_path, device="GPU.1")
    b._load_pipe()

    assert captured["device"] == "GPU.1"
    assert captured["model_path"] == str(tmp_path)
