"""Unit tests for LlamaCppCaptionBackend - the GGUF + mmproj multimodal inference backend.

Same style as the test_model_vlm_openvino series: uses fake injected doubles
to skip the real llama_cpp dependency, and only verifies the
`_BaseCaptionBackend` skeleton plus this backend's own version-string
composition / _infer payload-building logic.

Note: the tests below all bypass `_load_pipe`'s real handler-construction
logic by overriding `b._load_pipe`, so a previous typo'd
`Qwen25VLChatHandler` (a class name that doesn't exist) went undetected by
any test case until it surfaced during a local smoke test.
`test_load_pipe_uses_mtmd_handler` instead monkeypatches
`sys.modules["llama_cpp"]` to inject a fake module, letting `_load_pipe` run
its real implementation, and asserts on the handler class name and
arguments it constructs.
"""
import sys
import threading
import time

import pytest

from parser.model_vlm import CaptionError, PROMPT_V1
from parser.model_vlm_llamacpp import LlamaCppCaptionBackend


class _FakeLlama:
    """A stand-in Llama instance: records call count and peak concurrency, return content is configurable."""

    def __init__(self, content="A dog on grass."):
        self.calls = 0
        self.concurrent = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()
        self._content = content
        self.last_messages = None
        self.last_max_tokens = None

    def create_chat_completion(self, messages, max_tokens=None):
        with self._lock:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        time.sleep(0.05)
        with self._lock:
            self.concurrent -= 1
            self.calls += 1
        self.last_messages = messages
        self.last_max_tokens = max_tokens
        return {"choices": [{"message": {"content": self._content}}]}


def _backend(tmp_path, content="A dog on grass.", **kwargs):
    gguf = tmp_path / "m.gguf"
    mmproj = tmp_path / "mm.gguf"
    gguf.write_bytes(b"fake-gguf")
    mmproj.write_bytes(b"fake-mmproj")
    b = LlamaCppCaptionBackend(gguf_path=gguf, mmproj_path=mmproj,
                                backend_tag="cpu", **kwargs)
    fake = _FakeLlama(content=content)
    b._load_pipe = lambda: fake  # skip the real llama_cpp load
    return b, fake


def test_caption_and_version(tmp_path):
    b, fake = _backend(tmp_path)
    assert "gguf" in b.version and "cpu" in b.version
    assert b.caption(b"\xff\xd8fake") == "A dog on grass."
    assert fake.calls == 1 and b.is_loaded
    # _infer must place PROMPT_V1 into the text part of the user message
    content_parts = fake.last_messages[0]["content"]
    texts = [p["text"] for p in content_parts if p.get("type") == "text"]
    assert PROMPT_V1 in texts
    image_parts = [p for p in content_parts if p.get("type") == "image_url"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"].startswith("data:image")


def test_single_concurrency(tmp_path):
    # Same concurrency-lock test as the OpenVINO backend (4 threads, max_concurrent==1)
    b, fake = _backend(tmp_path)
    threads = [threading.Thread(target=b.caption, args=(b"x",)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert fake.calls == 4
    assert fake.max_concurrent == 1, "inference must be serialized by the lock"


def test_load_failure_wrapped(tmp_path):
    b, _ = _backend(tmp_path)
    b._load_pipe = lambda: (_ for _ in ()).throw(RuntimeError("gguf broken"))
    with pytest.raises(CaptionError):
        b.caption(b"x")
    assert not b.is_loaded


def test_empty_output_raises(tmp_path):
    # create_chat_completion returning empty content -> CaptionError
    b, fake = _backend(tmp_path, content="   ")
    with pytest.raises(CaptionError):
        b.caption(b"x")


def test_missing_gguf_raises_caption_error(tmp_path):
    # When the gguf/mmproj files don't exist, _load_pipe should proactively
    # raise CaptionError, rather than letting the raw exception from the
    # underlying import/construction propagate through.
    b = LlamaCppCaptionBackend(gguf_path=tmp_path / "missing.gguf",
                                mmproj_path=tmp_path / "missing-mm.gguf",
                                backend_tag="cpu")
    with pytest.raises(CaptionError):
        b.caption(b"x")
    assert not b.is_loaded


def test_gpu_layers_and_backend_tag_in_version(tmp_path):
    gguf = tmp_path / "m.gguf"
    mmproj = tmp_path / "mm.gguf"
    gguf.write_bytes(b"fake-gguf")
    mmproj.write_bytes(b"fake-mmproj")
    b = LlamaCppCaptionBackend(gguf_path=gguf, mmproj_path=mmproj,
                                n_gpu_layers=32, backend_tag="rocm")
    assert b.n_gpu_layers == 32
    assert "rocm" in b.version and "gguf" in b.version


class _FakeMTMDChatHandler:
    """A fake handler that records its construction args - asserting on the class name/args is the crux of this test case."""

    instances = []

    def __init__(self, clip_model_path):
        self.clip_model_path = clip_model_path
        _FakeMTMDChatHandler.instances.append(self)


class _FakeQwen25VLChatHandler:
    """Placeholder: exists only to prove the real code doesn't construct a handler of this (deprecated) class name."""

    instances = []

    def __init__(self, clip_model_path):
        self.clip_model_path = clip_model_path
        _FakeQwen25VLChatHandler.instances.append(self)


class _FakeChatFormatModule:
    MTMDChatHandler = _FakeMTMDChatHandler
    Qwen25VLChatHandler = _FakeQwen25VLChatHandler


class _FakeLlamaCtor:
    """A fake class that records llama_cpp.Llama(...) construction calls."""

    instances = []

    def __init__(self, model_path, chat_handler, n_gpu_layers, n_ctx, verbose):
        self.model_path = model_path
        self.chat_handler = chat_handler
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.verbose = verbose
        _FakeLlamaCtor.instances.append(self)


class _FakeLlamaCppModule:
    llama_chat_format = _FakeChatFormatModule
    Llama = _FakeLlamaCtor


def test_load_pipe_uses_mtmd_handler(tmp_path, monkeypatch):
    """Regression test: _load_pipe must construct MTMDChatHandler (not the
    deprecated/nonexistent Qwen25VLChatHandler), and must pass the mmproj
    path through unchanged to clip_model_path.

    Injects a fake module by monkeypatching sys.modules["llama_cpp"], so
    LlamaCppCaptionBackend._load_pipe runs its real implementation (instead
    of overriding _load_pipe) - only this way would a typo'd class name be
    caught by the test.
    """
    _FakeMTMDChatHandler.instances.clear()
    _FakeQwen25VLChatHandler.instances.clear()
    _FakeLlamaCtor.instances.clear()
    monkeypatch.setitem(sys.modules, "llama_cpp", _FakeLlamaCppModule())

    gguf = tmp_path / "m.gguf"
    mmproj = tmp_path / "mm.gguf"
    gguf.write_bytes(b"fake-gguf")
    mmproj.write_bytes(b"fake-mmproj")
    b = LlamaCppCaptionBackend(gguf_path=gguf, mmproj_path=mmproj,
                                n_gpu_layers=16, backend_tag="rocm")

    pipe = b._load_pipe()

    # Must go through MTMDChatHandler, exactly once, never constructing the deprecated class name.
    assert len(_FakeMTMDChatHandler.instances) == 1
    assert not _FakeQwen25VLChatHandler.instances
    handler = _FakeMTMDChatHandler.instances[0]
    assert handler.clip_model_path == str(mmproj)

    # Llama(...) receives exactly this handler instance, with path/gpu-layer count passed through correctly.
    assert len(_FakeLlamaCtor.instances) == 1
    llama = _FakeLlamaCtor.instances[0]
    assert llama.chat_handler is handler
    assert llama.model_path == str(gguf)
    assert llama.n_gpu_layers == 16
    assert pipe is llama


class _FakeExitStack:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakeMTMDHandlerWithStack:
    """A fake with _exit_stack (matching llama-cpp-python 0.3.34's real structure)."""

    instances = []

    def __init__(self, clip_model_path):
        self.clip_model_path = clip_model_path
        self._exit_stack = _FakeExitStack()
        type(self).instances.append(self)


class _FakeMTMDHandlerNoStack:
    """A fake without _exit_stack (simulating a future llama-cpp-python private-interface change)."""

    def __init__(self, clip_model_path):
        self.clip_model_path = clip_model_path


class _FakeClosableLlama:
    instances = []

    def __init__(self, model_path, chat_handler, n_gpu_layers, n_ctx, verbose):
        self.chat_handler = chat_handler
        self.close_calls = 0
        type(self).instances.append(self)

    def create_chat_completion(self, messages, max_tokens=None):
        return {"choices": [{"message": {"content": "a dog"}}]}

    def close(self):
        self.close_calls += 1


def _fake_module(handler_cls):
    class _ChatFormat:
        MTMDChatHandler = handler_cls

    class _Module:
        llama_chat_format = _ChatFormat
        Llama = _FakeClosableLlama

    return _Module()


def _real_backend(tmp_path, monkeypatch, handler_cls):
    _FakeClosableLlama.instances.clear()
    monkeypatch.setitem(sys.modules, "llama_cpp", _fake_module(handler_cls))
    gguf = tmp_path / "m.gguf"
    mmproj = tmp_path / "mm.gguf"
    gguf.write_bytes(b"fake-gguf")
    mmproj.write_bytes(b"fake-mmproj")
    return LlamaCppCaptionBackend(gguf_path=gguf, mmproj_path=mmproj,
                                  backend_tag="cpu")


def test_unload_frees_mtmd_context(tmp_path, monkeypatch):
    """Regression for the main root cause of the 2026-07-28 OOM: unload must
    close the handler's _exit_stack (triggering mtmd_free to release
    ~836MB of mmproj), and must drop the handler reference."""
    _FakeMTMDHandlerWithStack.instances.clear()
    b = _real_backend(tmp_path, monkeypatch, _FakeMTMDHandlerWithStack)
    assert b.caption(b"\xff\xd8fake") == "a dog"
    handler = _FakeMTMDHandlerWithStack.instances[0]
    llama = _FakeClosableLlama.instances[0]
    assert b._chat_handler is handler

    b.unload()

    assert llama.close_calls == 1, "Llama.close() must be called (releases the language model)"
    assert handler._exit_stack.closed, "_exit_stack.close() must be called (mtmd_free)"
    assert b._chat_handler is None
    assert not b.is_loaded
    assert b._unload_disabled is False


def test_missing_exit_stack_disables_idle_unload(tmp_path, monkeypatch):
    """When llama-cpp-python's private interface changes: disable idle
    auto-unload (staying resident is safer than leaking every cycle);
    explicit unload still works and doesn't raise."""
    b = _real_backend(tmp_path, monkeypatch, _FakeMTMDHandlerNoStack)
    b.caption(b"x")
    assert b._unload_disabled is True
    b._sweep(now=time.monotonic() + 10 ** 6)
    assert b.is_loaded, "idle sweep must not unload when the interface is missing"
    b.unload()
    assert not b.is_loaded
