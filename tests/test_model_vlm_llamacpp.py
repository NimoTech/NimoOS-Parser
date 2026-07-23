"""LlamaCppCaptionBackend 单测 —— GGUF + mmproj 多模态推理后端。

与 test_model_vlm_openvino 系列同款风格:用 fake 注入替身跳过真实
llama_cpp 依赖,只验证 `_BaseCaptionBackend` 骨架 + 本后端特有的
version 拼接 / _infer 组包逻辑。

注意:以下测试全部通过重写 `b._load_pipe` 绕开了 `_load_pipe` 内部真实
的 handler 构造逻辑,因此此前 `Qwen25VLChatHandler`(不存在的类名)写错
也不会被任何用例发现,直到本机冒烟才暴露。`test_load_pipe_uses_mtmd_handler`
改为 monkeypatch `sys.modules["llama_cpp"]` 注入 fake 模块,让 `_load_pipe`
走真实实现,断言其构造的 handler 类名与参数。
"""
import sys
import threading
import time

import pytest

from parser.model_vlm import CaptionError, PROMPT_V1
from parser.model_vlm_llamacpp import LlamaCppCaptionBackend


class _FakeLlama:
    """替身 Llama 实例:记录调用次数与并发峰值,可配置返回内容。"""

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
    b._load_pipe = lambda: fake  # 跳过真实 llama_cpp 加载
    return b, fake


def test_caption_and_version(tmp_path):
    b, fake = _backend(tmp_path)
    assert "gguf" in b.version and "cpu" in b.version
    assert b.caption(b"\xff\xd8fake") == "A dog on grass."
    assert fake.calls == 1 and b.is_loaded
    # _infer 必须把 PROMPT_V1 塞进 user message 的文本部分
    content_parts = fake.last_messages[0]["content"]
    texts = [p["text"] for p in content_parts if p.get("type") == "text"]
    assert PROMPT_V1 in texts
    image_parts = [p for p in content_parts if p.get("type") == "image_url"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"].startswith("data:image")


def test_single_concurrency(tmp_path):
    # 与 OpenVINO 后端同款并发锁测试(4 线程,max_concurrent==1)
    b, fake = _backend(tmp_path)
    threads = [threading.Thread(target=b.caption, args=(b"x",)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert fake.calls == 4
    assert fake.max_concurrent == 1, "推理必须被锁串行化"


def test_load_failure_wrapped(tmp_path):
    b, _ = _backend(tmp_path)
    b._load_pipe = lambda: (_ for _ in ()).throw(RuntimeError("gguf broken"))
    with pytest.raises(CaptionError):
        b.caption(b"x")
    assert not b.is_loaded


def test_empty_output_raises(tmp_path):
    # create_chat_completion 返回空 content → CaptionError
    b, fake = _backend(tmp_path, content="   ")
    with pytest.raises(CaptionError):
        b.caption(b"x")


def test_missing_gguf_raises_caption_error(tmp_path):
    # gguf/mmproj 文件不存在时 _load_pipe 应主动抛 CaptionError,而非
    # 让底层 import/构造抛出的原始异常打穿。
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
    """记录构造参数的 fake handler —— 断言类名/参数是本用例的核心。"""

    instances = []

    def __init__(self, clip_model_path):
        self.clip_model_path = clip_model_path
        _FakeMTMDChatHandler.instances.append(self)


class _FakeQwen25VLChatHandler:
    """占位:仅用于证明真实代码没有构造这个(已废弃)类名的 handler。"""

    instances = []

    def __init__(self, clip_model_path):
        self.clip_model_path = clip_model_path
        _FakeQwen25VLChatHandler.instances.append(self)


class _FakeChatFormatModule:
    MTMDChatHandler = _FakeMTMDChatHandler
    Qwen25VLChatHandler = _FakeQwen25VLChatHandler


class _FakeLlamaCtor:
    """记录 llama_cpp.Llama(...) 构造调用的 fake 类。"""

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
    """回归测试:_load_pipe 必须构造 MTMDChatHandler(不是已废弃/不存在
    的 Qwen25VLChatHandler),且把 mmproj 路径原样传给 clip_model_path。

    通过 monkeypatch sys.modules["llama_cpp"] 注入 fake 模块,让
    LlamaCppCaptionBackend._load_pipe 走真实实现(不重写 _load_pipe),
    这样类名写错才会被测试捕捉到。
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

    # 必须走 MTMDChatHandler,一次且仅一次,不能构造已废弃的类名。
    assert len(_FakeMTMDChatHandler.instances) == 1
    assert not _FakeQwen25VLChatHandler.instances
    handler = _FakeMTMDChatHandler.instances[0]
    assert handler.clip_model_path == str(mmproj)

    # Llama(...) 拿到的正是上面这个 handler 实例,且路径/gpu 层数透传正确。
    assert len(_FakeLlamaCtor.instances) == 1
    llama = _FakeLlamaCtor.instances[0]
    assert llama.chat_handler is handler
    assert llama.model_path == str(gguf)
    assert llama.n_gpu_layers == 16
    assert pipe is llama
