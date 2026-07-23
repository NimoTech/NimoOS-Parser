"""LlamaCppCaptionBackend 单测 —— GGUF + mmproj 多模态推理后端。

与 test_model_vlm_openvino 系列同款风格:用 fake 注入替身跳过真实
llama_cpp 依赖,只验证 `_BaseCaptionBackend` 骨架 + 本后端特有的
version 拼接 / _infer 组包逻辑。
"""
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
