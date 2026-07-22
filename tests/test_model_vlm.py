import threading
import time

from parser.model_vlm import OpenVINOCaptionBackend, PROMPT_V1


class _FakePipe:
    """替身推理管线:记录调用并发,可注入延迟。"""

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
    b._load_pipe = lambda: fake          # 注入替身,跳过 openvino_genai
    b._decode_image = lambda data: data  # 跳过 PIL 解码
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
    assert fake.max_concurrent == 1, "推理必须被锁串行化"


def test_idle_ttl_unload(tmp_path):
    b, _ = _backend(tmp_path, ttl=1)
    b.caption(b"x")
    assert b.is_loaded
    b._sweep(now=time.monotonic() + 2)   # 模拟 2 秒后清扫
    assert not b.is_loaded


def test_version_and_prompt(tmp_path):
    b, _ = _backend(tmp_path)
    assert "qwen3-vl-4b-int4" in b.version and "prompt-v1" in b.version
    assert "English" in PROMPT_V1 or "sentences" in PROMPT_V1
