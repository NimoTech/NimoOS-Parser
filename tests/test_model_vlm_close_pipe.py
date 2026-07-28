"""_close_pipe 钩子与 _unload_disabled(2026-07-28 mmproj 泄漏 OOM 回归防线)。

卸载必须走 _close_pipe 显式释放原生资源;当后端声明无法安全释放时
(_unload_disabled),闲置清扫不得卸载(常驻比每周期泄漏安全),
但显式 unload()(降级/运维)仍然生效。
"""
import time

from parser.model_vlm import _BaseCaptionBackend


class _RecordingBackend(_BaseCaptionBackend):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.closed = []

    def _load_pipe(self):
        return object()

    def _infer(self, pipe, image_bytes):
        return "a cat"

    def _close_pipe(self, pipe):
        self.closed.append(pipe)


def test_unload_calls_close_pipe_before_dropping_ref():
    b = _RecordingBackend()
    b.caption(b"x")
    pipe = b._pipe
    b.unload()
    assert b.closed == [pipe]
    assert not b.is_loaded


def test_sweep_respects_unload_disabled_but_explicit_unload_works():
    b = _RecordingBackend(idle_ttl_s=1)
    b.caption(b"x")
    b._unload_disabled = True
    b._sweep(now=time.monotonic() + 999)
    assert b.is_loaded, "禁用后闲置清扫不得卸载"
    b.unload()          # 显式卸载(降级/运维)不受禁用影响
    assert not b.is_loaded


def test_close_pipe_failure_does_not_block_unload():
    b = _RecordingBackend()
    b.caption(b"x")

    def boom(pipe):
        raise RuntimeError("close failed")

    b._close_pipe = boom
    b.unload()
    assert not b.is_loaded
