"""_close_pipe hook and _unload_disabled (regression guard for the 2026-07-28 mmproj leak OOM).

Unload must go through _close_pipe to explicitly release native resources;
when a backend has declared it cannot safely release them (_unload_disabled),
idle sweep must not unload (staying resident is safer than leaking every
cycle), but an explicit unload() (demotion/ops) still takes effect.
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
    assert b.is_loaded, "idle sweep must not unload once disabled"
    b.unload()          # explicit unload (demotion/ops) is unaffected by the disable flag
    assert not b.is_loaded


def test_close_pipe_failure_does_not_block_unload():
    b = _RecordingBackend()
    b.caption(b"x")

    def boom(pipe):
        raise RuntimeError("close failed")

    b._close_pipe = boom
    b.unload()
    assert not b.is_loaded
