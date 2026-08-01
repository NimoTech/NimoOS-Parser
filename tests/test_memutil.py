"""memutil.trim_malloc: best-effort glibc page return, must never raise in any environment."""
import parser.memutil as memutil


def test_trim_malloc_returns_bool():
    # True on glibc environments; silently False on non-glibc (musl) - both are valid, but it must never raise.
    assert memutil.trim_malloc() in (True, False)


def test_trim_malloc_swallows_errors(monkeypatch):
    monkeypatch.setattr(
        memutil.ctypes, "CDLL",
        lambda *_: (_ for _ in ()).throw(OSError("no libc")))
    assert memutil.trim_malloc() is False
