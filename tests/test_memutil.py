"""memutil.trim_malloc:尽力而为的 glibc 归还,任何环境下都不许抛异常。"""
import parser.memutil as memutil


def test_trim_malloc_returns_bool():
    # glibc 环境 True;非 glibc(musl)静默 False——两者都合法,但不能抛。
    assert memutil.trim_malloc() in (True, False)


def test_trim_malloc_swallows_errors(monkeypatch):
    monkeypatch.setattr(
        memutil.ctypes, "CDLL",
        lambda *_: (_ for _ in ()).throw(OSError("no libc")))
    assert memutil.trim_malloc() is False
