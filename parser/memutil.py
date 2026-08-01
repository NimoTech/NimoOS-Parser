"""glibc allocator helper: return idle heap pages to the kernel after model unload.

After Python/torch/llama.cpp free GB-scale objects, glibc often keeps the pages
in the process heap (RSS doesn't drop, looks like a leak). malloc_trim(0)
proactively releases idle pages. Silently skips on non-glibc environments (musl)
or on failure - this is a best-effort optimization; callers must not depend on
it succeeding.
"""
import ctypes
import logging

log = logging.getLogger("parser.memutil")


def trim_malloc() -> bool:
    """Call glibc malloc_trim(0); returns whether it actually ran. Failures only log at debug."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
        return True
    except Exception as exc:  # noqa: BLE001 - no failure here may ever affect the caller
        log.debug("malloc_trim unavailable: %s", exc)
        return False
