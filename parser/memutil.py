"""glibc 分配器工具:模型 unload 后把空闲堆页还给内核。

Python/torch/llama.cpp 释放 GB 级对象后,glibc 常把页留在进程堆里
(RSS 不回落,看起来像泄漏)。malloc_trim(0) 主动归还空闲页。
非 glibc 环境(musl)或调用失败时静默跳过——这是尽力而为的优化,
调用方不得依赖其成功。
"""
import ctypes
import logging

log = logging.getLogger("parser.memutil")


def trim_malloc() -> bool:
    """调用 glibc malloc_trim(0);返回是否实际执行。失败只记 debug 日志。"""
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
        return True
    except Exception as exc:  # noqa: BLE001 — 任何失败都不许影响调用方
        log.debug("malloc_trim unavailable: %s", exc)
        return False
