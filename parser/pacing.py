"""Adaptive job pacing (spec §4.7). Pure functions, no asyncio.

load_ratio deliberately uses /proc/loadavg (1-min) rather than CPU
utilisation: loadavg counts D-state (uninterruptible IO wait) processes,
so a Qdrant write bottleneck raises it even with idle CPUs — one signal
backpressures both CPU and disk IO.
"""
import os

_LOADAVG_PATH = "/proc/loadavg"
_KNEE = 0.7
# concurrency value doubles as the pacing tier (zero API change):
#   4 = full speed, 2 = balanced (default), 1 = eco
_BASE = {4: 0.0, 2: 1.0, 1: 5.0}
_CAP = {4: 0.0, 2: 30.0, 1: 60.0}


def load_ratio() -> float:
    """1-min loadavg divided by CPU count; 0.0 on any failure."""
    try:
        with open(_LOADAVG_PATH) as f:
            la1 = float(f.read().split()[0])
        return la1 / max(1, os.cpu_count() or 1)
    except (OSError, ValueError, IndexError):
        return 0.0


def sleep_seconds(mode: int, ratio: float) -> float:
    """Seconds to pause between jobs for the given tier and load ratio.

    ratio <= 0.7 -> base; above the knee the pause quadruples per +0.3
    of ratio, capped per tier: sleep = min(cap, base * 4**((ratio-0.7)/0.3)).
    """
    base = _BASE.get(mode, _BASE[2])
    cap = _CAP.get(mode, _CAP[2])
    if base <= 0:
        return 0.0
    if ratio <= _KNEE:
        return base
    return min(cap, base * 4 ** ((ratio - _KNEE) / 0.3))
