import time

import pytest

from parser.workers import WorkerPool


def _pool(ratio, concurrency):
    return WorkerPool(None, text_pipeline=None, concurrency=concurrency,
                      load_ratio_fn=lambda: ratio)


def test_pacing_delay_follows_tier_and_load():
    assert _pool(0.0, 4)._pacing_delay() == 0.0
    assert _pool(0.0, 2)._pacing_delay() == 1.0
    # 4 ** ((1.0-0.7)/0.3) is 4.000000000000001 in IEEE float (same fp noise
    # tests/test_pacing.py already works around with approx for this ratio).
    assert _pool(1.0, 2)._pacing_delay() == pytest.approx(4.0)
    assert _pool(9.9, 1)._pacing_delay() == 60.0


def test_pacing_follows_live_concurrency_changes():
    p = _pool(0.0, 2)
    p.concurrency = 1          # set_concurrency 动态改档后,pacing 立即跟随
    assert p._pacing_delay() == 5.0


def test_throughput_window():
    p = _pool(0.0, 2)
    now = time.time()
    p._done_ts.extend([now - 700, now - 500, now - 10, now - 1])  # 700s 的应被剪掉
    tp = p.throughput()
    assert tp["done_last_10m"] == 3
    assert tp["rate_per_min"] == round(3 / 10.0, 2)


def test_throughput_empty():
    p = _pool(0.0, 2)
    assert p.throughput() == {"done_last_10m": 0, "rate_per_min": 0.0}
