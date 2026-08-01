import pytest

from parser import pacing


@pytest.mark.parametrize("mode,ratio,expected", [
    # max-performance tier: always 0
    (4, 0.0, 0.0), (4, 1.0, 0.0), (4, 5.0, 0.0),
    # balanced tier: below knee = base
    (2, 0.0, 1.0), (2, 0.7, 1.0),
    # balanced tier: scales as 4^((ratio-0.7)/0.3)
    (2, 1.0, 4.0),          # 4^1
    (2, 1.3, 16.0),         # 4^2
    (2, 2.0, 30.0),         # over cap -> 30
    # power-saving tier
    (1, 0.5, 5.0),
    (1, 1.0, 20.0),         # 5 * 4^1
    (1, 3.0, 60.0),         # cap
    # unknown tier treated as balanced
    (3, 0.0, 1.0),
])
def test_sleep_seconds_table(mode, ratio, expected):
    assert pacing.sleep_seconds(mode, ratio) == pytest.approx(expected, rel=1e-6)


def test_load_ratio_reads_proc(monkeypatch, tmp_path):
    f = tmp_path / "loadavg"
    f.write_text("3.50 2.00 1.00 2/345 6789\n")
    monkeypatch.setattr(pacing, "_LOADAVG_PATH", str(f))
    monkeypatch.setattr(pacing.os, "cpu_count", lambda: 7)
    assert pacing.load_ratio() == pytest.approx(0.5)


def test_load_ratio_failure_returns_zero(monkeypatch):
    monkeypatch.setattr(pacing, "_LOADAVG_PATH", "/nonexistent/loadavg")
    assert pacing.load_ratio() == 0.0
