from parser.hardware import detect_profile, Profile


def test_detect_profile_returns_lean_on_low_ram(monkeypatch):
    monkeypatch.setattr("parser.hardware._total_ram_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr("parser.hardware._has_nvidia_gpu", lambda: False)
    assert detect_profile() == Profile.LEAN


def test_detect_profile_returns_balanced_on_mid(monkeypatch):
    monkeypatch.setattr("parser.hardware._total_ram_bytes", lambda: 32 * 1024**3)
    monkeypatch.setattr("parser.hardware._has_nvidia_gpu", lambda: False)
    assert detect_profile() == Profile.BALANCED


def test_detect_profile_returns_gpu_when_gpu(monkeypatch):
    monkeypatch.setattr("parser.hardware._total_ram_bytes", lambda: 64 * 1024**3)
    monkeypatch.setattr("parser.hardware._has_nvidia_gpu", lambda: True)
    assert detect_profile() == Profile.GPU


def test_override_via_env(monkeypatch):
    monkeypatch.setenv("PARSER_PROFILE", "lean")
    monkeypatch.setattr("parser.hardware._total_ram_bytes", lambda: 128 * 1024**3)
    monkeypatch.setattr("parser.hardware._has_nvidia_gpu", lambda: True)
    assert detect_profile() == Profile.LEAN
