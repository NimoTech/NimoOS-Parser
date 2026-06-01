from parser import hardware
from parser.hardware import detect_profile, resolve_device, Profile


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


def test_has_nvidia_gpu_false_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr("parser.hardware.shutil.which", lambda _: None)
    # torch check must not even be consulted when nvidia-smi is absent
    monkeypatch.setattr("parser.hardware._torch_cuda_available",
                        lambda: (_ for _ in ()).throw(AssertionError("should not be called")))
    assert hardware._has_nvidia_gpu() is False


def test_has_nvidia_gpu_false_when_smi_present_but_torch_cannot_use_cuda(monkeypatch):
    # nvidia-smi present and `-L` succeeds, but torch can't use a GPU
    # (driver/utils installed on a CPU-only box) → must NOT report a GPU.
    monkeypatch.setattr("parser.hardware.shutil.which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr("parser.hardware.subprocess.run", lambda *a, **k: None)
    monkeypatch.setattr("parser.hardware._torch_cuda_available", lambda: False)
    assert hardware._has_nvidia_gpu() is False


def test_has_nvidia_gpu_true_when_smi_and_torch_agree(monkeypatch):
    monkeypatch.setattr("parser.hardware.shutil.which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr("parser.hardware.subprocess.run", lambda *a, **k: None)
    monkeypatch.setattr("parser.hardware._torch_cuda_available", lambda: True)
    assert hardware._has_nvidia_gpu() is True


def test_resolve_device_auto_picks_cpu_when_no_usable_gpu(monkeypatch):
    monkeypatch.setattr("parser.hardware._has_nvidia_gpu", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_resolve_device_cuda_honored_even_without_gpu(monkeypatch):
    # explicit 'cuda' is returned as-is (clean error at load, not silent downgrade)
    monkeypatch.setattr("parser.hardware._has_nvidia_gpu", lambda: False)
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("cpu") == "cpu"
