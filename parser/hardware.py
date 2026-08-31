import functools
import os
import shutil
import subprocess
from enum import Enum


class Profile(str, Enum):
    LEAN = "lean"
    BALANCED = "balanced"
    GPU = "gpu"


def _total_ram_bytes() -> int:
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb * 1024
    except OSError:
        pass
    return 0


def _torch_cuda_available() -> bool:
    """Authoritative check for whether torch can actually use a CUDA device.

    nvidia-smi merely tells us the driver/utils are installed — it returns
    success on hosts that have nvidia-utils but no usable GPU (e.g. an AMD
    box). torch.cuda.is_available() is what actually gates model placement.
    """
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _has_nvidia_gpu() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    try:
        subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True, timeout=2, check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    # nvidia-smi present is necessary but NOT sufficient — confirm torch can
    # really use the GPU, otherwise 'auto' falsely resolves to cuda on
    # CPU-only machines that happen to have nvidia-utils installed and crashes
    # at model load.
    return _torch_cuda_available()


def _openvino_core_factory():
    # Indirection point so tests can monkeypatch Core construction.
    import openvino
    return openvino.Core()


@functools.lru_cache(maxsize=1)
def _has_openvino_gpu() -> bool:
    """Best-effort probe: is an OpenVINO GPU device visible?

    Any failure (openvino missing, driver broken) returns False — a failed
    probe must never take down resolution (same philosophy as backendselect).
    Cached: device presence does not change within a process lifetime.
    """
    try:
        return "GPU" in _openvino_core_factory().available_devices
    except Exception:
        return False


def detect_profile() -> Profile:
    override = os.environ.get("PARSER_PROFILE", "").lower()
    if override in (p.value for p in Profile):
        return Profile(override)
    if _has_nvidia_gpu():
        return Profile.GPU
    if _total_ram_bytes() >= 16 * 1024**3:
        return Profile.BALANCED
    return Profile.LEAN


def resolve_device(device_pref: str) -> str:
    """Resolve a user device preference (auto|cuda|gpu|cpu) to a device string.

    'auto' picks 'cuda' if an NVIDIA GPU is present, else 'gpu' if an
    OpenVINO GPU (Intel iGPU/dGPU) is visible, else 'cpu'. Explicit values
    are returned as-is ('cuda'/'gpu' are honoured even when no device is
    detected so the model load surface raises/falls back explicitly instead
    of silently downgrading here). 'gpu' means OpenVINO GPU; torch paths
    never receive it (see parser/text_backend.py).
    """
    pref = (device_pref or "auto").lower()
    if pref == "auto":
        if _has_nvidia_gpu():
            return "cuda"
        if _has_openvino_gpu():
            return "gpu"
        return "cpu"
    if pref in ("cuda", "gpu", "cpu"):
        return pref
    raise ValueError(f"unknown device preference: {device_pref!r}")
