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


def _has_nvidia_gpu() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    try:
        subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True, timeout=2, check=True,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
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
