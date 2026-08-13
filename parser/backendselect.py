"""The selection brain for multi-platform adaptive caption VLM backends.

Single-machine deployments have wildly varying hardware (Intel iGPU/dGPU,
NVIDIA, AMD, CPU-only). This module probes the inference paths available on
the local machine at startup, scores and ranks them, picks a preferred
backend, and automatically demotes down the scored order if the preferred
backend fails to load - the final fallback is always openvino:CPU (pure CPU
inference is slow but always works, with no dependency on any vendor driver).

All three probe functions (`_probe_openvino`/`_probe_nvidia`/`_probe_amd`)
are best-effort: the libraries/CLI tools they depend on are absent in many
environments, so any exception is swallowed and an empty list returned -
a failed probe must never take down service startup.

Tier scoring scheme:
- 30 = discrete GPU (dGPU): Intel Arc / NVIDIA discrete
- 20 = integrated GPU (iGPU), or AMD when probe precision can't distinguish
  discrete from integrated (vulkaninfo/rocminfo can only confirm "a gfx
  device is visible", not reliably tell dGPU from iGPU, so it's uniformly
  scored as iGPU)
- 10 = CPU

When tiers are equal, break the tie by runtime priority: openvino >
llamacpp:cuda > llamacpp:vulkan > llamacpp:cpu (OpenVINO is the most deeply
optimized for the Intel ecosystem, so it's preferred among equally-tiered
discrete GPUs).
"""
import logging
import shutil
import subprocess
from pathlib import Path

from parser.model_vlm import CaptionError, OpenVINOCaptionBackend
from parser.model_vlm_llamacpp import LlamaCppCaptionBackend

log = logging.getLogger("parser.backendselect")

# Timeout ceiling for probes/subprocess calls - avoids hanging startup on a misbehaving driver.
_PROBE_TIMEOUT_S = 5

# Fallback candidate: the last-resort tier that must always be reachable (pure CPU inference).
_FALLBACK_CANDIDATE = {"runtime": "openvino", "device": "CPU", "tier": 10, "label": "CPU"}

# The GGUF twin of _FALLBACK_CANDIDATE, used when only GGUF weights are on
# disk: llama.cpp's CPU path runs on any machine, while openvino:CPU still
# needs the IR weights and would just fail to load.
_GGUF_FALLBACK_CANDIDATE = {"runtime": "llamacpp", "device": "cpu", "tier": 10, "label": "CPU (gguf)"}


def _weights_present(model_path: Path, gguf_path: Path, mmproj_path: Path) -> tuple[bool, bool]:
    """Which weight forms are actually on disk: (IR, GGUF).

    The install channel ships only the GGUF weights by default (the IR form is
    produced by converting on an Intel machine, see scripts/vlm/README.md), so
    "hardware says openvino, disk says gguf" is the normal state of a fresh
    install — candidate selection has to consult the disk, not just the probes.
    """
    try:
        ir_ok = model_path.is_dir() and any(model_path.iterdir())
    except OSError:
        ir_ok = False
    gguf_ok = gguf_path.is_file() and mmproj_path.is_file()
    return ir_ok, gguf_ok

# runtime(+device) priority, lower number = higher priority; used to break ties when tier is equal.
_RUNTIME_PRIORITY = {
    "openvino": 0,
    ("llamacpp", "cuda"): 1,
    ("llamacpp", "vulkan"): 2,
    ("llamacpp", "cpu"): 3,
}


def _priority(candidate: dict) -> int:
    runtime = candidate["runtime"]
    device = candidate.get("device")
    if runtime == "openvino":
        return _RUNTIME_PRIORITY["openvino"]
    return _RUNTIME_PRIORITY.get((runtime, device), 99)


# ---------------------------------------------------------------------------
# Hardware probing (best-effort, all exceptions swallowed and [] returned)
# ---------------------------------------------------------------------------

def _probe_openvino() -> list[dict]:
    """Probe devices visible to OpenVINO: CPU is always present; GPU.x is
    classified discrete/integrated via FULL_DEVICE_NAME."""
    try:
        import openvino as ov

        core = ov.Core()
        out: list[dict] = []
        for device in core.available_devices:
            if device == "CPU":
                out.append({"runtime": "openvino", "device": "CPU",
                            "tier": 10, "label": "CPU"})
            elif device.startswith("GPU"):
                try:
                    full_name = str(core.get_property(device, "FULL_DEVICE_NAME"))
                except Exception:
                    full_name = ""
                is_discrete = "Arc" in full_name or "dGPU" in full_name
                tier = 30 if is_discrete else 20
                out.append({"runtime": "openvino", "device": device,
                            "tier": tier, "label": full_name or device})
        return out
    except Exception:
        return []


def _probe_nvidia() -> list[dict]:
    """Probe for NVIDIA GPU: nvidia-smi exists and lists at least one card."""
    try:
        if not shutil.which("nvidia-smi"):
            return []
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True,
            timeout=_PROBE_TIMEOUT_S)
        lines = [ln for ln in result.stdout.splitlines() if "GPU" in ln]
        if result.returncode != 0 or not lines:
            return []
        return [{"runtime": "llamacpp", "device": "cuda", "tier": 30,
                 "label": lines[0].strip()}]
    except Exception:
        return []


def _probe_amd() -> list[dict]:
    """Probe for AMD GPU: vulkaninfo/rocminfo can detect a gfx device.

    vulkaninfo/rocminfo's output format isn't reliable enough to distinguish
    discrete from integrated, so it's uniformly scored as iGPU (tier=20) -
    better to underscore than let an integrated GPU get misjudged as
    discrete and jump the priority queue.
    """
    try:
        for tool, args in (("vulkaninfo", ["vulkaninfo", "--summary"]),
                            ("rocminfo", ["rocminfo"])):
            if not shutil.which(tool):
                continue
            result = subprocess.run(
                args, capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S)
            if "gfx" in result.stdout.lower():
                return [{"runtime": "llamacpp", "device": "vulkan", "tier": 20,
                         "label": f"AMD ({tool})"}]
        return []
    except Exception:
        return []


def probe_hardware() -> list[dict]:
    """Merge the candidate lists from all three probes, unsorted (see `rank` for sorting)."""
    candidates: list[dict] = []
    candidates.extend(_probe_openvino())
    candidates.extend(_probe_nvidia())
    candidates.extend(_probe_amd())
    return candidates


def rank(candidates: list[dict]) -> list[dict]:
    """Sort by (tier descending, runtime priority)."""
    return sorted(candidates, key=lambda c: (-c["tier"], _priority(c)))


# ---------------------------------------------------------------------------
# candidate -> backend instance
# ---------------------------------------------------------------------------

def _candidate_spec(candidate: dict) -> str:
    return f"{candidate['runtime']}:{candidate['device']}"


def _is_valid_spec(spec: str) -> bool:
    """Validate whether an explicitly configured `runtime:device` is a value build_backend recognizes."""
    if ":" not in spec:
        return False
    runtime, _, device = spec.partition(":")
    if runtime == "openvino":
        return bool(device)
    if runtime == "llamacpp":
        return device in ("cuda", "vulkan", "cpu")
    return False


def build_backend(spec: str, *, model_path: Path, gguf_path: Path,
                   mmproj_path: Path, idle_ttl_s: int):
    """Turn a `runtime:device` spec into a (not-yet-loaded) backend instance."""
    runtime, _, device = spec.partition(":")
    if runtime == "openvino":
        return OpenVINOCaptionBackend(
            model_path=model_path, device=device, idle_ttl_s=idle_ttl_s)
    if runtime == "llamacpp":
        if device == "cuda":
            return LlamaCppCaptionBackend(
                gguf_path=gguf_path, mmproj_path=mmproj_path,
                n_gpu_layers=-1, backend_tag="cuda", idle_ttl_s=idle_ttl_s)
        if device == "vulkan":
            return LlamaCppCaptionBackend(
                gguf_path=gguf_path, mmproj_path=mmproj_path,
                n_gpu_layers=-1, backend_tag="vulkan", idle_ttl_s=idle_ttl_s)
        if device == "cpu":
            return LlamaCppCaptionBackend(
                gguf_path=gguf_path, mmproj_path=mmproj_path,
                n_gpu_layers=0, backend_tag="cpu", idle_ttl_s=idle_ttl_s)
    raise ValueError(f"illegal backend spec: {spec!r}")


# ---------------------------------------------------------------------------
# selection entry point + fallback wrapper
# ---------------------------------------------------------------------------

def select_caption_backend(*, vlm_device: str, model_path: Path,
                            gguf_path: Path, mmproj_path: Path,
                            idle_ttl_s: int) -> "SelectingCaptionBackend":
    """Pick a candidate chain per config, and return a `SelectingCaptionBackend`
    with built-in failure-demotion capability.

    - `vlm_device == "auto"`: probe + score, and use the ranked result as the candidate chain.
    - Explicit `runtime:device` (e.g. `llamacpp:cuda`): only this one candidate
      (the chain's tail still gets an automatic CPU fallback appended in
      `SelectingCaptionBackend` — openvino:CPU or llamacpp:cpu, whichever the
      on-disk weights can actually load).
    - Invalid value (neither auto nor a recognized spec): warn, then fall back to auto.
    """
    probed = True
    if vlm_device == "auto":
        ranked = rank(probe_hardware())
    elif _is_valid_spec(vlm_device):
        runtime, _, device = vlm_device.partition(":")
        ranked = [{"runtime": runtime, "device": device, "tier": 0,
                   "label": vlm_device}]
        probed = False  # an explicit spec is the operator's call; don't second-guess it
    else:
        log.warning("invalid vlm_device config %r, falling back to auto hardware probe", vlm_device)
        ranked = rank(probe_hardware())

    # Probes describe the hardware, not the weights: an Intel-only machine
    # yields a pure-openvino chain, and if the disk only holds GGUF weights
    # (what the installer ships), every link would fail to load and captions
    # would be dead even though llama.cpp could serve them on CPU. When
    # exactly one weight form is present, drop the candidates that can never
    # load; the guaranteed tail matching the surviving form is appended by
    # SelectingCaptionBackend.
    if probed:
        ir_ok, gguf_ok = _weights_present(model_path, gguf_path, mmproj_path)
        if ir_ok != gguf_ok:
            keep = "openvino" if ir_ok else "llamacpp"
            kept = [c for c in ranked if c["runtime"] == keep]
            if len(kept) != len(ranked):
                log.info("VLM weights on disk only fit %s (IR present=%s, GGUF present=%s); "
                         "dropping %d candidate(s) that could never load",
                         keep, ir_ok, gguf_ok, len(ranked) - len(kept))
            ranked = kept

    if not ranked:
        ranked = []  # SelectingCaptionBackend appends the weight-appropriate tail

    return SelectingCaptionBackend(
        ranked, model_path=model_path, gguf_path=gguf_path,
        mmproj_path=mmproj_path, idle_ttl_s=idle_ttl_s)


class SelectingCaptionBackend:
    """Fallback wrapper: holds a priority-ordered candidate chain, exposing
    the same CaptionBackend interface as a concrete backend
    (caption/unload/version/is_loaded), transparent to callers like
    VisualPipeline.

    `caption()` only demotes down the candidate chain when the currently
    active backend **fails to load** - a single image's inference
    failure/empty output (backend already loaded successfully, `is_loaded`
    is True) does not trigger demotion; the exception is re-raised to the
    caller as-is, so a healthy backend doesn't get permanently and wrongly
    demoted to CPU over an occasional bad image. On demotion, the backend is
    rebuilt and retried; the chain's tail always has a CPU fallback appended
    (openvino:CPU, or llamacpp:cpu when only GGUF weights are on disk —
    pure CPU inference, which should in theory always work) - if
    it still fails after demoting all the way to the tail (and it's still a
    load failure), the exception is re-raised as-is. Each demotion logs a
    warning, so ops can figure out "why isn't this using the expected
    backend".
    """

    def __init__(self, ranked: list[dict], *, model_path: Path,
                 gguf_path: Path, mmproj_path: Path, idle_ttl_s: int) -> None:
        self._ranked = list(ranked)

        # The guaranteed tail must be loadable with the weights actually on
        # disk: openvino:CPU when the IR form exists (the historical default),
        # llamacpp:cpu when only GGUF weights are present — an openvino tail
        # without IR weights is a fallback in name only.
        ir_ok, gguf_ok = _weights_present(model_path, gguf_path, mmproj_path)
        tail = dict(_GGUF_FALLBACK_CANDIDATE) if (gguf_ok and not ir_ok) else dict(_FALLBACK_CANDIDATE)
        last = self._ranked[-1] if self._ranked else None
        if last is None or (last["runtime"], last["device"]) != (tail["runtime"], tail["device"]):
            self._ranked.append(tail)

        self._model_path = model_path
        self._gguf_path = gguf_path
        self._mmproj_path = mmproj_path
        self._idle_ttl_s = idle_ttl_s

        self._index = 0
        self._backend = self._build(self._index)

    def _build(self, index: int):
        candidate = self._ranked[index]
        return build_backend(
            _candidate_spec(candidate),
            model_path=self._model_path, gguf_path=self._gguf_path,
            mmproj_path=self._mmproj_path, idle_ttl_s=self._idle_ttl_s)

    def _demote(self) -> bool:
        """Demote to the next item in the candidate chain; returns False once
        the chain's tail is reached.

        Explicitly `.unload()`s the old backend before rebuilding (following
        the `_BaseCaptionBackend._unload_locked` pattern), rather than
        relying on GC to implicitly reclaim GB-scale native objects.
        """
        if self._index + 1 >= len(self._ranked):
            return False
        self._backend.unload()
        self._index += 1
        candidate = self._ranked[self._index]
        log.warning("VLM backend demoted: switching to %s (%s)",
                    _candidate_spec(candidate), candidate.get("label", ""))
        self._backend = self._build(self._index)
        return True

    def caption(self, image_bytes: bytes) -> str:
        while True:
            try:
                return self._backend.caption(image_bytes)
            except CaptionError:
                # is_loaded must be read after the exception occurs, not from a
                # snapshot sampled before the call - on a cold-start first
                # frame, is_loaded=False before the call, but `_load_pipe`
                # inside this call may already have succeeded (`_pipe` set),
                # with `_infer` failing afterward; only the post-exception
                # is_loaded reflects the real state. Only demote on a load
                # failure (_pipe stays None, is_loaded still False); a single
                # inference failure/empty output on an already-loaded backend
                # (is_loaded is True) doesn't change backend selection, the
                # exception is re-raised to the caller as-is.
                if self._backend.is_loaded or not self._demote():
                    raise

    def unload(self) -> None:
        self._backend.unload()

    @property
    def is_loaded(self) -> bool:
        return self._backend.is_loaded

    @property
    def version(self) -> str:
        return self._backend.version
