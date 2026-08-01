"""Qwen3-VL caption adapter - the swappable inference backend for the visual pipeline.

Shares the lazy-loaded singleton approach with model_bge_m3.py, but adds two things:
1. Idle unload: the VLM int4 runtime is 4-6GB, so it can't stay resident like
   BGE-M3 does; a background sweeper thread unloads it once idle exceeds
   idle_ttl_s (borrowed from immich-ml's MODEL_TTL semantics).
2. Single-concurrency lock: VLM inference saturates CPU cores, so concurrent
   calls would just slow each other down and double memory use - a single
   lock serializes them (spec fixes VlmConcurrency=1; change the semantics
   here if that's ever relaxed).
openvino_genai / PIL / numpy are all deferred imports, so environments
without these deps (CI unit tests) can still import this module and test with
injected doubles.

`_BaseCaptionBackend` carries the lifecycle skeleton that's independent of
the concrete inference engine (lazy load / idle sweep / single-concurrency
lock / three-state wrapping); different inference backends (OpenVINO /
future other platforms) only need to subclass and implement `_load_pipe()`
(returns the underlying pipeline object) and `_infer(pipe, image_bytes) ->
str` (runs one inference with the given pipeline).
"""
import logging
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("parser.model_vlm")

PROMPT_V1 = (
    "Describe this photo in 1-3 English sentences for search indexing. "
    "Cover the main subjects, the scene, any visible actions, and notable "
    "details such as colors, counts, or visible text. Output plain text "
    "only, no preamble, no speculation about intent."
)

# The model identifier is fixed as qwen3-vl-4b-int4, decoupled from the
# actual storage path (model_path) - the version string is used for
# model_versions ledger comparisons and shouldn't drift just because the
# local path gets renamed.
_MODEL_ID = "qwen3-vl-4b-int4"
_PROMPT_VERSION = "prompt-v1"

_SWEEP_INTERVAL_S = 60


class CaptionError(Exception):
    """Caption generation failed (load failure / inference exception / empty output)."""


class _BaseCaptionBackend:
    """Lazy-loaded singleton skeleton for a caption inference backend (independent of the concrete inference engine).

    Subclasses must implement:
    - `_load_pipe()`: build and return the underlying pipeline object; should
      raise `CaptionError` on load failure (or any exception, which `caption`
      will wrap as a fallback).
    - `_infer(pipe, image_bytes) -> str`: run one inference on an image with
      the given pipeline, returning text (need not be stripped or
      empty-checked - `caption` handles that uniformly).

    Thread safety: `_lock` guards both the load state and inference calls,
    following the lazy-loaded singleton style (mirroring model_bge_m3.py),
    plus idle auto-unload and a single-concurrency inference lock.
    """

    def __init__(self, idle_ttl_s: int = 300) -> None:
        self.idle_ttl_s = idle_ttl_s
        self._pipe = None
        self._lock = threading.Lock()      # single-concurrency for load + inference
        self._last_used = 0.0
        self._sweeper_started = False
        # True = this backend cannot safely release native resources (e.g. a
        # missing llama-cpp private interface); idle sweep skips unloading
        # (staying resident is safer than "leaking every unload cycle");
        # explicit unload is unaffected.
        self._unload_disabled = False

    # -- injectable points (implemented by subclasses) --------------------
    def _load_pipe(self):
        raise NotImplementedError

    def _infer(self, pipe, image_bytes: bytes) -> str:
        raise NotImplementedError

    # -- lifecycle --------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def unload(self) -> None:
        with self._lock:
            self._unload_locked()

    def _close_pipe(self, pipe) -> None:
        """Native resource release hook run before unload (no-op by default).

        Backends holding native handles (llama.cpp's mtmd/mmproj context,
        etc.) must override this method to explicitly free them - just
        setting _pipe to None and leaving it to GC is not enough: the main
        root cause of the 2026-07-28 OOM was mmproj (~836MB) leaking on every
        unload cycle.
        """

    def _unload_locked(self) -> None:
        if self._pipe is not None:
            log.info("unloading idle VLM (ttl=%ss)", self.idle_ttl_s)
            try:
                self._close_pipe(self._pipe)
            except Exception:  # release failure must not block the unload, but must be logged
                log.exception("vlm _close_pipe failed (continuing unload)")
            self._pipe = None
            import gc
            gc.collect()
            from parser.memutil import trim_malloc
            trim_malloc()

    def _sweep(self, now: Optional[float] = None) -> None:
        """Idle sweep; the now parameter lets tests inject a virtual clock."""
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._unload_disabled:
                return
            if self._pipe is not None and now - self._last_used > self.idle_ttl_s:
                self._unload_locked()

    def _ensure_sweeper(self) -> None:
        if self._sweeper_started:
            return
        self._sweeper_started = True

        def loop() -> None:
            while True:
                time.sleep(_SWEEP_INTERVAL_S)
                try:
                    self._sweep()
                except Exception:  # a failed sweep isn't fatal, retry next round
                    log.exception("vlm sweeper failed")

        threading.Thread(target=loop, daemon=True,
                          name="vlm-idle-sweeper").start()

    # -- inference ----------------------------------------------------
    def caption(self, image_bytes: bytes) -> str:
        with self._lock:
            if self._pipe is None:
                # The three-state coverage (load failure / inference
                # exception / empty output) requires load failures to also be
                # normalized to CaptionError: _load_pipe already proactively
                # raises CaptionError for "directory doesn't exist", but the
                # raw exception thrown by the underlying pipeline
                # construction itself (corrupt IR, runtime error, etc.) would
                # otherwise propagate with its original type if not caught
                # here. Already-CaptionError exceptions pass through as-is,
                # to avoid double-wrapping and losing the original meaning.
                # On load failure self._pipe stays None (never assigned), so the next call can retry.
                try:
                    pipe = self._load_pipe()
                except CaptionError:
                    raise
                except Exception as exc:
                    raise CaptionError(f"vlm load failed: {exc}") from exc
                self._pipe = pipe
                self._ensure_sweeper()
            self._last_used = time.monotonic()
            try:
                result = self._infer(self._pipe, image_bytes)
            except Exception as exc:
                raise CaptionError(f"vlm inference failed: {exc}") from exc
            self._last_used = time.monotonic()
        # The underlying inference call's return type may not be a plain str
        # depending on the engine/version - uniformly coerce with str().
        text = str(result).strip()
        if not text:
            raise CaptionError("vlm returned empty caption")
        return text


class OpenVINOCaptionBackend(_BaseCaptionBackend):
    """Qwen3-VL (OpenVINO GenAI, int4) caption inference backend."""

    def __init__(self, model_path: Path, device: str = "CPU",
                 idle_ttl_s: int = 300) -> None:
        super().__init__(idle_ttl_s=idle_ttl_s)
        self.model_path = Path(model_path)
        self.device = device
        self.version = f"{_MODEL_ID}/{_PROMPT_VERSION}/{device}"

    def _load_pipe(self):
        import openvino_genai  # deferred: pip-installed at deploy time, not needed for unit tests
        if not self.model_path.is_dir():
            raise CaptionError(
                f"VLM model dir not found: {self.model_path} — run "
                "scripts/vlm/convert_qwen3vl.sh first")
        log.info("loading VLM from %s (%s)", self.model_path, self.device)
        return openvino_genai.VLMPipeline(str(self.model_path), self.device)

    def _decode_image(self, image_bytes: bytes):
        import io
        import numpy as np
        import openvino as ov
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return ov.Tensor(np.array(img))

    def _infer(self, pipe, image_bytes: bytes) -> str:
        tensor = self._decode_image(image_bytes)
        result = pipe.generate(PROMPT_V1, images=[tensor], max_new_tokens=128)
        return str(result)
