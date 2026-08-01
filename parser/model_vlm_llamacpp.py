"""LlamaCpp (GGUF) caption adapter - the other inference path for multi-platform adaptation.

A sibling implementation to `OpenVINOCaptionBackend` in model_vlm.py under the
same `_BaseCaptionBackend` skeleton, targeting platforms without OpenVINO
support (or that prefer a generic GGUF quantized model) - e.g. AMD discrete
GPU (ROCm), NVIDIA (CUDA), or pure-CPU scenarios. It uniformly goes through
llama-cpp-python's multimodal (mmproj) chat completion interface, controlling
how many layers get offloaded to the GPU via `n_gpu_layers` (0 means pure CPU
inference).

llama_cpp is an optional dependency, deferred import: environments without
the package installed (CI unit tests) can still import this module and run
tests with an injected double (replacing `_load_pipe`), without erroring at
import time.
"""
import base64
import logging
from pathlib import Path

from parser.model_vlm import (
    PROMPT_V1,
    CaptionError,
    _BaseCaptionBackend,
    _MODEL_ID,
    _PROMPT_VERSION,
)

log = logging.getLogger("parser.model_vlm_llamacpp")


class LlamaCppCaptionBackend(_BaseCaptionBackend):
    """Qwen3-VL (llama.cpp GGUF quantized + mmproj multimodal projection) caption backend.

    - `gguf_path` / `mmproj_path`: the main model and multimodal projection
      weights (both required; mmproj is the bridging weight that projects
      the vision encoder into the language model's embedding space).
    - `n_gpu_layers`: number of transformer layers offloaded to the GPU, 0
      means pure CPU inference; the actual value is tuned by the caller
      based on VRAM size and platform (ROCm/CUDA).
    - `backend_tag`: labels the actual hardware/backend running (e.g.
      "cpu"/"rocm"/"cuda"), written into the version string so the
      model_versions ledger can distinguish captions produced by different
      hardware paths (same prompt but different inference engine/precision
      can't be treated as the same version).
    """

    def __init__(self, gguf_path: Path, mmproj_path: Path,
                 n_gpu_layers: int = 0, backend_tag: str = "cpu",
                 idle_ttl_s: int = 300) -> None:
        super().__init__(idle_ttl_s=idle_ttl_s)
        self.gguf_path = Path(gguf_path)
        self.mmproj_path = Path(mmproj_path)
        self.n_gpu_layers = n_gpu_layers
        self.backend_tag = backend_tag
        self.version = f"{_MODEL_ID}-gguf/{_PROMPT_VERSION}/{backend_tag}"
        self._chat_handler = None  # kept so the mtmd context can be explicitly released on unload

    def _load_pipe(self):
        import llama_cpp  # deferred: llama-cpp-python is installed per-platform at deploy time, not needed for unit tests

        if not self.gguf_path.is_file():
            raise CaptionError(f"VLM gguf not found: {self.gguf_path}")
        if not self.mmproj_path.is_file():
            raise CaptionError(f"VLM mmproj not found: {self.mmproj_path}")

        log.info("loading GGUF VLM from %s + mmproj %s (n_gpu_layers=%s, %s)",
                  self.gguf_path, self.mmproj_path, self.n_gpu_layers,
                  self.backend_tag)
        # MTMDChatHandler is llama.cpp's new unified multimodal entry point
        # (mtmd), which takes the GGUF mmproj projection weights and wires
        # the image encoding into the image_url content block of a chat
        # message; Qwen3-VL has no dedicated handler, so it goes through this
        # generic mtmd handler (verified via a local CPU smoke test).
        chat_handler = llama_cpp.llama_chat_format.MTMDChatHandler(
            clip_model_path=str(self.mmproj_path))
        if not hasattr(chat_handler, "_exit_stack"):
            # llama-cpp-python's private interface changed, so mmproj (the
            # mtmd context) can't be explicitly freed. Disable idle
            # auto-unload: staying resident is safer than "leaking ~836MB
            # every unload cycle" (the main root cause of the 2026-07-28 OOM).
            # This interface must be re-verified before upgrading this
            # dependency; see the version pin note in requirements.txt.
            self._unload_disabled = True
            log.error("MTMDChatHandler lacks _exit_stack; idle auto-unload "
                      "disabled to avoid mmproj native leak")
        self._chat_handler = chat_handler
        return llama_cpp.Llama(
            model_path=str(self.gguf_path),
            chat_handler=chat_handler,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=4096,
            verbose=False,
        )

    def _infer(self, pipe, image_bytes: bytes) -> str:
        # llama-cpp-python's multimodal chat completion follows the
        # OpenAI-compatible format: image_url takes a base64 data URI, it
        # doesn't support passing raw bytes directly.
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_uri = f"data:image/jpeg;base64,{b64}"
        response = pipe.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": PROMPT_V1},
                    ],
                }
            ],
            max_tokens=128,
        )
        return response["choices"][0]["message"]["content"]

    def _close_pipe(self, pipe) -> None:
        # llama-cpp-python <=0.3.34: Llama.close() only releases the language
        # model's own _stack; chat_handler is a plain attribute not on that
        # release chain. MTMDChatHandler has no close()/__del__, and
        # mtmd_free hangs off an ExitStack that's never closed -> without an
        # explicit close, every reload leaks the entire mmproj vision encoder (~836MB).
        try:
            close = getattr(pipe, "close", None)
            if close is not None:
                close()
        finally:
            handler, self._chat_handler = self._chat_handler, None
            stack = getattr(handler, "_exit_stack", None)
            if stack is not None:
                stack.close()
