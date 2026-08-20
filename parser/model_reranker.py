import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from FlagEmbedding import FlagReranker


class BGEReranker:
    _instance: Optional["BGEReranker"] = None
    _instance_device: Optional[str] = None
    version = "bge-reranker-v2-m3/v1"
    # Single-concurrency for load + inference (mirroring model_vlm.py). The
    # /rerank route is a sync handler, so FastAPI runs it in a threadpool and
    # two requests really can overlap; on a CPU box that means two ~26s
    # inferences fighting for cores and double the resident model.
    # Reentrant because load() calls unload() and both take this lock.
    _lock = threading.RLock()

    def __init__(self, model: "FlagReranker", device: str) -> None:
        self._model = model
        self.device = device

    @classmethod
    def load(cls, *, device: str = "cuda", use_fp16: Optional[bool] = None) -> "BGEReranker":
        if use_fp16 is None:
            use_fp16 = device == "cuda"
        if device == "cpu":
            use_fp16 = False

        with cls._lock:
            if cls._instance is not None and cls._instance_device == device:
                return cls._instance
            cls.unload()

            from FlagEmbedding import FlagReranker  # deferred
            model = FlagReranker(
                "BAAI/bge-reranker-v2-m3", use_fp16=use_fp16, devices=[device],
            )
            cls._instance = cls(model, device)
            cls._instance_device = device
            return cls._instance

    @classmethod
    def unload(cls) -> None:
        with cls._lock:
            cls._instance = None
            cls._instance_device = None
            try:
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            from parser.memutil import trim_malloc
            trim_malloc()

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        pairs = [[query, c["text"]] for c in candidates]
        with self._lock:
            scores = self._model.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [scores]
        return [{"id": c["id"], "score": float(s)}
                for c, s in zip(candidates, scores)]
