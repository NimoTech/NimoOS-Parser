"""bge-reranker-v2-m3 on OpenVINO GPU.

Cross-encoder with a single-logit classification head (text-classification
IR export). Scores are sigmoid-normalized to match FlagReranker's
compute_score(normalize=True) contract that Search relies on. Pair encoding
uses the standard two-sequence path with max_length=512 (FlagReranker's
default passage budget); golden parity vs the torch class is enforced by
tests/test_text_ov_parity.py.
"""
import logging
import threading
from typing import Optional

import numpy as np

from parser.config import load_settings

log = logging.getLogger("parser.model_reranker_ov")

_MAX_LENGTH = 512  # FlagReranker default


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class BGERerankerOV:
    _instance: Optional["BGERerankerOV"] = None
    version = "bge-reranker-v2-m3/v1"
    # Single-concurrency, mirroring model_reranker.py: /rerank is a sync
    # handler run in a threadpool, requests really can overlap.
    _lock = threading.RLock()

    def __init__(self, compiled, tokenizer):
        self._compiled = compiled
        self._tokenizer = tokenizer
        self.device = "gpu"

    @classmethod
    def load(cls) -> "BGERerankerOV":
        with cls._lock:
            if cls._instance is not None:
                return cls._instance

            import openvino
            from transformers import AutoTokenizer

            path = load_settings().text_rerank_ov_path
            core = openvino.Core()
            compiled = core.compile_model(str(path / "openvino_model.xml"), "GPU")
            tokenizer = AutoTokenizer.from_pretrained(str(path))
            cls._instance = cls(compiled, tokenizer)
            log.info("BGERerankerOV loaded on GPU from %s", path)
            return cls._instance

    @classmethod
    def unload(cls) -> None:
        with cls._lock:
            cls._instance = None
        from parser.memutil import trim_malloc
        trim_malloc()

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        with self._lock:
            enc = self._tokenizer(
                [query] * len(candidates),
                [c["text"] for c in candidates],
                padding=True, truncation=True,
                max_length=_MAX_LENGTH, return_tensors="np")
            feed = {"input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"]}
            logits = self._compiled(feed)[self._compiled.output(0)]
            scores = _sigmoid(logits.reshape(-1).astype(np.float64))
        return [{"id": c["id"], "score": float(s)}
                for c, s in zip(candidates, scores)]
