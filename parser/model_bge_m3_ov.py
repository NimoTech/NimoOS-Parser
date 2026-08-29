"""bge-m3 embedding on OpenVINO GPU, sparse head included.

The backbone (XLM-R, exported to OpenVINO IR with last_hidden_state output)
runs on the iGPU. The sparse head is BAAI's published Linear(1024->1)+ReLU
(weights shipped as sparse_linear.npz next to the IR), re-applied here in
numpy with the exact aggregation semantics of FlagEmbedding's
_process_token_weights: drop special tokens, drop w<=0, max per token id.
Output shape matches parser.model_bge_m3.BGEM3.embed_text one-for-one;
golden parity is enforced by tests/test_text_ov_parity.py.
"""
import logging
import threading
from typing import Optional

import numpy as np

from parser.config import load_settings

log = logging.getLogger("parser.model_bge_m3_ov")

_BATCH_SIZE = 8      # match BGEM3.embed_text
_MAX_LENGTH = 1024   # match BGEM3.embed_text


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _aggregate_sparse(token_weights: np.ndarray, input_ids: np.ndarray,
                      unused_ids: set[int]) -> dict[int, float]:
    result: dict[int, float] = {}
    for w, idx in zip(token_weights.tolist(), input_ids.tolist()):
        if idx in unused_ids or w <= 0:
            continue
        if w > result.get(idx, 0.0):
            result[idx] = w
    return result


class BGEM3OV:
    _instance: Optional["BGEM3OV"] = None
    version = "bge-m3/v1"
    dim = 1024
    _lock = threading.RLock()

    def __init__(self, compiled, tokenizer, sparse_w, sparse_b, unused_ids):
        self._compiled = compiled
        self._tokenizer = tokenizer
        self._sparse_w = sparse_w      # (1024,) float32
        self._sparse_b = sparse_b      # scalar float32
        self._unused_ids = unused_ids
        self.device = "gpu"

    @classmethod
    def load(cls) -> "BGEM3OV":
        with cls._lock:
            if cls._instance is not None:
                return cls._instance

            import openvino
            from transformers import AutoTokenizer

            path = load_settings().text_embed_ov_path
            core = openvino.Core()
            compiled = core.compile_model(str(path / "openvino_model.xml"), "GPU")
            tokenizer = AutoTokenizer.from_pretrained(str(path))

            head = np.load(path / "sparse_linear.npz")
            sparse_w = head["weight"].reshape(-1).astype(np.float32)  # (1024,)
            sparse_b = float(head["bias"].reshape(-1)[0])

            # Same special-token exclusion set as FlagEmbedding m3.py.
            unused_ids = set()
            for tok in ("cls_token", "eos_token", "pad_token", "unk_token"):
                if tok in tokenizer.special_tokens_map:
                    unused_ids.add(tokenizer.convert_tokens_to_ids(
                        tokenizer.special_tokens_map[tok]))

            cls._instance = cls(compiled, tokenizer, sparse_w, sparse_b, unused_ids)
            log.info("BGEM3OV loaded on GPU from %s", path)
            return cls._instance

    @classmethod
    def unload(cls) -> None:
        with cls._lock:
            cls._instance = None
        from parser.memutil import trim_malloc
        trim_malloc()

    def embed_text(self, texts: list[str]) -> list[dict]:
        out: list[dict] = []
        with self._lock:
            for start in range(0, len(texts), _BATCH_SIZE):
                batch = texts[start:start + _BATCH_SIZE]
                enc = self._tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=_MAX_LENGTH, return_tensors="np")
                res = self._compiled({
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                })
                hidden = res[self._compiled.output(0)]  # (B, T, 1024)
                dense = _l2_normalize(hidden[:, 0].astype(np.float32))
                # sparse head: relu(hidden @ W + b), then max-per-token-id
                tw = hidden.astype(np.float32) @ self._sparse_w + self._sparse_b
                tw = np.maximum(tw, 0.0)  # (B, T)
                for i in range(len(batch)):
                    n_tok = int(enc["attention_mask"][i].sum())
                    lex = _aggregate_sparse(
                        tw[i, :n_tok], enc["input_ids"][i, :n_tok],
                        self._unused_ids)
                    out.append({
                        "dense": dense[i].tolist(),
                        "sparse": {
                            "indices": [int(k) for k in lex.keys()],
                            "values": [float(v) for v in lex.values()],
                        },
                    })
        return out
