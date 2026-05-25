from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from FlagEmbedding import FlagReranker


class BGEReranker:
    _instance: Optional["BGEReranker"] = None
    version = "bge-reranker-v2-m3/v1"

    def __init__(self, model: "FlagReranker") -> None:
        self._model = model

    @classmethod
    def load(cls, *, use_fp16: bool = True) -> "BGEReranker":
        if cls._instance is None:
            from FlagEmbedding import FlagReranker  # deferred
            model = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=use_fp16)
            cls._instance = cls(model)
        return cls._instance

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        pairs = [[query, c["text"]] for c in candidates]
        scores = self._model.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [scores]
        return [{"id": c["id"], "score": float(s)}
                for c, s in zip(candidates, scores)]
