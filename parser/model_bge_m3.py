from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from FlagEmbedding import BGEM3FlagModel


class BGEM3:
    _instance: Optional["BGEM3"] = None
    version = "bge-m3/v1"
    dim = 1024

    def __init__(self, model: "BGEM3FlagModel") -> None:
        self._model = model

    @classmethod
    def load(cls, *, use_fp16: bool = False) -> "BGEM3":
        if cls._instance is None:
            from FlagEmbedding import BGEM3FlagModel  # deferred until first use
            model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=use_fp16)
            cls._instance = cls(model)
        return cls._instance

    def embed_text(self, texts: list[str]) -> list[dict]:
        result = self._model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            batch_size=16,
            max_length=8192,
        )
        out = []
        for i, _ in enumerate(texts):
            dense = result["dense_vecs"][i].tolist()
            lex = result["lexical_weights"][i]
            indices = [int(k) for k in lex.keys()]
            values = [float(lex[k]) for k in lex.keys()]
            out.append({
                "dense": dense,
                "sparse": {"indices": indices, "values": values},
            })
        return out
