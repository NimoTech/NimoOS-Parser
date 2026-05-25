from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from FlagEmbedding import BGEM3FlagModel


class BGEM3:
    _instance: Optional["BGEM3"] = None
    _instance_device: Optional[str] = None
    version = "bge-m3/v1"
    dim = 1024

    def __init__(self, model: "BGEM3FlagModel", device: str) -> None:
        self._model = model
        self.device = device

    @classmethod
    def load(cls, *, device: str = "cuda", use_fp16: Optional[bool] = None) -> "BGEM3":
        # CPU does not support fp16 in PyTorch (silent slowdown / incorrect output),
        # so we force fp32 on CPU regardless of caller preference.
        if use_fp16 is None:
            use_fp16 = device == "cuda"
        if device == "cpu":
            use_fp16 = False

        if cls._instance is not None and cls._instance_device == device:
            return cls._instance
        # device changed — unload previous instance first
        cls.unload()

        from FlagEmbedding import BGEM3FlagModel  # deferred until first use
        model = BGEM3FlagModel(
            "BAAI/bge-m3", use_fp16=use_fp16, devices=[device],
        )
        cls._instance = cls(model, device)
        cls._instance_device = device
        return cls._instance

    @classmethod
    def unload(cls) -> None:
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

    def embed_text(self, texts: list[str]) -> list[dict]:
        result = self._model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            batch_size=8,
            max_length=1024,
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
