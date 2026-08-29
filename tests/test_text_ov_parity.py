"""Golden parity: OpenVINO text backend vs the torch (FlagEmbedding) classes.

Heavy and environment-dependent (needs converted IR + an OpenVINO GPU +
the HF-cached torch models), so it only runs when explicitly requested:

    PARSER_OV_PARITY=1 python -m pytest tests/test_text_ov_parity.py -v

Thresholds are the spec's acceptance bar (spec 2026-08-29 §6): dense cosine
>= 0.999, identical sparse token-id sets with per-weight |delta| <= 0.01,
rerank per-pair |delta| <= 0.01 with identical ordering.
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("PARSER_OV_PARITY") != "1",
    reason="opt-in: set PARSER_OV_PARITY=1 (needs IR weights + GPU + models)")

TEXTS = [
    "NimoOS 的网关服务是系统唯一对外入口,默认监听 80 端口。",
    "The parser service pins Python 3.11 because rapidocr has no 3.12 wheel.",
    "残差连接通过将输入直接加到输出上,缓解深层网络的梯度消失问题。",
    "short",
    "标点、Punctuation!?;——混合 mixed 123 テスト",
    "长文本 " + "深度学习模型的训练需要大量数据和算力。" * 60,  # exercises truncation
]

QUERY = "什么是残差连接?"
CANDIDATES = [
    {"id": "a", "text": "残差连接把输入加到输出,缓解梯度消失。"},
    {"id": "b", "text": "今天的天气很好,适合户外运动。"},
    {"id": "c", "text": "Residual connections add the input to the output."},
    {"id": "d", "text": "梯度下降是一种优化算法。"},
]


def _cos(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_embed_parity():
    from parser.model_bge_m3 import BGEM3
    from parser.model_bge_m3_ov import BGEM3OV

    ref = BGEM3.load(device="cpu").embed_text(TEXTS)
    got = BGEM3OV.load().embed_text(TEXTS)

    for i, (r, g) in enumerate(zip(ref, got)):
        assert _cos(r["dense"], g["dense"]) >= 0.999, f"dense drift at text {i}"
        r_sparse = dict(zip(r["sparse"]["indices"], r["sparse"]["values"]))
        g_sparse = dict(zip(g["sparse"]["indices"], g["sparse"]["values"]))
        assert set(r_sparse) == set(g_sparse), f"sparse token set differs at {i}"
        for tok, rv in r_sparse.items():
            assert abs(rv - g_sparse[tok]) <= 0.01, f"sparse weight drift tok {tok} at {i}"


def test_rerank_parity():
    from parser.model_reranker import BGEReranker
    from parser.model_reranker_ov import BGERerankerOV

    ref = BGEReranker.load(device="cpu").rerank(QUERY, CANDIDATES)
    got = BGERerankerOV.load().rerank(QUERY, CANDIDATES)

    ref_by_id = {r["id"]: r["score"] for r in ref}
    got_by_id = {g["id"]: g["score"] for g in got}
    for cid in ref_by_id:
        assert abs(ref_by_id[cid] - got_by_id[cid]) <= 0.01, f"score drift {cid}"
    ref_order = sorted(ref_by_id, key=ref_by_id.get, reverse=True)
    got_order = sorted(got_by_id, key=got_by_id.get, reverse=True)
    assert ref_order == got_order
