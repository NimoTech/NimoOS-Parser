import pytest

pytest.importorskip("FlagEmbedding")
pytestmark = pytest.mark.slow

from parser.model_reranker import BGEReranker


def test_rerank_orders_by_relevance():
    r = BGEReranker.load()
    query = "How to use Python decorators?"
    candidates = [
        {"id": "a", "text": "Python decorators wrap functions to add behavior."},
        {"id": "b", "text": "The capital of France is Paris."},
        {"id": "c", "text": "Decorator pattern in software design."},
    ]
    out = r.rerank(query, candidates)
    ids_sorted = [s["id"] for s in sorted(out, key=lambda s: -s["score"])]
    assert ids_sorted[-1] == "b"


def test_model_version():
    r = BGEReranker.load()
    assert r.version.startswith("bge-reranker-v2-m3/")
