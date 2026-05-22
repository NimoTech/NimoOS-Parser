import pytest

pytest.importorskip("FlagEmbedding")
pytestmark = pytest.mark.slow

from parser.model_bge_m3 import BGEM3


def test_embed_text_returns_dense_1024_and_sparse():
    m = BGEM3.load()
    out = m.embed_text(["hello world"])
    assert len(out) == 1
    assert len(out[0]["dense"]) == 1024
    assert "indices" in out[0]["sparse"]
    assert "values" in out[0]["sparse"]
    assert len(out[0]["sparse"]["indices"]) > 0


def test_embed_text_batch():
    m = BGEM3.load()
    out = m.embed_text(["alpha", "beta", "gamma"])
    assert len(out) == 3


def test_model_version_string():
    m = BGEM3.load()
    assert m.version.startswith("bge-m3/")
