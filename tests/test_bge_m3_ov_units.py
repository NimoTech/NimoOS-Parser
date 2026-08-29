import numpy as np

from parser.model_bge_m3_ov import _aggregate_sparse, _l2_normalize


def test_l2_normalize_rows():
    m = np.array([[3.0, 4.0], [0.0, 5.0]])
    out = _l2_normalize(m)
    assert np.allclose(out, [[0.6, 0.8], [0.0, 1.0]])


def test_aggregate_sparse_max_per_token_and_exclusions():
    # semantics replicated from FlagEmbedding m3.py _process_token_weights:
    # drop unused token ids, drop w<=0, keep max weight per token id.
    weights = np.array([0.5, 0.2, 0.9, 0.0, -0.1, 0.4])
    ids = np.array([7, 7, 7, 8, 9, 1])  # 1 = special token (excluded)
    out = _aggregate_sparse(weights, ids, unused_ids={1})
    assert out == {7: 0.9}


def test_aggregate_sparse_multiple_tokens():
    weights = np.array([0.3, 0.6, 0.6])
    ids = np.array([10, 11, 10])
    out = _aggregate_sparse(weights, ids, unused_ids=set())
    assert out == {10: 0.6, 11: 0.6}
