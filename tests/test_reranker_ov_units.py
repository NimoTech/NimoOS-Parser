import numpy as np

from parser.model_reranker_ov import _sigmoid


def test_sigmoid_matches_reference():
    x = np.array([-2.0, 0.0, 3.0])
    out = _sigmoid(x)
    ref = 1.0 / (1.0 + np.exp(-x))
    assert np.allclose(out, ref)
    assert out[1] == 0.5
