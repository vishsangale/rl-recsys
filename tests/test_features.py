import numpy as np
from rl_recsys.environments.features import hashed_vector


def test_hashed_vector_shape():
    v = hashed_vector("user", 42, 16)
    assert v.shape == (16,)


def test_hashed_vector_is_unit_norm():
    v = hashed_vector("item", 7, 8)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_hashed_vector_deterministic():
    v1 = hashed_vector("user", 1, 16)
    v2 = hashed_vector("user", 1, 16)
    np.testing.assert_array_equal(v1, v2)


def test_hashed_vector_differs_by_prefix():
    v1 = hashed_vector("user", 1, 16)
    v2 = hashed_vector("item", 1, 16)
    assert not np.allclose(v1, v2)
