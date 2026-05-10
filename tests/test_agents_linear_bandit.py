from __future__ import annotations

import numpy as np
import pytest

from rl_recsys.agents._linear_base import LinearBanditBase
from rl_recsys.environments.base import RecObs


class _DummyLinear(LinearBanditBase):
    def select_slate(self, obs):
        return np.array([0, 1, 2], dtype=np.int64)

    def score_items(self, obs):
        features = self._candidate_features(obs)
        theta = np.linalg.solve(self._a_matrix, self._b_vector)
        return features @ theta


def test_linear_base_update_matches_linucb_accumulator():
    from rl_recsys.agents.linucb import LinUCBAgent

    obs = RecObs(
        user_features=np.array([1.0, 0.0, 0.0, 0.0]),
        candidate_features=np.eye(5, 3),
        candidate_ids=np.arange(5, dtype=np.int64),
    )
    slate = np.array([0, 1, 2], dtype=np.int64)
    clicks = np.array([1, 0, 1], dtype=np.float64)

    a = _DummyLinear(slate_size=3, user_dim=4, item_dim=3)
    b = LinUCBAgent(slate_size=3, user_dim=4, item_dim=3)
    a.update(obs, slate, reward=2.0, clicks=clicks, next_obs=obs)
    b.update(obs, slate, reward=2.0, clicks=clicks, next_obs=obs)
    np.testing.assert_allclose(a._a_matrix, b._a_matrix)
    np.testing.assert_allclose(a._b_vector, b._b_vector)


def test_linear_base_features_validate_shape():
    bad_obs = RecObs(
        user_features=np.zeros(99),  # wrong dim
        candidate_features=np.zeros((5, 3)),
        candidate_ids=np.arange(5, dtype=np.int64),
    )
    a = _DummyLinear(slate_size=3, user_dim=4, item_dim=3)
    with pytest.raises(ValueError, match="user_features shape"):
        a.score_items(bad_obs)
