from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rl_recsys.environments.base import RecObs, RecStep
from rl_recsys.environments.dataset_base import BanditDatasetEnv


class _SimpleBanditEnv(BanditDatasetEnv):
    def _compute_reward(self, row, clicks):
        return float(clicks.sum())


def _interactions(n_users=5, n_items=20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = [
        {"user_id": u, "item_id": i, "rating": float(rng.uniform(0, 1))}
        for u in range(n_users)
        for i in range(n_items)
    ]
    return pd.DataFrame(rows)


def test_bandit_reset_obs_shape():
    env = _SimpleBanditEnv(_interactions(), slate_size=2, num_candidates=10, feature_dim=8, feature_source="hashed", seed=0)
    obs = env.reset(seed=42)
    assert obs.user_features.shape == (8,)
    assert obs.candidate_features.shape == (10, 8)
    assert obs.candidate_ids.shape == (10,)


def test_bandit_step_done_true():
    env = _SimpleBanditEnv(_interactions(), slate_size=2, num_candidates=10, feature_dim=8, feature_source="hashed", seed=0)
    env.reset(seed=0)
    step = env.step(np.array([0, 1]))
    assert step.done is True
    assert step.clicks.shape == (2,)
    assert isinstance(step.reward, float)


def test_bandit_step_before_reset_raises():
    env = _SimpleBanditEnv(_interactions(), num_candidates=10, feature_dim=8, feature_source="hashed", seed=0)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.array([0]))


def test_bandit_empty_df_raises():
    with pytest.raises(ValueError, match="empty"):
        _SimpleBanditEnv(pd.DataFrame(columns=["user_id", "item_id"]), num_candidates=5, feature_dim=4, feature_source="hashed")


def test_bandit_too_many_candidates_raises():
    with pytest.raises(ValueError, match="num_candidates"):
        _SimpleBanditEnv(_interactions(n_items=5), num_candidates=10, feature_dim=4, feature_source="hashed")


def test_bandit_slate_exceeds_candidates_raises():
    with pytest.raises(ValueError, match="slate_size"):
        _SimpleBanditEnv(_interactions(), slate_size=15, num_candidates=10, feature_dim=4, feature_source="hashed")


def test_bandit_feature_dim_too_small_raises():
    with pytest.raises(ValueError, match="feature_dim"):
        _SimpleBanditEnv(_interactions(), num_candidates=10, feature_dim=2, feature_source="hashed")


def test_bandit_properties():
    env = _SimpleBanditEnv(_interactions(), slate_size=3, num_candidates=10, feature_dim=8, feature_source="hashed")
    assert env.slate_size == 3
    assert env.num_candidates == 10
    assert env.user_dim == 8
    assert env.item_dim == 8


def test_bandit_positive_item_in_candidates():
    env = _SimpleBanditEnv(_interactions(), slate_size=1, num_candidates=10, feature_dim=8, feature_source="hashed", seed=7)
    obs = env.reset(seed=7)
    # Selecting all candidates must hit the positive item at least once
    step = env.step(np.arange(10))
    assert step.clicks.sum() >= 1.0
