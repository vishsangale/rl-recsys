from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rl_recsys.environments.kuairec import KuaiRecEnv


def _interactions(n_users=5, n_items=20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame([
        {"user_id": u, "item_id": i, "rating": float(rng.uniform(0, 1)), "timestamp": u * 1000 + i}
        for u in range(n_users) for i in range(n_items)
    ])


def _item_features(n_items=20) -> pd.DataFrame:
    return pd.DataFrame([
        {"item_id": i, "cat_0": i % 2, "cat_1": (i // 2) % 2}
        for i in range(n_items)
    ])


def test_reset_obs_shape(tmp_path):
    _interactions().to_parquet(tmp_path / "interactions.parquet", index=False)
    env = KuaiRecEnv(tmp_path, slate_size=1, num_candidates=10, feature_dim=8, feature_source="hashed", seed=0)
    obs = env.reset(seed=42)
    assert obs.user_features.shape == (8,)
    assert obs.candidate_features.shape == (10, 8)
    assert obs.candidate_ids.shape == (10,)


def test_step_done_true(tmp_path):
    _interactions().to_parquet(tmp_path / "interactions.parquet", index=False)
    env = KuaiRecEnv(tmp_path, slate_size=1, num_candidates=10, feature_dim=8, feature_source="hashed", seed=0)
    env.reset(seed=0)
    step = env.step(np.array([0]))
    assert step.done is True


def test_reward_in_zero_one(tmp_path):
    _interactions().to_parquet(tmp_path / "interactions.parquet", index=False)
    env = KuaiRecEnv(tmp_path, slate_size=1, num_candidates=10, feature_dim=8, feature_source="hashed", seed=0)
    for seed in range(20):
        obs = env.reset(seed=seed)
        step = env.step(np.array([0]))
        assert 0.0 <= step.reward <= 1.0


def test_reward_bounded_when_rating_exceeds_one(tmp_path):
    # watch_ratio > 1.0 occurs in real KuaiRec for replayed videos; pipeline must clip
    df = _interactions()
    df["rating"] = 2.5
    df.to_parquet(tmp_path / "interactions.parquet", index=False)
    env = KuaiRecEnv(tmp_path, slate_size=1, num_candidates=10, feature_dim=8, feature_source="hashed", seed=0)
    for seed in range(5):
        env.reset(seed=seed)
        step = env.step(np.array([0]))
        assert step.reward <= 1.0


def test_native_raises_without_item_features(tmp_path):
    _interactions().to_parquet(tmp_path / "interactions.parquet", index=False)
    with pytest.raises(FileNotFoundError, match="item_features.parquet"):
        KuaiRecEnv(tmp_path, feature_source="native", num_candidates=10, feature_dim=8)


def test_native_candidate_feature_shape(tmp_path):
    _interactions().to_parquet(tmp_path / "interactions.parquet", index=False)
    _item_features().to_parquet(tmp_path / "item_features.parquet", index=False)
    env = KuaiRecEnv(tmp_path, slate_size=1, num_candidates=10, feature_dim=8, feature_source="native", seed=0)
    obs = env.reset(seed=0)
    assert obs.candidate_features.shape == (10, 8)
