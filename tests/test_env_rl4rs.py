from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rl_recsys.environments.rl4rs import RL4RSEnv


def _sessions_parquet(tmp_path, n_sessions=2, n_steps=3, slate_size=3, n_user_feats=4, n_item_feats=4):
    rng = np.random.default_rng(0)
    rows = []
    for sid in range(n_sessions):
        for step in range(n_steps):
            slate = rng.integers(0, 100, size=slate_size).tolist()
            rows.append({
                "session_id": sid,
                "step": step,
                "user_state": rng.standard_normal(n_user_feats).tolist(),
                "slate": [int(x) for x in slate],
                "item_features": rng.standard_normal((slate_size, n_item_feats)).tolist(),
                "clicks": rng.integers(0, 2, size=slate_size).tolist(),
            })
    df = pd.DataFrame(rows)
    df.to_parquet(tmp_path / "sessions.parquet", index=False)


def test_reset_obs_shape(tmp_path):
    _sessions_parquet(tmp_path, n_user_feats=4, n_item_feats=4, slate_size=3)
    env = RL4RSEnv(tmp_path, slate_size=3, feature_dim=4, feature_source="native", seed=0)
    obs = env.reset(seed=0)
    assert obs.user_features.shape == (4,)
    assert obs.candidate_features.shape == (3, 4)
    assert obs.candidate_ids.shape == (3,)


def test_done_false_mid_session(tmp_path):
    _sessions_parquet(tmp_path, n_steps=3, slate_size=3, n_user_feats=4, n_item_feats=4)
    env = RL4RSEnv(tmp_path, slate_size=3, feature_dim=4, feature_source="native", seed=0)
    env.reset(seed=0)
    step = env.step(np.array([0, 1, 2]))
    assert step.done is False


def test_done_true_at_session_end(tmp_path):
    _sessions_parquet(tmp_path, n_steps=3, slate_size=3, n_user_feats=4, n_item_feats=4)
    env = RL4RSEnv(tmp_path, slate_size=3, feature_dim=4, feature_source="native", seed=0)
    env.reset(seed=0)
    env.step(np.array([0, 1, 2]))
    env.step(np.array([0, 1, 2]))
    step = env.step(np.array([0, 1, 2]))
    assert step.done is True


def test_native_feature_dim_mismatch_raises(tmp_path):
    _sessions_parquet(tmp_path, n_user_feats=4, n_item_feats=4, slate_size=3)
    with pytest.raises(ValueError, match="feature_dim"):
        RL4RSEnv(tmp_path, slate_size=3, feature_dim=8, feature_source="native", seed=0)


def test_hashed_mode_works(tmp_path):
    _sessions_parquet(tmp_path, n_user_feats=4, n_item_feats=4, slate_size=3)
    env = RL4RSEnv(tmp_path, slate_size=3, feature_dim=8, feature_source="hashed", seed=0)
    obs = env.reset(seed=0)
    assert obs.user_features.shape == (8,)
    assert obs.candidate_features.shape == (3, 8)


def test_reward_accumulates_across_steps(tmp_path):
    _sessions_parquet(tmp_path, n_steps=3, slate_size=3, n_user_feats=4, n_item_feats=4)
    env = RL4RSEnv(tmp_path, slate_size=3, feature_dim=4, feature_source="native", seed=0)
    env.reset(seed=0)
    total = 0.0
    for _ in range(3):
        step = env.step(np.array([0, 1, 2]))
        total += step.reward
    assert total >= 0.0
