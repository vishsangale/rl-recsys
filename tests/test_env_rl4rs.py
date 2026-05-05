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


def test_reward_equals_click_sum(tmp_path):
    rows = []
    rng = np.random.default_rng(0)
    for step in range(3):
        rows.append({
            "session_id": 0,
            "step": step,
            "user_state": [0.0, 0.0, 0.0, 0.0],
            "slate": [10, 20, 30],
            "item_features": rng.standard_normal((3, 4)).tolist(),
            "clicks": [1, 0, 1],
        })
    pd.DataFrame(rows).to_parquet(tmp_path / "sessions.parquet", index=False)

    env = RL4RSEnv(tmp_path, slate_size=3, feature_dim=4, feature_source="native", seed=0)
    env.reset(seed=0)
    result = env.step(np.array([0, 1, 2]))
    assert result.reward == pytest.approx(2.0)


def test_inconsistent_slate_lengths_raises(tmp_path):
    rows = [
        {"session_id": 0, "step": 0, "user_state": [0.0] * 4,
         "slate": [1, 2, 3], "item_features": [[0.0] * 4] * 3, "clicks": [0] * 3},
        {"session_id": 0, "step": 1, "user_state": [0.0] * 4,
         "slate": [1, 2], "item_features": [[0.0] * 4] * 2, "clicks": [0] * 2},
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "sessions.parquet", index=False)
    with pytest.raises(ValueError, match="Inconsistent slate lengths"):
        RL4RSEnv(tmp_path, slate_size=3, feature_dim=4, feature_source="native", seed=0)
