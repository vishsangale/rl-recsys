from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from rl_recsys.environments.finn_no_slate import FinnNoSlateEnv


def _slates(n=30) -> pd.DataFrame:
    # n=30 so unique clicked items (30) >= num_candidates (25) for base class validation
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        slate = list(range(i * 25, i * 25 + 25))  # non-overlapping slates
        click_idx = int(rng.integers(0, 25))
        rows.append({
            "request_id": i,
            "user_id": i % 5,
            "slate": slate,
            "clicks": click_idx,
            "timestamp": 1600000000 + i * 1000,
        })
    return pd.DataFrame(rows)


def test_reset_obs_shape(tmp_path):
    _slates().to_parquet(tmp_path / "slates.parquet", index=False)
    env = FinnNoSlateEnv(tmp_path, slate_size=5, feature_dim=8, seed=0)
    obs = env.reset(seed=0)
    assert obs.user_features.shape == (8,)
    assert obs.candidate_features.shape == (25, 8)
    assert obs.candidate_ids.shape == (25,)


def test_candidate_ids_match_a_logged_slate(tmp_path):
    df = _slates()
    df.to_parquet(tmp_path / "slates.parquet", index=False)
    env = FinnNoSlateEnv(tmp_path, slate_size=1, feature_dim=8, seed=0)
    for _ in range(5):
        obs = env.reset()
        candidate_set = set(obs.candidate_ids.tolist())
        found = any(
            set(row["slate"]) == candidate_set
            for _, row in df.iterrows()
        )
        assert found, "candidate_ids do not match any logged slate"


def test_step_done_true(tmp_path):
    _slates().to_parquet(tmp_path / "slates.parquet", index=False)
    env = FinnNoSlateEnv(tmp_path, slate_size=1, feature_dim=8, seed=0)
    env.reset(seed=0)
    step = env.step(np.array([0]))
    assert step.done is True


def test_reward_binary(tmp_path):
    _slates().to_parquet(tmp_path / "slates.parquet", index=False)
    env = FinnNoSlateEnv(tmp_path, slate_size=25, feature_dim=8, seed=0)
    for seed in range(10):
        env.reset(seed=seed)
        step = env.step(np.arange(25))
        assert step.reward in (0.0, 1.0)


def test_native_fallback_warns(tmp_path):
    _slates().to_parquet(tmp_path / "slates.parquet", index=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        FinnNoSlateEnv(tmp_path, feature_dim=8, feature_source="native", seed=0)
    assert any(
        issubclass(x.category, UserWarning) and "hashed" in str(x.message).lower()
        for x in w
    )


def test_reward_one_when_positive_selected(tmp_path):
    _slates().to_parquet(tmp_path / "slates.parquet", index=False)
    env = FinnNoSlateEnv(tmp_path, slate_size=1, feature_dim=8, seed=0)
    obs = env.reset(seed=0)
    positive_id = int(env._current_row["item_id"])
    idx = int(np.where(obs.candidate_ids == positive_id)[0][0])
    step = env.step(np.array([idx]))
    assert step.reward == 1.0


def test_reward_zero_when_positive_not_selected(tmp_path):
    _slates().to_parquet(tmp_path / "slates.parquet", index=False)
    env = FinnNoSlateEnv(tmp_path, slate_size=1, feature_dim=8, seed=0)
    obs = env.reset(seed=0)
    positive_id = int(env._current_row["item_id"])
    non_positive = np.where(obs.candidate_ids != positive_id)[0]
    step = env.step(np.array([non_positive[0]]))
    assert step.reward == 0.0
