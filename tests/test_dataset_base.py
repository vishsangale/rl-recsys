from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rl_recsys.environments.base import RecObs, RecStep
from rl_recsys.environments.dataset_base import BanditDatasetEnv, SessionDatasetEnv


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
    env.reset(seed=7)
    step = env.step(np.arange(10))
    assert step.clicks.sum() >= 1.0


def test_bandit_invalid_feature_source_raises():
    with pytest.raises(ValueError, match="feature_source"):
        _SimpleBanditEnv(_interactions(), num_candidates=10, feature_dim=4, feature_source="bad")


def test_bandit_candidate_fallback_when_user_has_all_items():
    # User 0 has interacted with all 20 items; fallback relaxes exclusion pool
    df = _interactions(n_users=1, n_items=20)
    env = _SimpleBanditEnv(df, slate_size=1, num_candidates=5, feature_dim=4, feature_source="hashed", seed=0)
    obs = env.reset(seed=0)
    assert obs.candidate_ids.shape == (5,)


class _SimpleSessionEnv(SessionDatasetEnv):
    def _compute_reward(self, row, clicks):
        return float(clicks.sum())


def _sessions(n_sessions=2, n_steps=3, slate_size=3) -> dict[int, pd.DataFrame]:
    rng = np.random.default_rng(1)
    result = {}
    for sid in range(n_sessions):
        rows = []
        for _ in range(n_steps):
            items = rng.integers(0, 100, size=slate_size).tolist()
            feats = rng.standard_normal((slate_size, 4)).tolist()
            clicks_vec = [0] * slate_size
            clicks_vec[int(rng.integers(0, slate_size))] = 1
            rows.append({
                "slate": [int(x) for x in items],
                "item_features": feats,
                "clicks": clicks_vec,
                "user_state": rng.standard_normal(4).tolist(),
            })
        result[sid] = pd.DataFrame(rows)
    return result


def test_session_reset_obs_shape():
    env = _SimpleSessionEnv(_sessions(), slate_size=3, num_candidates=3, feature_dim=4, feature_source="hashed", seed=0)
    obs = env.reset(seed=0)
    assert obs.user_features.shape == (4,)
    assert obs.candidate_features.shape == (3, 4)
    assert obs.candidate_ids.shape == (3,)


def test_session_done_false_mid_session():
    env = _SimpleSessionEnv(_sessions(), slate_size=3, num_candidates=3, feature_dim=4, feature_source="hashed", seed=0)
    env.reset(seed=0)
    step = env.step(np.array([0, 1, 2]))
    assert step.done is False


def test_session_done_true_at_end():
    env = _SimpleSessionEnv(_sessions(n_steps=3), slate_size=3, num_candidates=3, feature_dim=4, feature_source="hashed", seed=0)
    env.reset(seed=0)
    env.step(np.array([0, 1, 2]))
    env.step(np.array([0, 1, 2]))
    step = env.step(np.array([0, 1, 2]))
    assert step.done is True


def test_session_step_before_reset_raises():
    env = _SimpleSessionEnv(_sessions(), slate_size=3, num_candidates=3, feature_dim=4, feature_source="hashed", seed=0)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.array([0, 1, 2]))


def test_session_empty_dict_raises():
    with pytest.raises(ValueError, match="empty"):
        _SimpleSessionEnv({}, slate_size=1, num_candidates=1, feature_dim=4, feature_source="hashed")


def test_session_next_obs_shape_mid_session():
    env = _SimpleSessionEnv(_sessions(n_steps=3), slate_size=3, num_candidates=3, feature_dim=4, feature_source="hashed", seed=0)
    env.reset(seed=0)
    step = env.step(np.array([0, 1, 2]))
    assert step.obs.user_features.shape == (4,)
    assert step.obs.candidate_features.shape == (3, 4)


def test_session_clicks_match_logged_by_index():
    # Verify click derivation is index-based, not item-ID-based
    rng = np.random.default_rng(0)
    sessions = {
        0: pd.DataFrame([
            {"slate": [10, 20, 30], "item_features": rng.standard_normal((3, 4)).tolist(),
             "clicks": [0, 0, 1], "user_state": rng.standard_normal(4).tolist()},
        ])
    }
    env = _SimpleSessionEnv(sessions, slate_size=1, num_candidates=3, feature_dim=4, feature_source="hashed", seed=0)
    env.reset(seed=0)
    step = env.step(np.array([2]))  # pick index 2 (item 30, the clicked one)
    assert step.clicks[0] == 1.0
    env.reset(seed=0)
    step = env.step(np.array([0]))  # pick index 0 (item 10, not clicked)
    assert step.clicks[0] == 0.0


def test_session_cursor_resets_on_new_episode():
    env = _SimpleSessionEnv(_sessions(n_steps=2), slate_size=3, num_candidates=3, feature_dim=4, feature_source="hashed", seed=0)
    env.reset(seed=0)
    env.step(np.array([0, 1, 2]))
    env.step(np.array([0, 1, 2]))  # done=True
    env.reset(seed=0)             # new episode: cursor must be 0
    step = env.step(np.array([0, 1, 2]))
    assert step.done is False     # mid-session again, not immediately done


def test_session_single_step_done_immediately():
    rng = np.random.default_rng(0)
    sessions = {
        0: pd.DataFrame([
            {"slate": [1, 2, 3], "item_features": rng.standard_normal((3, 4)).tolist(),
             "clicks": [1, 0, 0], "user_state": rng.standard_normal(4).tolist()},
        ])
    }
    env = _SimpleSessionEnv(sessions, slate_size=3, num_candidates=3, feature_dim=4, feature_source="hashed", seed=0)
    env.reset(seed=0)
    step = env.step(np.array([0, 1, 2]))
    assert step.done is True
