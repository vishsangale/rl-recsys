from pathlib import Path

import numpy as np
import pandas as pd

from rl_recsys.evaluation.behavior_policy import BehaviorPolicy
from rl_recsys.evaluation.ope_trajectory import LoggedTrajectoryStep


def _fixture_b_parquet(tmp_path: Path) -> Path:
    rows = [
        {"session_id": 1, "sequence_id": 1, "user_state": [1.0, 0.0],
         "slate": [10, 11], "user_feedback": [1, 0],
         "item_features": [[0.0, 0.0], [1.0, 0.0]]},
        {"session_id": 1, "sequence_id": 2, "user_state": [1.0, 0.0],
         "slate": [11, 12], "user_feedback": [0, 1],
         "item_features": [[1.0, 0.0], [0.5, 0.5]]},
        {"session_id": 2, "sequence_id": 1, "user_state": [0.0, 1.0],
         "slate": [10, 12], "user_feedback": [0, 0],
         "item_features": [[0.0, 0.0], [0.5, 0.5]]},
    ]
    p = tmp_path / "sessions_b.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    return p


def test_loader_emits_trajectories_grouped_by_session(tmp_path: Path) -> None:
    from rl_recsys.data.loaders.rl4rs_trajectory_ope import (
        RL4RSTrajectoryOPESource,
    )
    parquet = _fixture_b_parquet(tmp_path)

    model = BehaviorPolicy(
        user_dim=2, item_dim=2, slate_size=2, num_items=3,
        hidden_dim=4, seed=0,
    )

    source = RL4RSTrajectoryOPESource(
        parquet_path=parquet, behavior_policy=model, slate_size=2,
    )
    trajectories = list(source.iter_trajectories(max_trajectories=10, seed=0))

    assert len(trajectories) == 2  # 2 unique session_ids
    s1 = next(t for t in trajectories if len(t) == 2)
    s2 = next(t for t in trajectories if len(t) == 1)

    assert all(isinstance(s, LoggedTrajectoryStep) for s in s1)
    # logged_action shape == (slate_size,)
    assert s1[0].logged_action.shape == (2,)
    np.testing.assert_array_equal(s1[0].logged_action, np.array([0, 1]))
    np.testing.assert_array_equal(s1[1].logged_action, np.array([1, 2]))
    np.testing.assert_array_equal(s2[0].logged_action, np.array([0, 2]))
    # rewards = sum(user_feedback)
    assert s1[0].logged_reward == 1.0
    assert s1[1].logged_reward == 1.0
    assert s2[0].logged_reward == 0.0
    # propensity in (0, 1]
    for t in trajectories:
        for step in t:
            assert 0.0 < step.propensity <= 1.0


def test_loader_emits_logged_clicks(tmp_path: Path) -> None:
    from rl_recsys.data.loaders.rl4rs_trajectory_ope import (
        RL4RSTrajectoryOPESource,
    )
    parquet = _fixture_b_parquet(tmp_path)

    model = BehaviorPolicy(
        user_dim=2, item_dim=2, slate_size=2, num_items=3,
        hidden_dim=4, seed=0,
    )
    source = RL4RSTrajectoryOPESource(
        parquet_path=parquet, behavior_policy=model, slate_size=2,
    )
    trajectories = list(source.iter_trajectories(max_trajectories=10, seed=0))

    s1 = next(t for t in trajectories if len(t) == 2)
    s2 = next(t for t in trajectories if len(t) == 1)

    np.testing.assert_array_equal(s1[0].logged_clicks, np.array([1, 0]))
    np.testing.assert_array_equal(s1[1].logged_clicks, np.array([0, 1]))
    np.testing.assert_array_equal(s2[0].logged_clicks, np.array([0, 0]))
    assert s1[0].logged_clicks.dtype == np.int64


def test_end_to_end_seq_dr_on_synthetic_b_fixture(tmp_path: Path) -> None:
    # Write a tiny multi-step parquet, fit BehaviorPolicy, build the loader,
    # run evaluate_trajectory_ope_agent, assert avg_seq_dr_value is finite
    # and ess is reported.
    rng = np.random.default_rng(0)
    rows = []
    for sid in range(40):
        for seq in range(2):
            rows.append({
                "session_id": sid,
                "sequence_id": seq,
                "user_state": rng.standard_normal(2).tolist(),
                "slate": [rng.integers(0, 3), rng.integers(0, 3)],
                "user_feedback": [int(rng.integers(0, 2)), int(rng.integers(0, 2))],
                "item_features": [[0.0, 0.0], [1.0, 0.0]],
            })
    parquet = tmp_path / "sessions_b.parquet"
    pd.DataFrame(rows).to_parquet(parquet, index=False)

    from rl_recsys.evaluation.behavior_policy import (
        fit_behavior_policy_with_calibration,
    )
    from rl_recsys.data.loaders.rl4rs_trajectory_ope import (
        RL4RSTrajectoryOPESource,
    )
    from rl_recsys.evaluation import evaluate_trajectory_ope_agent

    model = fit_behavior_policy_with_calibration(
        parquet, user_dim=2, item_dim=2, slate_size=2, num_items=3,
        epochs=5, batch_size=16, seed=0, nll_threshold=10.0,
    )

    class _FlatScoreAgent:
        def select_slate(self, obs):
            return np.array([0, 1], dtype=np.int64)
        def score_items(self, obs):
            return np.zeros(len(obs.candidate_features), dtype=np.float64)
        def update(self, *a, **kw):
            return {}

    source = RL4RSTrajectoryOPESource(
        parquet_path=parquet, behavior_policy=model, slate_size=2,
    )
    result = evaluate_trajectory_ope_agent(
        source, _FlatScoreAgent(), agent_name="flat",
        max_trajectories=40, seed=0, gamma=0.9, temperature=1.0,
    )

    assert np.isfinite(result.avg_seq_dr_value)
    assert np.isfinite(result.avg_logged_discounted_return)
    assert result.ess > 0.0
    assert result.trajectories > 0


def test_loader_session_filter_yields_only_filtered_sessions(tmp_path: Path) -> None:
    from rl_recsys.data.loaders.rl4rs_trajectory_ope import (
        RL4RSTrajectoryOPESource,
    )
    parquet = _fixture_b_parquet(tmp_path)
    model = BehaviorPolicy(
        user_dim=2, item_dim=2, slate_size=2, num_items=3,
        hidden_dim=4, seed=0,
    )

    # Filter to only session 1
    source = RL4RSTrajectoryOPESource(
        parquet_path=parquet, behavior_policy=model, slate_size=2,
        session_filter={1},
    )
    trajectories = list(source.iter_trajectories(max_trajectories=10, seed=0))

    # Only session 1 emitted (it has 2 steps)
    assert len(trajectories) == 1
    assert len(trajectories[0]) == 2
    # Universe still includes all items from the full parquet (10, 11, 12)
    assert source._candidate_ids.tolist() == [10, 11, 12]


def test_loader_empty_session_filter_raises_on_iter(tmp_path: Path) -> None:
    from rl_recsys.data.loaders.rl4rs_trajectory_ope import (
        RL4RSTrajectoryOPESource,
    )
    import pytest
    parquet = _fixture_b_parquet(tmp_path)
    model = BehaviorPolicy(
        user_dim=2, item_dim=2, slate_size=2, num_items=3,
        hidden_dim=4, seed=0,
    )
    # Filter excludes everything
    source = RL4RSTrajectoryOPESource(
        parquet_path=parquet, behavior_policy=model, slate_size=2,
        session_filter={9999},
    )
    with pytest.raises(ValueError, match="session_filter"):
        list(source.iter_trajectories(max_trajectories=10, seed=0))


def test_pretrained_linucb_diverges_from_fresh_linucb(tmp_path: Path) -> None:
    from rl_recsys.evaluation.behavior_policy import (
        fit_behavior_policy_with_calibration,
    )
    from rl_recsys.data.loaders.rl4rs_trajectory_ope import (
        RL4RSTrajectoryOPESource,
    )
    from rl_recsys.evaluation import evaluate_trajectory_ope_agent
    from rl_recsys.agents import LinUCBAgent
    from rl_recsys.training import (
        pretrain_agent_on_logged, split_session_ids,
    )

    rng = np.random.default_rng(0)
    rows = []
    for sid in range(60):
        for seq in range(2):
            rows.append({
                "session_id": sid,
                "sequence_id": seq,
                "user_state": rng.standard_normal(2).tolist(),
                "slate": [int(rng.integers(0, 3)), int(rng.integers(0, 3))],
                "user_feedback": [int(rng.integers(0, 2)), int(rng.integers(0, 2))],
                "item_features": [[0.0, 0.0], [1.0, 0.0]],
            })
    parquet = tmp_path / "sessions_b.parquet"
    pd.DataFrame(rows).to_parquet(parquet, index=False)

    model = fit_behavior_policy_with_calibration(
        parquet, user_dim=2, item_dim=2, slate_size=2, num_items=3,
        epochs=5, batch_size=16, seed=0, nll_threshold=10.0,
    )

    train_ids, eval_ids = split_session_ids(parquet, train_fraction=0.5, seed=42)

    train_source = RL4RSTrajectoryOPESource(
        parquet_path=parquet, behavior_policy=model, slate_size=2,
        session_filter=train_ids,
    )
    eval_source_fresh = RL4RSTrajectoryOPESource(
        parquet_path=parquet, behavior_policy=model, slate_size=2,
        session_filter=eval_ids,
    )
    eval_source_pretrained = RL4RSTrajectoryOPESource(
        parquet_path=parquet, behavior_policy=model, slate_size=2,
        session_filter=eval_ids,
    )

    fresh = LinUCBAgent(slate_size=2, user_dim=2, item_dim=2, alpha=1.0)
    pretrained = LinUCBAgent(slate_size=2, user_dim=2, item_dim=2, alpha=1.0)
    pretrain_agent_on_logged(pretrained, train_source)

    fresh_result = evaluate_trajectory_ope_agent(
        eval_source_fresh, fresh, agent_name="fresh",
        max_trajectories=60, seed=0, gamma=0.95, temperature=1.0,
    )
    pretrained_result = evaluate_trajectory_ope_agent(
        eval_source_pretrained, pretrained, agent_name="pretrained",
        max_trajectories=60, seed=0, gamma=0.95, temperature=1.0,
    )

    assert np.isfinite(fresh_result.avg_seq_dr_value)
    assert np.isfinite(pretrained_result.avg_seq_dr_value)
    # The two values must differ — pretraining changed agent.score_items.
    assert (
        abs(fresh_result.avg_seq_dr_value - pretrained_result.avg_seq_dr_value)
        > 1e-6
    )
