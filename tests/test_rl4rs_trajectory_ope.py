from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from rl_recsys.evaluation.behavior_policy import BehaviorPolicy
from rl_recsys.evaluation.ope_trajectory import LoggedTrajectoryStep


def _fixture_b_parquet(tmp_path: Path) -> Path:
    rows = [
        {"session_id": 1, "sequence_id": 1, "user_state": [1.0, 0.0],
         "slate": [10, 11], "user_feedback": [1, 0],
         "item_features": [[0.0, 0.0], [1.0, 0.0]],
         "candidate_ids": [10, 11, 12],
         "candidate_features": [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]]},
        {"session_id": 1, "sequence_id": 2, "user_state": [1.0, 0.0],
         "slate": [11, 12], "user_feedback": [0, 1],
         "item_features": [[1.0, 0.0], [0.5, 0.5]],
         "candidate_ids": [10, 11, 12],
         "candidate_features": [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]]},
        {"session_id": 2, "sequence_id": 1, "user_state": [0.0, 1.0],
         "slate": [10, 12], "user_feedback": [0, 0],
         "item_features": [[0.0, 0.0], [0.5, 0.5]],
         "candidate_ids": [10, 11, 12],
         "candidate_features": [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]]},
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
