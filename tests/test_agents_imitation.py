from __future__ import annotations

import numpy as np
import pytest

from rl_recsys.environments.base import RecObs
from rl_recsys.evaluation.ope_trajectory import LoggedTrajectoryStep


def _stub_step(num_candidates=8):
    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.eye(num_candidates, 3),
        candidate_ids=np.arange(num_candidates, dtype=np.int64),
    )
    return LoggedTrajectoryStep(
        obs=obs,
        logged_action=np.array([0, 1, 2], dtype=np.int64),
        logged_reward=1.0,
        logged_clicks=np.array([1, 0, 1], dtype=np.int64),
        propensity=0.1,
    )


class _StubSource:
    def iter_trajectories(self, *, max_trajectories=None, seed=0):
        yield [_stub_step()] * 2


def test_bc_score_items_shape():
    from rl_recsys.agents.bc import BCAgent
    from rl_recsys.evaluation.behavior_policy import BehaviorPolicy

    policy = BehaviorPolicy(
        user_dim=4, item_dim=3, slate_size=3, num_items=8, device="cpu",
    )
    agent = BCAgent(
        slate_size=3, behavior_policy=policy,
        candidate_features=np.eye(8, 3),
    )
    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.eye(8, 3),
        candidate_ids=np.arange(8, dtype=np.int64),
    )
    assert agent.score_items(obs).shape == (8,)


def test_bc_train_offline_with_prefit_policy_is_noop():
    from rl_recsys.agents.bc import BCAgent
    from rl_recsys.evaluation.behavior_policy import BehaviorPolicy

    policy = BehaviorPolicy(
        user_dim=4, item_dim=3, slate_size=3, num_items=8, device="cpu",
    )
    agent = BCAgent(
        slate_size=3, behavior_policy=policy,
        candidate_features=np.eye(8, 3),
    )
    metrics = agent.train_offline(_StubSource(), seed=0)
    assert metrics == {}


def test_bc_raises_when_score_called_without_policy():
    from rl_recsys.agents.bc import BCAgent

    agent = BCAgent(
        slate_size=3, behavior_policy=None,
        candidate_features=np.eye(8, 3),
    )
    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.eye(8, 3),
        candidate_ids=np.arange(8, dtype=np.int64),
    )
    with pytest.raises(RuntimeError, match="behavior_policy"):
        agent.score_items(obs)


def test_gbdt_train_offline_returns_metrics():
    from rl_recsys.agents.gbdt import GBDTAgent

    agent = GBDTAgent(
        slate_size=3, candidate_features=np.eye(8, 3),
        n_estimators=10, max_depth=3,
    )
    metrics = agent.train_offline(_StubSource(), seed=0)
    assert "n_train_rows" in metrics


def test_gbdt_score_shape():
    from rl_recsys.agents.gbdt import GBDTAgent

    agent = GBDTAgent(
        slate_size=3, candidate_features=np.eye(8, 3),
        n_estimators=10, max_depth=3,
    )
    agent.train_offline(_StubSource(), seed=0)
    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.eye(8, 3),
        candidate_ids=np.arange(8, dtype=np.int64),
    )
    assert agent.score_items(obs).shape == (8,)
