from __future__ import annotations

import numpy as np
import pytest

from rl_recsys.environments.base import RecObs
from rl_recsys.evaluation.ope_trajectory import LoggedTrajectoryStep


def _make_step(num_candidates=8, slate_size=3):
    obs = RecObs(
        user_features=np.array([1.0, 0.0, 0.0, 0.0]),
        candidate_features=np.eye(num_candidates, 3),
        candidate_ids=np.arange(num_candidates, dtype=np.int64),
    )
    return LoggedTrajectoryStep(
        obs=obs,
        logged_action=np.array([0, 1, 2], dtype=np.int64)[:slate_size],
        logged_reward=1.0,
        logged_clicks=np.array([1, 0, 1], dtype=np.int64)[:slate_size],
        propensity=0.1,
    )


class _StubSource:
    def __init__(self, n: int = 4):
        self._n = n

    def iter_trajectories(self, *, max_trajectories=None, seed=0):
        for _ in range(self._n):
            yield [_make_step(), _make_step()]


def test_neural_linear_train_offline_completes_on_tiny_dataset():
    from rl_recsys.agents.neural_linear import NeuralLinearAgent

    agent = NeuralLinearAgent(
        slate_size=3, user_dim=4, item_dim=3,
        hidden_dim=8, embedding_dim=4, mlp_epochs=1, alpha=1.0,
        device="cpu",
    )
    metrics = agent.train_offline(_StubSource(n=4), seed=0)
    assert "mlp_loss" in metrics


def test_neural_linear_score_shape():
    from rl_recsys.agents.neural_linear import NeuralLinearAgent

    agent = NeuralLinearAgent(
        slate_size=3, user_dim=4, item_dim=3,
        hidden_dim=8, embedding_dim=4, mlp_epochs=1, alpha=1.0,
        device="cpu",
    )
    agent.train_offline(_StubSource(n=2), seed=0)
    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.eye(7, 3),
        candidate_ids=np.arange(7, dtype=np.int64),
    )
    assert agent.score_items(obs).shape == (7,)
