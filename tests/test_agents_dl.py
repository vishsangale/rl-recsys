from __future__ import annotations

import numpy as np
import pytest

from rl_recsys.environments.base import HistoryStep, RecObs
from rl_recsys.evaluation.ope_trajectory import LoggedTrajectoryStep


def _stub_step(history=()):
    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.eye(8, 3),
        candidate_ids=np.arange(8, dtype=np.int64),
        history=history,
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
        h: tuple = ()
        traj = []
        for _ in range(3):
            step = _stub_step(history=h)
            traj.append(step)
            h = h + (HistoryStep(step.logged_action, step.logged_clicks),)
        yield traj


def test_sasrec_forward_shape():
    from rl_recsys.agents.sasrec import SASRecAgent

    agent = SASRecAgent(
        slate_size=3, num_candidates=8, item_dim=3,
        hidden_dim=8, n_heads=2, n_blocks=1, max_history_len=5,
        epochs=1, device="cpu",
    )
    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.eye(8, 3),
        candidate_ids=np.arange(8, dtype=np.int64),
    )
    assert agent.score_items(obs).shape == (8,)


def test_sasrec_handles_empty_history_via_sentinel():
    from rl_recsys.agents.sasrec import SASRecAgent

    agent = SASRecAgent(
        slate_size=3, num_candidates=8, item_dim=3,
        hidden_dim=8, n_heads=2, n_blocks=1, max_history_len=5,
        epochs=1, device="cpu",
    )
    obs_empty = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.eye(8, 3),
        candidate_ids=np.arange(8, dtype=np.int64),
        history=(),
    )
    scores = agent.score_items(obs_empty)
    assert scores.shape == (8,)


def test_sasrec_train_offline_runs():
    from rl_recsys.agents.sasrec import SASRecAgent

    agent = SASRecAgent(
        slate_size=3, num_candidates=8, item_dim=3,
        hidden_dim=8, n_heads=2, n_blocks=1, max_history_len=5,
        epochs=1, device="cpu",
    )
    metrics = agent.train_offline(_StubSource(), seed=0)
    assert "loss" in metrics or "epochs" in metrics
