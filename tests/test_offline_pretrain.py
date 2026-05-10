from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pytest

from rl_recsys.agents import LinUCBAgent, RandomAgent
from rl_recsys.environments.base import RecObs
from rl_recsys.evaluation.ope_trajectory import LoggedTrajectoryStep


class _SpyAgent:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def select_slate(self, obs):  # pragma: no cover — pretrain doesn't call this
        return np.array([0], dtype=np.int64)

    def score_items(self, obs):  # pragma: no cover — pretrain doesn't call this
        return np.zeros(len(obs.candidate_features), dtype=np.float64)

    def update(self, obs, slate, reward, clicks, next_obs):
        self.calls.append({
            "obs_id": id(obs),
            "next_obs_id": id(next_obs),
            "slate": np.array(slate, copy=True),
            "reward": float(reward),
            "clicks": np.array(clicks, copy=True),
        })
        return {}


class _ListSource:
    def __init__(self, trajectories: list[list[LoggedTrajectoryStep]]) -> None:
        self._t = trajectories

    def iter_trajectories(
        self, *, max_trajectories=None, seed=None
    ) -> Iterator[list[LoggedTrajectoryStep]]:
        for t in self._t[: max_trajectories if max_trajectories else len(self._t)]:
            yield t


def _make_step(slate=(0, 1), clicks=(1, 0), reward=1.0, propensity=0.25):
    obs = RecObs(
        user_features=np.array([1.0, 0.0]),
        candidate_features=np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]]),
        candidate_ids=np.array([10, 11, 12], dtype=np.int64),
    )
    return LoggedTrajectoryStep(
        obs=obs,
        logged_action=np.array(slate, dtype=np.int64),
        logged_reward=reward,
        logged_clicks=np.array(clicks, dtype=np.int64),
        propensity=propensity,
    )


def test_pretrain_calls_update_per_step() -> None:
    from rl_recsys.training.offline_pretrain import pretrain_agent_on_logged

    s1 = [_make_step(slate=(0, 1), clicks=(1, 0)),
          _make_step(slate=(1, 2), clicks=(0, 1))]
    s2 = [_make_step(slate=(0, 2), clicks=(0, 0))]
    source = _ListSource([s1, s2])
    agent = _SpyAgent()

    metrics = pretrain_agent_on_logged(agent, source)

    assert len(agent.calls) == 3
    np.testing.assert_array_equal(agent.calls[0]["clicks"], np.array([1, 0]))
    np.testing.assert_array_equal(agent.calls[1]["clicks"], np.array([0, 1]))
    np.testing.assert_array_equal(agent.calls[2]["clicks"], np.array([0, 0]))
    assert agent.calls[0]["obs_id"] == agent.calls[0]["next_obs_id"]
    assert metrics["trajectories"] == 2
    assert metrics["total_steps"] == 3
    assert np.isfinite(metrics["seconds"])


def test_pretrain_changes_linucb_state() -> None:
    from rl_recsys.training.offline_pretrain import pretrain_agent_on_logged

    s1 = [_make_step(slate=(0, 1), clicks=(1, 0)),
          _make_step(slate=(1, 2), clicks=(0, 1))]
    source = _ListSource([s1])
    agent = LinUCBAgent(slate_size=2, user_dim=2, item_dim=2, alpha=1.0)
    a_before = agent._a_matrix.copy()
    b_before = agent._b_vector.copy()

    pretrain_agent_on_logged(agent, source)

    assert not np.allclose(agent._a_matrix, a_before)
    assert not np.allclose(agent._b_vector, b_before)


def test_pretrain_random_is_noop_safe() -> None:
    from rl_recsys.training.offline_pretrain import pretrain_agent_on_logged

    s1 = [_make_step(slate=(0, 1), clicks=(1, 0))]
    source = _ListSource([s1])
    agent = RandomAgent(slate_size=2, seed=0)

    metrics = pretrain_agent_on_logged(agent, source)
    assert metrics["total_steps"] == 1
    assert np.isfinite(metrics["seconds"])


def test_pretrain_raises_on_empty_source() -> None:
    from rl_recsys.training.offline_pretrain import pretrain_agent_on_logged

    source = _ListSource([])
    agent = _SpyAgent()
    with pytest.raises(ValueError, match="zero trajectories"):
        pretrain_agent_on_logged(agent, source)
