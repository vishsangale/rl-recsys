from __future__ import annotations

import numpy as np
import pytest

from rl_recsys.agents.most_popular import MostPopularAgent
from rl_recsys.environments.base import RecObs
from rl_recsys.evaluation.ope_trajectory import LoggedTrajectoryStep


def _make_obs(num_candidates: int = 5):
    return RecObs(
        user_features=np.zeros(4),
        candidate_features=np.zeros((num_candidates, 3)),
        candidate_ids=np.arange(num_candidates, dtype=np.int64),
    )


def _make_step(slate: list[int], clicks: list[int]):
    obs = _make_obs(num_candidates=5)
    return LoggedTrajectoryStep(
        obs=obs,
        logged_action=np.array(slate, dtype=np.int64),
        logged_reward=float(sum(clicks)),
        logged_clicks=np.array(clicks, dtype=np.int64),
        propensity=0.1,
    )


class _StubSource:
    def __init__(self, trajs):
        self._trajs = trajs

    def iter_trajectories(self, *, max_trajectories=None, seed=0):
        yield from self._trajs


def test_most_popular_train_offline_counts_clicks_per_item():
    source = _StubSource([
        [_make_step([0, 1, 2], [1, 0, 1])],
        [_make_step([1, 2, 3], [0, 1, 0])],
    ])
    agent = MostPopularAgent(slate_size=2, num_candidates=5)
    agent.train_offline(source)
    np.testing.assert_array_equal(
        agent._clicks_per_item, np.array([1, 0, 2, 0, 0], dtype=np.float64),
    )


def test_most_popular_select_slate_picks_top_k():
    agent = MostPopularAgent(slate_size=2, num_candidates=5)
    agent._clicks_per_item = np.array([1, 5, 0, 3, 2], dtype=np.float64)
    slate = agent.select_slate(_make_obs())
    assert set(slate.tolist()) == {1, 3}


def test_most_popular_select_slate_raises_when_slate_too_large():
    agent = MostPopularAgent(slate_size=10, num_candidates=5)
    with pytest.raises(ValueError, match="exceeds num_candidates"):
        agent.select_slate(_make_obs(num_candidates=5))


def test_logged_replay_returns_logged_slate():
    from rl_recsys.agents.logged_replay import LoggedReplayAgent

    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.zeros((10, 3)),
        candidate_ids=np.arange(10, dtype=np.int64),
        logged_action=np.array([4, 7, 1], dtype=np.int64),
    )
    agent = LoggedReplayAgent(slate_size=3)
    np.testing.assert_array_equal(agent.select_slate(obs), [4, 7, 1])


def test_logged_replay_raises_without_logged_action():
    from rl_recsys.agents.logged_replay import LoggedReplayAgent

    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.zeros((10, 3)),
        candidate_ids=np.arange(10, dtype=np.int64),
    )
    agent = LoggedReplayAgent(slate_size=3)
    with pytest.raises(ValueError, match="replay-mode source"):
        agent.select_slate(obs)


def test_logged_replay_score_items_peaks_on_logged_slate():
    from rl_recsys.agents.logged_replay import LoggedReplayAgent

    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.zeros((6, 3)),
        candidate_ids=np.arange(6, dtype=np.int64),
        logged_action=np.array([2, 4], dtype=np.int64),
    )
    agent = LoggedReplayAgent(slate_size=2)
    scores = agent.score_items(obs)
    assert scores.shape == (6,)
    assert scores[2] > scores[0] and scores[4] > scores[0]
