from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
import pytest

from rl_recsys.agents import LinUCBAgent, RandomAgent
from rl_recsys.environments.base import RecObs
from rl_recsys.evaluation.trajectory import (
    Session,
    TrajectoryEvaluation,
    TrajectoryStep,
    evaluate_trajectory_agent,
)


def _make_step(*, clicked_id: int, candidate_ids: np.ndarray) -> TrajectoryStep:
    cand = np.asarray(candidate_ids, dtype=np.int64)
    feature_dim = 4
    obs = RecObs(
        user_features=np.zeros(feature_dim, dtype=np.float64),
        candidate_features=np.zeros((len(cand), feature_dim), dtype=np.float64),
        candidate_ids=cand,
    )
    return TrajectoryStep(
        obs=obs,
        logged_slate=cand[:3].copy(),
        logged_clicked_id=clicked_id,
        logged_reward=1.0 if clicked_id != -1 else 0.0,
    )


@dataclass
class _StaticDataset:
    sessions: list[Session]

    def iter_sessions(
        self, *, max_sessions: int | None = None, seed: int | None = None
    ) -> Iterator[Session]:
        out = self.sessions if max_sessions is None else self.sessions[:max_sessions]
        for s in out:
            yield s


class _DeterministicSlateAgent:
    """Always returns slate = [0, 1, 2] — first 3 candidates."""

    def __init__(self, slate_size: int = 3) -> None:
        self._slate_size = slate_size

    def select_slate(self, obs: RecObs) -> np.ndarray:
        return np.arange(self._slate_size, dtype=np.int64)

    def update(
        self,
        obs: RecObs,
        slate: np.ndarray,
        reward: float,
        clicks: np.ndarray,
        next_obs: RecObs,
    ) -> dict[str, float]:
        return {}


def test_replay_reward_when_slate_covers_logged_click() -> None:
    step = _make_step(clicked_id=11, candidate_ids=np.array([10, 11, 12, 13]))
    session = Session(session_id=1, steps=[step])
    dataset = _StaticDataset(sessions=[session])
    agent = _DeterministicSlateAgent(slate_size=3)

    result = evaluate_trajectory_agent(
        dataset, agent, agent_name="det", max_sessions=1, seed=0
    )

    assert result.avg_session_reward == pytest.approx(1.0)
    assert result.avg_session_hit_rate == pytest.approx(1.0)


def test_replay_reward_zero_when_slate_misses_logged_click() -> None:
    step = _make_step(clicked_id=13, candidate_ids=np.array([10, 11, 12, 13]))
    session = Session(session_id=1, steps=[step])
    dataset = _StaticDataset(sessions=[session])
    agent = _DeterministicSlateAgent(slate_size=3)

    result = evaluate_trajectory_agent(
        dataset, agent, agent_name="det", max_sessions=1, seed=0
    )

    assert result.avg_session_reward == pytest.approx(0.0)
    assert result.avg_session_hit_rate == pytest.approx(0.0)


def test_evaluate_trajectory_agent_aggregates_per_session() -> None:
    sessions = []
    for sid in (1, 2):
        steps = [
            _make_step(clicked_id=11, candidate_ids=np.array([10, 11, 12, 13]))
            for _ in range(3)
        ]
        sessions.append(Session(session_id=sid, steps=steps))
    dataset = _StaticDataset(sessions=sessions)
    agent = _DeterministicSlateAgent(slate_size=3)

    result = evaluate_trajectory_agent(
        dataset, agent, agent_name="det", max_sessions=2, seed=0, gamma=0.95
    )

    assert isinstance(result, TrajectoryEvaluation)
    assert result.sessions == 2
    assert result.total_steps == 6
    assert result.avg_session_reward == pytest.approx(3.0)
    assert result.avg_session_length == pytest.approx(3.0)
    assert result.avg_discounted_return == pytest.approx(1 + 0.95 + 0.95 ** 2)
    assert result.avg_session_hit_rate == pytest.approx(1.0)


def test_evaluate_trajectory_agent_handles_uncovered_steps() -> None:
    cands = np.array([10, 11, 12, 13])
    session = Session(
        session_id=1,
        steps=[
            _make_step(clicked_id=11, candidate_ids=cands),  # covered
            _make_step(clicked_id=13, candidate_ids=cands),  # missed
            _make_step(clicked_id=10, candidate_ids=cands),  # covered
            _make_step(clicked_id=13, candidate_ids=cands),  # missed
        ],
    )
    dataset = _StaticDataset(sessions=[session])
    agent = _DeterministicSlateAgent(slate_size=3)

    result = evaluate_trajectory_agent(
        dataset, agent, agent_name="det", max_sessions=1, seed=0
    )

    assert result.avg_session_reward == pytest.approx(2.0)
    assert result.avg_session_length == pytest.approx(4.0)
    assert result.total_steps == 4


def test_evaluate_trajectory_agent_does_not_mutate_agent_state() -> None:
    cands = np.array([10, 11, 12, 13])
    feature_dim = 4
    obs = RecObs(
        user_features=np.ones(feature_dim, dtype=np.float64),
        candidate_features=np.eye(4, feature_dim, dtype=np.float64),
        candidate_ids=cands,
    )
    step = TrajectoryStep(
        obs=obs,
        logged_slate=cands[:3].copy(),
        logged_clicked_id=11,
        logged_reward=1.0,
    )
    session = Session(session_id=1, steps=[step] * 5)
    dataset = _StaticDataset(sessions=[session])
    agent = LinUCBAgent(slate_size=3, user_dim=feature_dim, item_dim=feature_dim, alpha=1.0)
    a_before = agent._a_matrix.copy()
    b_before = agent._b_vector.copy()

    evaluate_trajectory_agent(dataset, agent, agent_name="linucb", max_sessions=1, seed=0)

    assert np.array_equal(agent._a_matrix, a_before)
    assert np.array_equal(agent._b_vector, b_before)


def test_replay_reward_zero_when_no_logged_click() -> None:
    # logged_clicked_id == -1 sentinel: no click was logged, reward must be 0
    # regardless of what the agent picks.
    cand = np.array([10, 11, 12, 13])
    feature_dim = 4
    obs = RecObs(
        user_features=np.zeros(feature_dim, dtype=np.float64),
        candidate_features=np.zeros((len(cand), feature_dim), dtype=np.float64),
        candidate_ids=cand,
    )
    step = TrajectoryStep(
        obs=obs,
        logged_slate=cand[:3].copy(),
        logged_clicked_id=-1,
        logged_reward=0.0,
    )
    session = Session(session_id=1, steps=[step])
    dataset = _StaticDataset(sessions=[session])
    agent = _DeterministicSlateAgent(slate_size=3)

    result = evaluate_trajectory_agent(
        dataset, agent, agent_name="det", max_sessions=1, seed=0
    )

    assert result.avg_session_reward == pytest.approx(0.0)
    assert result.avg_session_hit_rate == pytest.approx(0.0)


from rl_recsys.data.loaders.finn_no_slate_trajectory import FinnNoSlateTrajectoryLoader


def test_finn_no_slate_loader_emits_sessions(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "request_id": [0, 1, 2, 3, 4, 5, 6],
            "user_id": [10, 10, 10, 11, 11, 12, 12],
            "clicks": [0, 2, 1, 3, 0, 4, 2],
            "slate": [
                [100, 101, 102, 103, 104],
                [200, 201, 202, 203, 204],
                [300, 301, 302, 303, 304],
                [400, 401, 402, 403, 404],
                [500, 501, 502, 503, 504],
                [600, 601, 602, 603, 604],
                [700, 701, 702, 703, 704],
            ],
        }
    )
    parquet_path = tmp_path / "slates.parquet"
    df.to_parquet(parquet_path, index=False)

    loader = FinnNoSlateTrajectoryLoader(
        parquet_path,
        num_candidates=8,
        feature_dim=4,
        slate_size=3,
        seed=0,
    )
    sessions = list(loader.iter_sessions())

    assert len(sessions) == 3
    sessions_by_id = {s.session_id: s for s in sessions}
    assert sessions_by_id[10].steps[0].logged_clicked_id == 100
    assert sessions_by_id[10].steps[1].logged_clicked_id == 202
    assert sessions_by_id[10].steps[2].logged_clicked_id == 301
    assert len(sessions_by_id[11].steps) == 2
    assert sessions_by_id[11].steps[0].logged_clicked_id == 403
    assert len(sessions_by_id[12].steps) == 2

    for session in sessions:
        for step in session.steps:
            assert step.obs.candidate_ids.shape == (8,)
            assert step.obs.candidate_features.shape == (8, 4)
            assert step.obs.user_features.shape == (4,)
            assert set(step.logged_slate.tolist()).issubset(
                step.obs.candidate_ids.tolist()
            )

    capped = list(loader.iter_sessions(max_sessions=2))
    assert len(capped) == 2


def test_finn_no_slate_loader_rejects_small_num_candidates(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "request_id": [0],
            "user_id": [10],
            "clicks": [0],
            "slate": [[100, 101, 102, 103, 104]],
        }
    )
    parquet_path = tmp_path / "slates.parquet"
    df.to_parquet(parquet_path, index=False)

    with pytest.raises(ValueError, match="num_candidates"):
        FinnNoSlateTrajectoryLoader(
            parquet_path,
            num_candidates=3,
            feature_dim=4,
            slate_size=3,
            seed=0,
        )
