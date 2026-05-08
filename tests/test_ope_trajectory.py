from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pytest

from rl_recsys.agents import LinUCBAgent
from rl_recsys.agents.random import RandomAgent
from rl_recsys.environments.base import RecObs
from rl_recsys.evaluation.ope_trajectory import (
    LoggedTrajectoryStep,
    TrajectoryOPEEvaluation,
    evaluate_trajectory_ope_agent,
    seq_dr_value,
)


def test_seq_dr_value_collapses_to_logged_return_when_target_equals_behavior() -> None:
    # target == behavior → w=1, W_t=1 → result = sum(γ^t r_t)
    rewards = np.array([1.0, 0.5, 2.0])
    probs = np.array([0.3, 0.4, 0.5])
    gamma = 0.9
    expected = sum(gamma ** t * r for t, r in enumerate(rewards))

    result = seq_dr_value(rewards, probs, probs, gamma=gamma)

    assert result == pytest.approx(expected)


def test_seq_dr_value_collapses_to_per_decision_is_when_baseline_zero() -> None:
    # reward_model = 0 → baseline cancels → result = sum(γ^t W_t r_t)
    rewards = np.array([1.0, 1.0, 1.0])
    target_probs = np.array([0.4, 0.5, 0.6])
    propensities = np.array([0.2, 0.5, 0.3])
    # ratios: [2.0, 1.0, 2.0]; W = [2.0, 2.0, 4.0]
    gamma = 0.9
    expected = (1.0 * 2.0 * 1.0) + (0.9 * 2.0 * 1.0) + (0.81 * 4.0 * 1.0)  # = 7.04

    result = seq_dr_value(
        rewards, target_probs, propensities,
        gamma=gamma, reward_model=lambda i: 0.0,
    )

    assert result == pytest.approx(expected)


def test_seq_dr_value_clips_extreme_ratios() -> None:
    # ratio[0] = 0.99/0.01 = 99 → clipped to 10
    # ratio[1] = 0.5/0.5 = 1
    # Use reward_model=lambda i: 0.0 so clipping is detectable.
    rewards = np.array([2.0, 0.0])
    target_probs = np.array([0.99, 0.5])
    propensities = np.array([0.01, 0.5])
    gamma = 1.0
    # Clipped: W = [10, 10]; V = 1*(10*2 + 0) + 1*(10*0 + 0) = 20
    clipped_expected = 20.0

    result = seq_dr_value(
        rewards, target_probs, propensities,
        gamma=gamma, reward_model=lambda i: 0.0,
    )

    assert result == pytest.approx(clipped_expected)
    # Unclipped would give W=[99, 99], V = 1*99*2 + 1*99*0 = 198 — must differ
    assert result != pytest.approx(198.0)


def test_seq_dr_value_uses_provided_reward_model() -> None:
    # reward_model returns i+1 → b = [1.0, 2.0]
    # target == behavior → W = [1, 1]
    # γ=0.5: V = 1*(1*(3-1) + 1) + 0.5*(1*(1-2) + 2) = 3 + 0.5 = 3.5
    rewards = np.array([3.0, 1.0])
    probs = np.array([0.5, 0.5])
    expected = 1.0 * (1 * (3.0 - 1.0) + 1.0) + 0.5 * (1 * (1.0 - 2.0) + 2.0)  # 3.5

    result = seq_dr_value(
        rewards, probs, probs,
        gamma=0.5, reward_model=lambda i: float(i + 1),
    )

    assert result == pytest.approx(expected)


def _make_obs(num_candidates: int = 4, feature_dim: int = 4) -> RecObs:
    return RecObs(
        user_features=np.zeros(feature_dim, dtype=np.float64),
        candidate_features=np.zeros((num_candidates, feature_dim), dtype=np.float64),
        candidate_ids=np.arange(num_candidates, dtype=np.int64),
    )


@dataclass
class _SyntheticTrajectorySource:
    trajectories: list[list[LoggedTrajectoryStep]]

    def iter_trajectories(
        self, *, max_trajectories: int | None = None, seed: int | None = None
    ) -> Iterator[list[LoggedTrajectoryStep]]:
        out = (
            self.trajectories
            if max_trajectories is None
            else self.trajectories[:max_trajectories]
        )
        for traj in out:
            yield traj


class _DetAgent:
    """Always picks slate=[0] — top action is candidate index 0."""

    def __init__(self, slate_size: int = 1) -> None:
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


def test_evaluate_trajectory_ope_agent_aggregates_per_trajectory() -> None:
    # _DetAgent picks index 0; logged_action=0 → target_prob=1.0 for non-Random agents.
    # Trajectory A (2 steps): rewards=[1.0, 0.0], propensity=[0.5, 0.5]
    #   w = [2.0, 2.0]; W = [2.0, 4.0]; b = mean([1, 0]) = 0.5
    #   V_A(γ=0.9) = 1*(2*(1-0.5)+0.5) + 0.9*(4*(0-0.5)+0.5) = 1.5 + 0.9*(-1.5) = 0.15
    # Trajectory B (1 step): rewards=[2.0], propensity=[1.0]
    #   w = [1.0]; W = [1.0]; b = 2.0; V_B = 1*(1*(2-2)+2) = 2.0
    # avg_seq_dr = (0.15 + 2.0) / 2 = 1.075
    # avg_logged_discounted_return:
    #   Logged_A(γ=0.9) = 1.0 + 0.9*0.0 = 1.0
    #   Logged_B = 2.0
    #   avg = 1.5
    obs = _make_obs(num_candidates=4)
    traj_a = [
        LoggedTrajectoryStep(obs=obs, logged_action=0, logged_reward=1.0, propensity=0.5),
        LoggedTrajectoryStep(obs=obs, logged_action=0, logged_reward=0.0, propensity=0.5),
    ]
    traj_b = [
        LoggedTrajectoryStep(obs=obs, logged_action=0, logged_reward=2.0, propensity=1.0),
    ]
    source = _SyntheticTrajectorySource(trajectories=[traj_a, traj_b])
    agent = _DetAgent(slate_size=1)

    result = evaluate_trajectory_ope_agent(
        source, agent, agent_name="det", max_trajectories=2, seed=0, gamma=0.9
    )

    assert isinstance(result, TrajectoryOPEEvaluation)
    assert result.trajectories == 2
    assert result.total_steps == 3
    assert result.avg_seq_dr_value == pytest.approx((0.15 + 2.0) / 2)
    assert result.avg_logged_discounted_return == pytest.approx((1.0 + 2.0) / 2)


def test_seq_dr_value_t1_collapses_to_bandit_dr_formula() -> None:
    # T=1 → V = w*(r-b) + b, the bandit per-sample DR formula.
    rewards = np.array([2.0])
    target_probs = np.array([0.6])
    propensities = np.array([0.3])
    # w = clip(2.0) = 2.0; W=[2.0]; b=[1.0]; γ^0=1
    # V = 2.0 * (2.0 - 1.0) + 1.0 = 3.0
    expected = 3.0

    result = seq_dr_value(
        rewards, target_probs, propensities,
        gamma=0.95, reward_model=lambda i: 1.0,
    )

    assert result == pytest.approx(expected)


def test_evaluate_trajectory_ope_agent_raises_on_nonpositive_max_trajectories() -> None:
    source = _SyntheticTrajectorySource(trajectories=[])
    agent = _DetAgent(slate_size=1)

    with pytest.raises(ValueError, match="max_trajectories must be positive"):
        evaluate_trajectory_ope_agent(
            source, agent, agent_name="det", max_trajectories=0, seed=0
        )
    with pytest.raises(ValueError, match="max_trajectories must be positive"):
        evaluate_trajectory_ope_agent(
            source, agent, agent_name="det", max_trajectories=-1, seed=0
        )


def test_evaluate_trajectory_ope_agent_raises_when_all_trajectories_empty() -> None:
    # Empty trajectories are skipped; if every one is empty the averages would
    # be undefined, so the evaluator must raise rather than return NaN.
    source = _SyntheticTrajectorySource(trajectories=[[], [], []])
    agent = _DetAgent(slate_size=1)

    with pytest.raises(ValueError, match="source produced zero trajectories"):
        evaluate_trajectory_ope_agent(
            source, agent, agent_name="det", max_trajectories=3, seed=0
        )


def test_evaluate_trajectory_ope_agent_uses_uniform_target_prob_for_random_agent() -> None:
    # RandomAgent branch of _target_probability returns 1/num_candidates.
    # 4 candidates → target_prob = 0.25 at every step.
    # 2 steps, rewards=[1.0, 0.0], propensity=[0.5, 0.5], reward_model=0:
    #   w = clip([0.5, 0.5]) = [0.5, 0.5]; W = [0.5, 0.25]
    #   γ=1: V = 1*(0.5*1 + 0) + 1*(0.25*0 + 0) = 0.5
    obs = _make_obs(num_candidates=4)
    traj = [
        LoggedTrajectoryStep(obs=obs, logged_action=0, logged_reward=1.0, propensity=0.5),
        LoggedTrajectoryStep(obs=obs, logged_action=0, logged_reward=0.0, propensity=0.5),
    ]
    source = _SyntheticTrajectorySource(trajectories=[traj])
    agent = RandomAgent(slate_size=1, seed=0)

    result = evaluate_trajectory_ope_agent(
        source, agent, agent_name="random",
        max_trajectories=1, seed=0, gamma=1.0, reward_model=lambda i: 0.0,
    )

    assert result.avg_seq_dr_value == pytest.approx(0.5)


def test_evaluate_trajectory_ope_agent_does_not_mutate_agent_state() -> None:
    # LinUCB internal matrices must be unchanged after eval — agent.update must not run.
    obs = RecObs(
        user_features=np.ones(4, dtype=np.float64),
        candidate_features=np.eye(4, 4, dtype=np.float64),
        candidate_ids=np.arange(4, dtype=np.int64),
    )
    step = LoggedTrajectoryStep(
        obs=obs, logged_action=0, logged_reward=1.0, propensity=0.5
    )
    source = _SyntheticTrajectorySource(trajectories=[[step] * 5])
    agent = LinUCBAgent(slate_size=1, user_dim=4, item_dim=4, alpha=1.0)
    a_before = agent._a_matrix.copy()
    b_before = agent._b_vector.copy()

    evaluate_trajectory_ope_agent(
        source, agent, agent_name="linucb", max_trajectories=1, seed=0
    )

    assert np.array_equal(agent._a_matrix, a_before)
    assert np.array_equal(agent._b_vector, b_before)
