from __future__ import annotations

import numpy as np
import pytest

from rl_recsys.evaluation.ope_trajectory import seq_dr_value


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
