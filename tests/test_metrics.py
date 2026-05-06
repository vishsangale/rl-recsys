from __future__ import annotations

import numpy as np
import pytest

from rl_recsys.training.metrics import discounted_return, per_session_reward


def test_discounted_return_geometric_decay() -> None:
    rewards = np.array([1.0, 1.0, 1.0])
    # gamma=0.5: 1*1 + 1*0.5 + 1*0.25 = 1.75
    assert discounted_return(rewards, gamma=0.5) == pytest.approx(1.75)


def test_discounted_return_single_step() -> None:
    rewards = np.array([3.7])
    assert discounted_return(rewards, gamma=0.95) == pytest.approx(3.7)
    # gamma=0.0: only first step counts, all subsequent are zeroed
    assert discounted_return(np.array([1.0, 2.0, 3.0]), gamma=0.0) == pytest.approx(1.0)


def test_per_session_reward_averages_sessions() -> None:
    # session 1 total=3, session 2 total=5 → mean=4
    sessions = [np.array([1.0, 2.0]), np.array([2.0, 3.0])]
    assert per_session_reward(sessions) == pytest.approx(4.0)
