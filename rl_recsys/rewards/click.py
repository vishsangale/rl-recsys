from __future__ import annotations

import numpy as np

from rl_recsys.rewards.base import RewardModel


class ClickSumReward(RewardModel):
    """Reward = total number of clicks on the slate."""

    def __call__(self, clicks: np.ndarray) -> float:
        return float(clicks.sum())


class DCGReward(RewardModel):
    """Discounted Cumulative Gain reward."""

    def __call__(self, clicks: np.ndarray) -> float:
        positions = np.arange(1, len(clicks) + 1, dtype=np.float32)
        return float((clicks / np.log2(positions + 1)).sum())
