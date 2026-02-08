from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from rl_recsys.environments.base import RecObs


class Agent(ABC):
    """Abstract base class for recommendation agents."""

    @abstractmethod
    def select_slate(self, obs: RecObs) -> np.ndarray:
        """Return an array of candidate indices forming the slate."""
        ...

    @abstractmethod
    def update(
        self,
        obs: RecObs,
        slate: np.ndarray,
        reward: float,
        clicks: np.ndarray,
        next_obs: RecObs,
    ) -> dict[str, float]:
        """Update the agent and return a dict of logged metrics."""
        ...
