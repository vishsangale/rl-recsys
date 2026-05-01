from __future__ import annotations

import numpy as np

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecObs


class RandomAgent(Agent):
    """Uniformly random slate selection baseline."""

    def __init__(self, slate_size: int, seed: int | None = None) -> None:
        self._slate_size = slate_size
        self._rng = np.random.default_rng(seed)

    def select_slate(self, obs: RecObs) -> np.ndarray:
        n = len(obs.candidate_features)
        if self._slate_size > n:
            raise ValueError(
                f"slate_size={self._slate_size} exceeds num_candidates={n}"
            )
        return self._rng.choice(n, size=self._slate_size, replace=False)

    def update(
        self,
        obs: RecObs,
        slate: np.ndarray,
        reward: float,
        clicks: np.ndarray,
        next_obs: RecObs,
    ) -> dict[str, float]:
        return {}
