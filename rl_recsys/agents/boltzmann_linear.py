from __future__ import annotations

import numpy as np

from rl_recsys.agents._linear_base import LinearBanditBase
from rl_recsys.environments.base import RecObs


class BoltzmannLinearAgent(LinearBanditBase):
    """Plackett-Luce sampling without replacement from softmax(scores/T)
    via the Gumbel-top-K trick."""

    def __init__(
        self,
        slate_size: int,
        user_dim: int,
        item_dim: int,
        *,
        temperature: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(slate_size=slate_size, user_dim=user_dim, item_dim=item_dim)
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self._temperature = float(temperature)
        self._rng = rng if rng is not None else np.random.default_rng()

    def score_items(self, obs: RecObs) -> np.ndarray:
        features = self._candidate_features(obs)
        theta = np.linalg.solve(self._a_matrix, self._b_vector)
        return features @ theta

    def select_slate(self, obs: RecObs) -> np.ndarray:
        n = len(obs.candidate_features)
        if self._slate_size > n:
            raise ValueError(
                f"slate_size={self._slate_size} exceeds num_candidates={n}"
            )
        scores = self.score_items(obs) / self._temperature
        # Gumbel-top-K: argsort(scores + Gumbel(0,1)) for sampling without
        # replacement from softmax(scores).
        gumbel = -np.log(-np.log(self._rng.uniform(size=n) + 1e-20) + 1e-20)
        return np.argsort(scores + gumbel)[-self._slate_size:][::-1].astype(np.int64)
