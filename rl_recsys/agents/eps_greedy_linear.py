from __future__ import annotations

import numpy as np

from rl_recsys.agents._linear_base import LinearBanditBase
from rl_recsys.environments.base import RecObs


class EpsGreedyLinearAgent(LinearBanditBase):
    """Linear regression scorer with ε-greedy exploration."""

    def __init__(
        self,
        slate_size: int,
        user_dim: int,
        item_dim: int,
        *,
        epsilon: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(slate_size=slate_size, user_dim=user_dim, item_dim=item_dim)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self._epsilon = float(epsilon)
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
        if self._rng.random() < self._epsilon:
            return self._rng.choice(n, size=self._slate_size, replace=False).astype(
                np.int64
            )
        scores = self.score_items(obs)
        return np.argsort(scores)[-self._slate_size:][::-1]
