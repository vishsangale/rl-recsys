from __future__ import annotations

import numpy as np

from rl_recsys.agents._linear_base import LinearBanditBase
from rl_recsys.environments.base import RecObs


class LinTSAgent(LinearBanditBase):
    """Linear Thompson sampling: sample θ ~ N(A^{-1} b, σ^2 A^{-1}) and
    score with features @ θ."""

    def __init__(
        self,
        slate_size: int,
        user_dim: int,
        item_dim: int,
        *,
        sigma: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(slate_size=slate_size, user_dim=user_dim, item_dim=item_dim)
        self._sigma = float(sigma)
        self._rng = rng if rng is not None else np.random.default_rng()

    def score_items(self, obs: RecObs) -> np.ndarray:
        features = self._candidate_features(obs)
        a_inv = np.linalg.inv(self._a_matrix)
        mu = a_inv @ self._b_vector
        cov = self._sigma ** 2 * a_inv
        # Symmetrize for numerical stability before Cholesky.
        cov = 0.5 * (cov + cov.T)
        # Add tiny jitter for safety.
        cov += 1e-9 * np.eye(cov.shape[0])
        l = np.linalg.cholesky(cov)
        z = self._rng.standard_normal(size=mu.shape[0])
        theta = mu + l @ z
        return features @ theta
