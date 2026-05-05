from __future__ import annotations

import numpy as np

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecObs


class LinUCBAgent(Agent):
    """Shared contextual LinUCB agent for slate recommendation."""

    def __init__(
        self,
        slate_size: int,
        user_dim: int,
        item_dim: int,
        alpha: float = 1.0,
    ) -> None:
        self._slate_size = slate_size
        self._user_dim = user_dim
        self._item_dim = item_dim
        self._interaction_dim = min(user_dim, item_dim)
        self._alpha = alpha
        self._feature_dim = user_dim + item_dim + self._interaction_dim
        self._a_matrix = np.eye(self._feature_dim, dtype=np.float64)
        self._b_vector = np.zeros(self._feature_dim, dtype=np.float64)

    def select_slate(self, obs: RecObs) -> np.ndarray:
        n = len(obs.candidate_features)
        if self._slate_size > n:
            raise ValueError(
                f"slate_size={self._slate_size} exceeds num_candidates={n}"
            )
        scores = self.score_candidates(obs)
        return np.argsort(scores)[-self._slate_size :][::-1]

    def update(
        self,
        obs: RecObs,
        slate: np.ndarray,
        reward: float,
        clicks: np.ndarray,
        next_obs: RecObs,
    ) -> dict[str, float]:
        features = self._candidate_features(obs)[np.asarray(slate)]
        clicks = np.asarray(clicks, dtype=np.float64)
        if len(clicks) != len(features):
            raise ValueError(
                f"clicks length {len(clicks)} does not match slate length {len(features)}"
            )

        for x, click in zip(features, clicks):
            self._a_matrix += np.outer(x, x)
            self._b_vector += float(click) * x

        return {
            "agent_updates": float(len(features)),
            "agent_click_mean": float(clicks.mean()) if len(clicks) else 0.0,
        }

    def score_candidates(self, obs: RecObs) -> np.ndarray:
        """Return LinUCB scores for every candidate in the observation."""
        features = self._candidate_features(obs)
        theta = np.linalg.solve(self._a_matrix, self._b_vector)
        mean_scores = features @ theta
        solved = np.linalg.solve(self._a_matrix, features.T).T
        variances = np.einsum("ij,ij->i", features, solved)
        bonuses = self._alpha * np.sqrt(np.clip(variances, a_min=0.0, a_max=None))
        return mean_scores + bonuses

    def _candidate_features(self, obs: RecObs) -> np.ndarray:
        user = np.asarray(obs.user_features, dtype=np.float64)
        items = np.asarray(obs.candidate_features, dtype=np.float64)
        if user.shape != (self._user_dim,):
            raise ValueError(
                f"user_features shape {user.shape} does not match ({self._user_dim},)"
            )
        if items.ndim != 2 or items.shape[1] != self._item_dim:
            raise ValueError(
                "candidate_features shape "
                f"{items.shape} does not match (*, {self._item_dim})"
            )
        user_block = np.broadcast_to(user, (items.shape[0], self._user_dim))
        d = self._interaction_dim
        interaction = user_block[:, :d] * items[:, :d]
        return np.concatenate([user_block, items, interaction], axis=1)
