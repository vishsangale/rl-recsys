from __future__ import annotations

import numpy as np

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecObs


class MostPopularAgent(Agent):
    """Ranks candidates by global click counts seen during training."""

    def __init__(self, slate_size: int, num_candidates: int) -> None:
        self._slate_size = int(slate_size)
        self._num_candidates = int(num_candidates)
        self._clicks_per_item = np.zeros(num_candidates, dtype=np.float64)

    def train_offline(self, source, *, seed: int = 0) -> dict[str, float]:
        for traj in source.iter_trajectories(seed=seed):
            for step in traj:
                self._clicks_per_item[step.logged_action] += step.logged_clicks
        return {"items_seen": float(self._clicks_per_item.sum())}

    def select_slate(self, obs: RecObs) -> np.ndarray:
        n = len(obs.candidate_features)
        if self._slate_size > n:
            raise ValueError(
                f"slate_size={self._slate_size} exceeds num_candidates={n}"
            )
        return np.argsort(self._clicks_per_item)[-self._slate_size:][::-1]

    def score_items(self, obs: RecObs) -> np.ndarray:
        return self._clicks_per_item.astype(np.float64)

    def update(self, obs, slate, reward, clicks, next_obs) -> dict[str, float]:
        return {}
