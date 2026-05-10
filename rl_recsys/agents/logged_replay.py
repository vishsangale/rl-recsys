from __future__ import annotations

import numpy as np

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecObs


class LoggedReplayAgent(Agent):
    """Replays the logged slate. Used as an OPE sanity check — the resulting
    DR value should match avg_logged_discounted_return up to IS noise."""

    def __init__(self, slate_size: int) -> None:
        self._slate_size = int(slate_size)

    def _require_logged(self, obs: RecObs) -> np.ndarray:
        if obs.logged_action is None:
            raise ValueError(
                "LoggedReplayAgent requires a replay-mode source — "
                "obs.logged_action is None"
            )
        return obs.logged_action

    def select_slate(self, obs: RecObs) -> np.ndarray:
        return self._require_logged(obs).astype(np.int64)

    def score_items(self, obs: RecObs) -> np.ndarray:
        scores = np.zeros(len(obs.candidate_features), dtype=np.float64)
        scores[self._require_logged(obs)] = 1.0
        return scores

    def update(self, obs, slate, reward, clicks, next_obs) -> dict[str, float]:
        return {}

    def train_offline(self, source, *, seed: int = 0) -> dict[str, float]:
        return {}
