from __future__ import annotations

import numpy as np

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecObs


class OracleClickAgent(Agent):
    """Cheats by reading the logged clicks to rank logged items first.
    Eval-only upper bound. Opt-in via the harness's --agents flag."""

    def __init__(self, slate_size: int) -> None:
        self._slate_size = int(slate_size)

    def select_slate(self, obs: RecObs) -> np.ndarray:
        if obs.logged_action is None or obs.logged_clicks is None:
            raise ValueError(
                "OracleClickAgent requires obs.logged_action and "
                "obs.logged_clicks (replay-mode source)"
            )
        # Rank logged items by their click value; take top-k.
        order = np.argsort(-obs.logged_clicks, kind="stable")
        ranked_logged = obs.logged_action[order]
        if len(ranked_logged) >= self._slate_size:
            return ranked_logged[: self._slate_size].astype(np.int64)
        # Pad with non-logged candidates in arbitrary order.
        used = set(int(x) for x in ranked_logged)
        n = len(obs.candidate_features)
        padding = [i for i in range(n) if i not in used][
            : self._slate_size - len(ranked_logged)
        ]
        return np.concatenate(
            [ranked_logged, np.asarray(padding, dtype=np.int64)]
        ).astype(np.int64)

    def score_items(self, obs: RecObs) -> np.ndarray:
        if obs.logged_action is None or obs.logged_clicks is None:
            raise ValueError(
                "OracleClickAgent.score_items requires logged_action and "
                "logged_clicks"
            )
        scores = np.zeros(len(obs.candidate_features), dtype=np.float64)
        scores[obs.logged_action] = obs.logged_clicks.astype(np.float64)
        return scores

    def update(self, obs, slate, reward, clicks, next_obs) -> dict[str, float]:
        return {}

    def train_offline(self, source, *, seed: int = 0) -> dict[str, float]:
        return {}
