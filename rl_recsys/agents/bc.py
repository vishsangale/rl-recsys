from __future__ import annotations

import numpy as np
import torch

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecObs
from rl_recsys.evaluation.behavior_policy import BehaviorPolicy


class BCAgent(Agent):
    """Behavior cloning: wraps a pre-fit BehaviorPolicy. Selects the top-k
    items by sum-of-position log-softmax scores. The grid runner injects
    the fitted BehaviorPolicy via inject_behavior_policy() before any
    score_items / select_slate call."""

    def __init__(
        self,
        slate_size: int,
        candidate_features: np.ndarray,
        *,
        behavior_policy: BehaviorPolicy | None = None,
    ) -> None:
        self._slate_size = int(slate_size)
        self._candidate_features = np.asarray(candidate_features, dtype=np.float64)
        self._behavior_policy = behavior_policy

    def inject_behavior_policy(self, policy: BehaviorPolicy) -> None:
        self._behavior_policy = policy

    def train_offline(self, source, *, seed: int = 0) -> dict[str, float]:
        # BC reuses an already-fit BehaviorPolicy; no per-agent training.
        return {}

    def _per_position_log_probs(self, obs: RecObs) -> np.ndarray:
        if self._behavior_policy is None:
            raise RuntimeError(
                "BCAgent has no behavior_policy. The grid runner must "
                "call inject_behavior_policy() before scoring."
            )
        user = obs.user_features.astype(np.float64)
        log_probs = np.zeros(
            (self._slate_size, len(obs.candidate_features)), dtype=np.float64
        )
        with torch.no_grad():
            for k in range(self._slate_size):
                logits = self._behavior_policy._score_position(
                    torch.tensor(user, dtype=torch.float64),
                    torch.tensor(obs.candidate_features, dtype=torch.float64),
                    k,
                )
                log_probs[k] = (
                    torch.nn.functional.log_softmax(logits, dim=-1)
                    .cpu().numpy().astype(np.float64)
                )
        return log_probs

    def score_items(self, obs: RecObs) -> np.ndarray:
        return self._per_position_log_probs(obs).sum(axis=0)

    def select_slate(self, obs: RecObs) -> np.ndarray:
        scores = self.score_items(obs)
        return np.argsort(scores)[-self._slate_size:][::-1].astype(np.int64)

    def update(self, obs, slate, reward, clicks, next_obs) -> dict[str, float]:
        return {}
