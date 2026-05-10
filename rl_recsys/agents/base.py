from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from rl_recsys.environments.base import RecObs

if TYPE_CHECKING:
    from rl_recsys.evaluation.ope_trajectory import LoggedTrajectorySource


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

    def score_items(self, obs: RecObs) -> np.ndarray:
        """Per-candidate scores. Default: zeros (uniform softmax under the
        Boltzmann shim). Override for any agent that wants to influence
        the target-policy probability used by Sequential DR."""
        return np.zeros(len(obs.candidate_features), dtype=np.float64)

    def train_offline(
        self,
        source: LoggedTrajectorySource,
        *,
        seed: int = 0,
    ) -> dict[str, float]:
        """Train on a logged trajectory source. Default: per-step update via
        pretrain_agent_on_logged. Heuristic agents override with no-op;
        DL/batch agents override with their own training loop."""
        # local import: offline_pretrain imports Agent, so a top-level import would be circular
        from rl_recsys.training.offline_pretrain import pretrain_agent_on_logged
        return pretrain_agent_on_logged(self, source, seed=seed)
