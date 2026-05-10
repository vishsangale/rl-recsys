from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class HistoryStep:
    """One past step within the current session.

    Used by sequence-aware agents (SASRec, TopK REINFORCE, Decision
    Transformer). Populated by trajectory loaders; outside replay
    contexts the consumer constructs an empty tuple.
    """

    slate: np.ndarray   # (slate_size,) candidate indices into RecObs.candidate_*
    clicks: np.ndarray  # (slate_size,) 0/1


@dataclass
class RecObs:
    """Observation returned by a recommendation environment."""

    user_features: np.ndarray  # (user_dim,)
    candidate_features: np.ndarray  # (num_candidates, item_dim)
    candidate_ids: np.ndarray  # (num_candidates,)
    history: tuple[HistoryStep, ...] = field(default_factory=tuple)
    # Replay-mode-only fields. None outside replay sources.
    logged_action: np.ndarray | None = None
    logged_clicks: np.ndarray | None = None


@dataclass
class RecStep:
    """Result of taking an action in the environment."""

    obs: RecObs
    reward: float
    clicks: np.ndarray  # (slate_size,) binary
    done: bool


class RecEnv(ABC):
    """Abstract base class for recommendation environments."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> RecObs:
        ...

    @abstractmethod
    def step(self, slate: np.ndarray) -> RecStep:
        """Take an action (a slate of candidate indices) and return the result."""
        ...

    @property
    @abstractmethod
    def slate_size(self) -> int:
        ...

    @property
    @abstractmethod
    def num_candidates(self) -> int:
        ...

    @property
    @abstractmethod
    def user_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def item_dim(self) -> int:
        ...
