from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class RewardModel(ABC):
    """Abstract base class for reward computation."""

    @abstractmethod
    def __call__(self, clicks: np.ndarray) -> float:
        """Compute a scalar reward from per-position click signals."""
        ...
