from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    user_features: np.ndarray
    candidate_features: np.ndarray
    slate: np.ndarray
    reward: float
    clicks: np.ndarray
    next_user_features: np.ndarray
    next_candidate_features: np.ndarray


class ReplayBuffer:
    """Fixed-size ring buffer of transitions."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._buffer: list[Transition] = []
        self._pos = 0

    def push(self, transition: Transition) -> None:
        if len(self._buffer) < self._capacity:
            self._buffer.append(transition)
        else:
            self._buffer[self._pos] = transition
        self._pos = (self._pos + 1) % self._capacity

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self._buffer, min(batch_size, len(self._buffer)))

    def __len__(self) -> int:
        return len(self._buffer)
