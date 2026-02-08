from __future__ import annotations

import numpy as np

from rl_recsys.config import EnvConfig
from rl_recsys.environments.base import RecEnv, RecObs, RecStep


class SyntheticEnv(RecEnv):
    """Latent-factor environment with position-biased clicks.

    Relevance is the dot product of user and item latent vectors.
    Click probability at position k: sigmoid(relevance) * (1 / (k+1))^decay.
    """

    def __init__(self, cfg: EnvConfig) -> None:
        self._cfg = cfg
        self._rng = np.random.default_rng(0)
        # fixed item pool
        self._all_items = self._rng.standard_normal(
            (cfg.num_items, cfg.item_dim)
        ).astype(np.float32)
        self._user: np.ndarray | None = None
        self._candidates: np.ndarray | None = None
        self._candidate_ids: np.ndarray | None = None

    # -- properties --

    @property
    def slate_size(self) -> int:
        return self._cfg.slate_size

    @property
    def num_candidates(self) -> int:
        return self._cfg.num_candidates

    @property
    def user_dim(self) -> int:
        return self._cfg.user_dim

    @property
    def item_dim(self) -> int:
        return self._cfg.item_dim

    # -- core --

    def reset(self, seed: int | None = None) -> RecObs:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._user = self._rng.standard_normal(self._cfg.user_dim).astype(np.float32)
        self._sample_candidates()
        return self._obs()

    def step(self, slate: np.ndarray) -> RecStep:
        assert self._user is not None
        assert len(slate) == self._cfg.slate_size

        slate_items = self._candidates[slate]  # (slate_size, item_dim)
        relevances = slate_items @ self._user  # (slate_size,)

        # position-biased click model
        probs = _sigmoid(relevances) * self._position_discount()
        clicks = (self._rng.random(self._cfg.slate_size) < probs).astype(np.float32)
        reward = float(clicks.sum())

        # new candidates for next step
        self._sample_candidates()
        return RecStep(obs=self._obs(), reward=reward, clicks=clicks, done=False)

    # -- helpers --

    def _sample_candidates(self) -> None:
        self._candidate_ids = self._rng.choice(
            self._cfg.num_items, size=self._cfg.num_candidates, replace=False
        )
        self._candidates = self._all_items[self._candidate_ids]

    def _obs(self) -> RecObs:
        return RecObs(
            user_features=self._user.copy(),
            candidate_features=self._candidates.copy(),
            candidate_ids=self._candidate_ids.copy(),
        )

    def _position_discount(self) -> np.ndarray:
        positions = np.arange(1, self._cfg.slate_size + 1, dtype=np.float32)
        return 1.0 / positions ** self._cfg.position_bias_decay


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
