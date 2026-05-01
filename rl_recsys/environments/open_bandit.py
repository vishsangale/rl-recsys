from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Hashable

import numpy as np
import pandas as pd

from rl_recsys.environments.base import RecObs


@dataclass(frozen=True)
class LoggedBanditEvent:
    """One logged bandit event with a sampled candidate set."""

    obs: RecObs
    logged_action: int
    logged_reward: float
    propensity: float
    logged_item_id: Hashable


class OpenBanditEventSampler:
    """Sample Open Bandit rows as top-1 logged bandit evaluation events."""

    def __init__(
        self,
        interactions: pd.DataFrame,
        *,
        num_candidates: int = 50,
        feature_dim: int = 16,
        seed: int = 0,
    ) -> None:
        required = {"user_id", "item_id", "rating", "propensity_score"}
        missing = required - set(interactions.columns)
        if missing:
            raise ValueError(f"open bandit data missing required columns: {sorted(missing)}")
        if num_candidates < 1:
            raise ValueError("num_candidates must be at least 1")
        if feature_dim < 3:
            raise ValueError("feature_dim must be at least 3")

        df = interactions.copy()
        df["rating"] = pd.to_numeric(df["rating"]).astype(float)
        df["propensity_score"] = pd.to_numeric(df["propensity_score"]).astype(float)
        propensities = df["propensity_score"].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(propensities)):
            raise ValueError("propensity_score values must be finite")
        if np.any(propensities <= 0.0) or np.any(propensities > 1.0):
            raise ValueError("propensity_score values must be probabilities in (0, 1]")

        self._df = df.reset_index(drop=True)
        self._num_candidates = num_candidates
        self._feature_dim = feature_dim
        self._rng = np.random.default_rng(seed)
        self._all_items = np.array(sorted(self._df["item_id"].unique().tolist()))
        if len(self._all_items) < num_candidates:
            raise ValueError(
                f"num_candidates={num_candidates} exceeds item count={len(self._all_items)}"
            )

        self._user_features = self._build_entity_features(
            self._df, entity_col="user_id", prefix="user"
        )
        self._item_features = self._build_entity_features(
            self._df, entity_col="item_id", prefix="item"
        )

    @property
    def num_candidates(self) -> int:
        return self._num_candidates

    @property
    def user_dim(self) -> int:
        return self._feature_dim

    @property
    def item_dim(self) -> int:
        return self._feature_dim

    def sample_event(self, seed: int | None = None) -> LoggedBanditEvent:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        row_idx = int(self._rng.integers(0, len(self._df)))
        user_id = self._df.at[row_idx, "user_id"]
        logged_item_id = self._df.at[row_idx, "item_id"]
        negative_ids = self._sample_negative_items(logged_item_id)
        candidate_ids = np.concatenate([[logged_item_id], negative_ids])
        self._rng.shuffle(candidate_ids)
        logged_positions = np.flatnonzero(candidate_ids == logged_item_id)
        if len(logged_positions) != 1:
            raise RuntimeError("sampled candidates must contain the logged item exactly once")

        obs = RecObs(
            user_features=self._user_features[user_id].copy(),
            candidate_features=np.stack(
                [self._item_features[item_id] for item_id in candidate_ids]
            ).astype(np.float32),
            candidate_ids=candidate_ids.copy(),
        )
        return LoggedBanditEvent(
            obs=obs,
            logged_action=int(logged_positions[0]),
            logged_reward=float(self._df.at[row_idx, "rating"]),
            propensity=float(self._df.at[row_idx, "propensity_score"]),
            logged_item_id=logged_item_id,
        )

    def _sample_negative_items(self, logged_item_id: Hashable) -> np.ndarray:
        needed = self._num_candidates - 1
        if needed == 0:
            return np.array([], dtype=self._all_items.dtype)
        pool = self._all_items[self._all_items != logged_item_id]
        return self._rng.choice(pool, size=needed, replace=False)

    def _build_entity_features(
        self, interactions: pd.DataFrame, *, entity_col: str, prefix: str
    ) -> dict[Hashable, np.ndarray]:
        grouped = interactions.groupby(entity_col)["rating"].agg(["count", "mean"])
        max_log_count = float(np.log1p(grouped["count"].max()))
        features: dict[Hashable, np.ndarray] = {}
        for entity_id, stats in grouped.iterrows():
            vec = _hashed_vector(prefix, entity_id, self._feature_dim)
            vec[0] = 1.0
            vec[1] = float(np.log1p(stats["count"]) / max_log_count)
            vec[2] = float(stats["mean"])
            features[entity_id] = vec.astype(np.float32)
        return features


def _hashed_vector(prefix: str, entity_id: Hashable, dim: int) -> np.ndarray:
    digest = hashlib.blake2b(f"{prefix}:{entity_id}".encode("utf-8"), digest_size=8)
    seed = int.from_bytes(digest.digest(), byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float64)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec
