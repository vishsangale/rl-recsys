from __future__ import annotations

import numpy as np
import pandas as pd

from rl_recsys.environments.base import RecEnv, RecObs, RecStep
from rl_recsys.environments.features import hashed_vector as _hashed_vector


class LoggedInteractionEnv(RecEnv):
    """Sampled bandit environment built from logged user-item interactions."""

    def __init__(
        self,
        interactions: pd.DataFrame,
        *,
        slate_size: int = 10,
        num_candidates: int = 50,
        feature_dim: int = 16,
        rating_threshold: float = 4.0,
        seed: int = 0,
    ) -> None:
        required = {"user_id", "item_id"}
        missing = required - set(interactions.columns)
        if missing:
            raise ValueError(f"interactions missing required columns: {sorted(missing)}")
        if feature_dim < 3:
            raise ValueError("feature_dim must be at least 3")
        if slate_size > num_candidates:
            raise ValueError("slate_size must be <= num_candidates")

        df = interactions.copy()
        if "rating" in df.columns:
            df["label"] = (pd.to_numeric(df["rating"]) >= rating_threshold).astype(float)
        else:
            df["label"] = 1.0
        df = df[df["label"] > 0].reset_index(drop=True)
        if df.empty:
            raise ValueError("no positive interactions after applying rating_threshold")

        self._df = df
        self._slate_size = slate_size
        self._num_candidates = num_candidates
        self._feature_dim = feature_dim
        self._rng = np.random.default_rng(seed)
        self._all_items = np.array(sorted(interactions["item_id"].unique()), dtype=np.int64)
        if len(self._all_items) < num_candidates:
            raise ValueError(
                f"num_candidates={num_candidates} exceeds item count={len(self._all_items)}"
            )

        all_labels = interactions[["user_id", "item_id"]].copy()
        if "rating" in interactions.columns:
            all_labels["label"] = (
                pd.to_numeric(interactions["rating"]) >= rating_threshold
            ).astype(float)
        else:
            all_labels["label"] = 1.0
        self._user_positive_items = {
            int(user_id): set(group["item_id"].astype(int).tolist())
            for user_id, group in all_labels[all_labels["label"] > 0].groupby("user_id")
        }
        self._user_features = self._build_entity_features(
            all_labels, entity_col="user_id", prefix="user"
        )
        self._item_features = self._build_entity_features(
            all_labels, entity_col="item_id", prefix="item"
        )
        self._current_positive_item_id: int | None = None
        self._current_candidate_ids: np.ndarray | None = None
        self._current_obs: RecObs | None = None

    @property
    def slate_size(self) -> int:
        return self._slate_size

    @property
    def num_candidates(self) -> int:
        return self._num_candidates

    @property
    def user_dim(self) -> int:
        return self._feature_dim

    @property
    def item_dim(self) -> int:
        return self._feature_dim

    def reset(self, seed: int | None = None) -> RecObs:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        row = self._df.iloc[int(self._rng.integers(0, len(self._df)))]
        user_id = int(row["user_id"])
        positive_item_id = int(row["item_id"])
        negative_ids = self._sample_negative_items(user_id, positive_item_id)
        candidate_ids = np.concatenate([[positive_item_id], negative_ids])
        self._rng.shuffle(candidate_ids)

        self._current_positive_item_id = positive_item_id
        self._current_candidate_ids = candidate_ids
        self._current_obs = RecObs(
            user_features=self._user_features[user_id].copy(),
            candidate_features=np.stack(
                [self._item_features[int(item_id)] for item_id in candidate_ids]
            ).astype(np.float32),
            candidate_ids=candidate_ids.copy(),
        )
        return self._current_obs

    def step(self, slate: np.ndarray) -> RecStep:
        if self._current_obs is None or self._current_candidate_ids is None:
            raise RuntimeError("reset must be called before step")
        slate = np.asarray(slate, dtype=np.int64)
        selected_item_ids = self._current_candidate_ids[slate]
        clicks = (selected_item_ids == self._current_positive_item_id).astype(np.float32)
        return RecStep(
            obs=self._current_obs,
            reward=float(clicks.sum()),
            clicks=clicks,
            done=True,
        )

    def _sample_negative_items(
        self, user_id: int, positive_item_id: int
    ) -> np.ndarray:
        excluded = set(self._user_positive_items.get(user_id, set()))
        excluded.add(positive_item_id)
        pool = np.array(
            [item_id for item_id in self._all_items if int(item_id) not in excluded],
            dtype=np.int64,
        )
        needed = self._num_candidates - 1
        if len(pool) < needed:
            pool = self._all_items[self._all_items != positive_item_id]
        return self._rng.choice(pool, size=needed, replace=False)

    def _build_entity_features(
        self, labels: pd.DataFrame, *, entity_col: str, prefix: str
    ) -> dict[int, np.ndarray]:
        grouped = labels.groupby(entity_col)["label"].agg(["count", "mean"])
        max_log_count = float(np.log1p(grouped["count"].max()))
        features: dict[int, np.ndarray] = {}
        for entity_id, stats in grouped.iterrows():
            vec = _hashed_vector(prefix, int(entity_id), self._feature_dim)
            vec[0] = 1.0
            vec[1] = float(np.log1p(stats["count"]) / max_log_count)
            vec[2] = float(stats["mean"])
            features[int(entity_id)] = vec.astype(np.float32)
        return features
