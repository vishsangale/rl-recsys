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
    campaign: Hashable | None = None


class OpenBanditEventSampler:
    """Sample Open Bandit rows as top-1 logged bandit evaluation events."""

    def __init__(
        self,
        interactions: pd.DataFrame,
        *,
        num_candidates: int = 50,
        feature_dim: int = 16,
        feature_source: str = "native",
        seed: int = 0,
    ) -> None:
        required = {"user_id", "item_id", "rating", "propensity_score"}
        missing = required - set(interactions.columns)
        if missing:
            raise ValueError(
                f"open bandit data missing required columns: {sorted(missing)}"
            )
        if num_candidates < 1:
            raise ValueError("num_candidates must be at least 1")
        if feature_dim < 3:
            raise ValueError("feature_dim must be at least 3")
        if feature_source not in {"native", "hashed"}:
            raise ValueError("feature_source must be 'native' or 'hashed'")

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
        self._feature_source = feature_source
        self._rng = np.random.default_rng(seed)
        self._campaign_col = "campaign" if "campaign" in self._df.columns else None
        self._all_items_by_campaign = self._build_item_pools(self._df)
        min_item_count = min(
            len(items) for items in self._all_items_by_campaign.values()
        )
        if min_item_count < num_candidates:
            raise ValueError(
                f"num_candidates={num_candidates} exceeds smallest item pool "
                f"size={min_item_count}"
            )

        self._user_context_cols = _prefixed_columns(self._df, "user_feature_")
        self._item_context_cols = _prefixed_columns(self._df, "item_feature_")
        self._affinity_cols = _prefixed_columns(self._df, "user_item_affinity_")
        self._use_native_features = feature_source == "native" and (
            bool(self._user_context_cols)
            or bool(self._item_context_cols)
            or bool(self._affinity_cols)
        )

        self._hashed_user_features = self._build_entity_features(
            self._df, entity_col="user_id", prefix="user"
        )
        self._hashed_item_features = self._build_entity_features(
            self._df, entity_col="item_id", prefix="item"
        )
        self._native_item_features = self._build_native_item_features(self._df)

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
        row = self._df.iloc[row_idx]
        user_id = row["user_id"]
        logged_item_id = row["item_id"]
        campaign = row[self._campaign_col] if self._campaign_col is not None else None
        negative_ids = self._sample_negative_items(logged_item_id, campaign)
        candidate_ids = np.concatenate([[logged_item_id], negative_ids])
        self._rng.shuffle(candidate_ids)
        logged_positions = np.flatnonzero(candidate_ids == logged_item_id)
        if len(logged_positions) != 1:
            raise RuntimeError(
                "sampled candidates must contain the logged item exactly once"
            )

        obs = RecObs(
            user_features=self._user_feature_vector(row).copy(),
            candidate_features=self._candidate_feature_matrix(row, candidate_ids),
            candidate_ids=candidate_ids.copy(),
        )
        return LoggedBanditEvent(
            obs=obs,
            logged_action=int(logged_positions[0]),
            logged_reward=float(self._df.at[row_idx, "rating"]),
            propensity=float(self._df.at[row_idx, "propensity_score"]),
            logged_item_id=logged_item_id,
            campaign=campaign,
        )

    def _sample_negative_items(
        self, logged_item_id: Hashable, campaign: Hashable | None
    ) -> np.ndarray:
        needed = self._num_candidates - 1
        if needed == 0:
            return np.array([], dtype=self._item_pool(campaign).dtype)
        all_items = self._item_pool(campaign)
        pool = all_items[all_items != logged_item_id]
        return self._rng.choice(pool, size=needed, replace=False)

    def _item_pool(self, campaign: Hashable | None) -> np.ndarray:
        return self._all_items_by_campaign[_campaign_key(campaign)]

    def _user_feature_vector(self, row: pd.Series) -> np.ndarray:
        if not self._use_native_features:
            return self._hashed_user_features[row["user_id"]]

        values = [(column, row[column]) for column in self._user_context_cols]
        if self._campaign_col is not None:
            values.append(("campaign", row[self._campaign_col]))
        return _hashed_feature_vector(values, self._feature_dim)

    def _candidate_feature_matrix(
        self,
        row: pd.Series,
        candidate_ids: np.ndarray,
    ) -> np.ndarray:
        campaign = row[self._campaign_col] if self._campaign_col is not None else None
        vectors = [
            self._candidate_feature_vector(row, item_id, campaign)
            for item_id in candidate_ids
        ]
        return np.stack(vectors).astype(np.float32)

    def _candidate_feature_vector(
        self,
        row: pd.Series,
        item_id: Hashable,
        campaign: Hashable | None,
    ) -> np.ndarray:
        if not self._use_native_features:
            return self._hashed_item_features[item_id]

        key = _item_key(campaign, item_id)
        base = self._native_item_features.get(key)
        if base is None:
            base = self._hashed_item_features[item_id]
        else:
            base = base.copy()

        affinity_col = _affinity_column(item_id)
        if affinity_col in self._affinity_cols:
            affinity = pd.to_numeric(row[affinity_col], errors="coerce")
            if pd.notna(affinity):
                base += _hashed_feature_vector(
                    [("user_item_affinity", float(affinity))],
                    self._feature_dim,
                    bias=False,
                )
        return base.astype(np.float32)

    def _build_item_pools(
        self, interactions: pd.DataFrame
    ) -> dict[Hashable | None, np.ndarray]:
        if self._campaign_col is None:
            return {
                _campaign_key(None): np.array(
                    sorted(interactions["item_id"].unique().tolist())
                )
            }
        return {
            _campaign_key(campaign): np.array(sorted(group["item_id"].unique().tolist()))
            for campaign, group in interactions.groupby(self._campaign_col)
        }

    def _build_native_item_features(
        self,
        interactions: pd.DataFrame,
    ) -> dict[tuple[Hashable | None, Hashable], np.ndarray]:
        if not self._item_context_cols:
            return {}

        key_cols = ["item_id"]
        if self._campaign_col is not None:
            key_cols.insert(0, self._campaign_col)
        item_context = interactions[key_cols + self._item_context_cols].drop_duplicates(
            key_cols
        )

        features: dict[tuple[Hashable | None, Hashable], np.ndarray] = {}
        for _, row in item_context.iterrows():
            campaign = (
                row[self._campaign_col] if self._campaign_col is not None else None
            )
            item_id = row["item_id"]
            values = [(column, row[column]) for column in self._item_context_cols]
            if self._campaign_col is not None:
                values.append(("campaign", campaign))
            features[_item_key(campaign, item_id)] = _hashed_feature_vector(
                values, self._feature_dim
            )
        return features

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


def _campaign_key(campaign: Hashable | None) -> Hashable | None:
    return campaign if campaign is not None else "__all__"


def _item_key(
    campaign: Hashable | None,
    item_id: Hashable,
) -> tuple[Hashable | None, Hashable]:
    return _campaign_key(campaign), item_id


def _affinity_column(item_id: Hashable) -> str:
    try:
        item_index = int(item_id)
    except (TypeError, ValueError):
        return f"user_item_affinity_{item_id}"
    return f"user_item_affinity_{item_index}"


def _prefixed_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    return sorted(
        [str(column) for column in df.columns if str(column).startswith(prefix)],
        key=_column_sort_key,
    )


def _column_sort_key(column: str) -> tuple[str, int | str]:
    prefix, _, suffix = column.rpartition("_")
    if suffix.isdigit():
        return prefix, int(suffix)
    return prefix, suffix


def _hashed_vector(prefix: str, entity_id: Hashable, dim: int) -> np.ndarray:
    digest = hashlib.blake2b(f"{prefix}:{entity_id}".encode("utf-8"), digest_size=8)
    seed = int.from_bytes(digest.digest(), byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float64)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _hashed_feature_vector(
    values: list[tuple[str, object]],
    dim: int,
    *,
    bias: bool = True,
) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float64)
    if bias:
        vec[0] = 1.0
    for name, value in values:
        if pd.isna(value):
            continue
        amount: float
        if _is_number(value):
            amount = float(value)
            key = str(name)
        else:
            amount = 1.0
            key = f"{name}={value}"
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8)
        raw = int.from_bytes(digest.digest(), byteorder="little", signed=False)
        index = 1 + (raw % (dim - 1))
        sign = 1.0 if (raw >> 63) == 0 else -1.0
        vec[index] += sign * amount
    return vec.astype(np.float32)


def _is_number(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(numeric)
