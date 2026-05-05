from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rl_recsys.environments.dataset_base import SessionDatasetEnv


class RL4RSEnv(SessionDatasetEnv):
    def __init__(
        self,
        processed_dir: str | Path,
        *,
        slate_size: int = 6,
        feature_dim: int = 32,
        feature_source: str = "native",
        seed: int = 0,
    ) -> None:
        processed_dir = Path(processed_dir)
        df = pd.read_parquet(processed_dir / "sessions.parquet")

        sample_row = df.iloc[0]
        if feature_source == "native":
            # Derive dims from data; user and item dims may differ.
            self._user_dim = len(sample_row["user_state"])
            self._item_dim = len(sample_row["item_features"][0])
            feature_dim = self._user_dim  # parent uses this for hashing fallback only
        else:
            self._user_dim = feature_dim
            self._item_dim = feature_dim

        slate_lengths = df["slate"].apply(len)
        if slate_lengths.nunique() != 1:
            raise ValueError(
                f"Inconsistent slate lengths in sessions.parquet: "
                f"found {sorted(slate_lengths.unique().tolist())}."
            )
        num_candidates = int(slate_lengths.iloc[0])

        sessions: dict[int, pd.DataFrame] = {
            int(sid): grp.sort_values("step").reset_index(drop=True)
            for sid, grp in df.groupby("session_id")
        }

        super().__init__(
            sessions,
            slate_size=slate_size,
            num_candidates=num_candidates,
            feature_dim=feature_dim,
            feature_source=feature_source,
            seed=seed,
        )

    @property
    def user_dim(self) -> int:
        return self._user_dim

    @property
    def item_dim(self) -> int:
        return self._item_dim

    def _compute_reward(self, row: pd.Series, clicks: np.ndarray) -> float:
        return float(clicks.sum())

    def _get_user_features(self, row: pd.Series) -> np.ndarray:
        if self._feature_source == "native":
            return np.array(row["user_state"], dtype=np.float32)
        return super()._get_user_features(row)

    def _get_item_features(
        self, row: pd.Series, candidate_ids: np.ndarray
    ) -> np.ndarray:
        if self._feature_source == "native":
            # coerce to plain Python lists: pyarrow may return object arrays of arrays
            return np.array([list(r) for r in row["item_features"]], dtype=np.float32)
        return super()._get_item_features(row, candidate_ids)
