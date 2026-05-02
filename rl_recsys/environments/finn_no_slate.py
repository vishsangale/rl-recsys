from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from rl_recsys.environments.dataset_base import BanditDatasetEnv


class FinnNoSlateEnv(BanditDatasetEnv):
    _LOGGED_SLATE_SIZE = 25

    def __init__(
        self,
        processed_dir: str | Path,
        *,
        slate_size: int = 25,
        feature_dim: int = 16,
        feature_source: str = "hashed",
        seed: int = 0,
    ) -> None:
        processed_dir = Path(processed_dir)
        df = pd.read_parquet(processed_dir / "slates.parquet")

        if feature_source == "native":
            warnings.warn(
                "FinnNoSlateEnv: feature_source='native' is not supported "
                "(FINN.no has no item metadata); falling back to 'hashed'.",
                UserWarning,
                stacklevel=2,
            )
            feature_source = "hashed"

        df = df.copy()
        invalid_clicks = (df["clicks"] < 0) | (df["clicks"] >= self._LOGGED_SLATE_SIZE)
        if invalid_clicks.any():
            raise ValueError(
                f"Found {invalid_clicks.sum()} rows with clicks index outside [0, {self._LOGGED_SLATE_SIZE})"
            )
        bad_length = df["slate"].apply(len) != self._LOGGED_SLATE_SIZE
        if bad_length.any():
            raise ValueError(
                f"Found {bad_length.sum()} rows where slate length != {self._LOGGED_SLATE_SIZE}"
            )
        df["item_id"] = df.apply(lambda r: r["slate"][int(r["clicks"])], axis=1)

        super().__init__(
            df,
            slate_size=slate_size,
            num_candidates=self._LOGGED_SLATE_SIZE,
            feature_dim=feature_dim,
            feature_source=feature_source,
            seed=seed,
        )

    def _compute_reward(self, row: pd.Series, clicks: np.ndarray) -> float:
        return float(clicks.sum())

    def _build_candidate_ids(
        self, row: pd.Series, positive_id: int, user_id: int
    ) -> np.ndarray:
        return np.array(row["slate"], dtype=np.int64)
