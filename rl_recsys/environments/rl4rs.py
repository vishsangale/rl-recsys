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

        if feature_source == "native":
            sample_row = df.iloc[0]
            actual_user_dim = len(sample_row["user_state"])
            actual_item_dim = len(sample_row["item_features"][0])
            if feature_dim != actual_user_dim:
                raise ValueError(
                    f"feature_dim={feature_dim} does not match user_state length={actual_user_dim} "
                    "in sessions.parquet. Set feature_dim to match the data."
                )
            if feature_dim != actual_item_dim:
                raise ValueError(
                    f"feature_dim={feature_dim} does not match item_features width={actual_item_dim} "
                    "in sessions.parquet. Set feature_dim to match the data."
                )

        sessions: dict[int, pd.DataFrame] = {
            int(sid): grp.sort_values("step").reset_index(drop=True)
            for sid, grp in df.groupby("session_id")
        }
        num_candidates = len(df.iloc[0]["slate"])

        super().__init__(
            sessions,
            slate_size=slate_size,
            num_candidates=num_candidates,
            feature_dim=feature_dim,
            feature_source=feature_source,
            seed=seed,
        )

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
