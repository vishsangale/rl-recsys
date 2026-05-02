from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rl_recsys.environments.dataset_base import BanditDatasetEnv
from rl_recsys.environments.features import hashed_vector


class KuaiRecEnv(BanditDatasetEnv):
    def __init__(
        self,
        processed_dir: str | Path,
        *,
        slate_size: int = 1,
        num_candidates: int = 50,
        feature_dim: int = 16,
        feature_source: str = "native",
        seed: int = 0,
    ) -> None:
        processed_dir = Path(processed_dir)
        interactions = pd.read_parquet(processed_dir / "interactions.parquet")

        self._native_item_feat_map: dict[int, np.ndarray] | None = None
        if feature_source == "native":
            feat_path = processed_dir / "item_features.parquet"
            if not feat_path.exists():
                raise FileNotFoundError(
                    f"item_features.parquet not found at {feat_path}. "
                    "Rerun pipeline with --process to generate it."
                )
            feat_df = pd.read_parquet(feat_path)
            feat_cols = [c for c in feat_df.columns if c != "item_id"]
            n_feat = len(feat_cols)
            self._native_item_feat_map = {}
            for _, row in feat_df.iterrows():
                raw = row[feat_cols].to_numpy(dtype=np.float32)
                padded = np.zeros(feature_dim, dtype=np.float32)
                padded[:min(n_feat, feature_dim)] = raw[:min(n_feat, feature_dim)]
                self._native_item_feat_map[int(row["item_id"])] = padded

        super().__init__(
            interactions,
            slate_size=slate_size,
            num_candidates=num_candidates,
            feature_dim=feature_dim,
            feature_source=feature_source,
            seed=seed,
        )

    def _compute_reward(self, row: pd.Series, clicks: np.ndarray) -> float:
        return float(row["rating"]) * float(clicks.sum())

    def _get_item_features(
        self, row: pd.Series, candidate_ids: np.ndarray
    ) -> np.ndarray:
        if self._feature_source == "native" and self._native_item_feat_map is not None:
            vecs = []
            for iid in candidate_ids:
                feat = self._native_item_feat_map.get(int(iid))
                if feat is None:
                    feat = hashed_vector("item", int(iid), self._feature_dim).astype(np.float32)
                vecs.append(feat)
            return np.stack(vecs)
        return super()._get_item_features(row, candidate_ids)
