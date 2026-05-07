from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from rl_recsys.environments.base import RecObs
from rl_recsys.environments.features import hashed_vector
from rl_recsys.evaluation.trajectory import Session, TrajectoryStep


class FinnNoSlateTrajectoryLoader:
    """TrajectoryDataset over data/processed/finn-no-slate/slates.parquet.

    Schema (per pipeline finn_no_slate.py): request_id, user_id, clicks, slate.
    The pipeline pre-filters to clicked rows only — every parquet row has a
    real click, and the `clicks` column stores the position of the click
    within `slate`. Therefore logged_clicked_id = slate[clicks].
    """

    def __init__(
        self,
        parquet_path: str | Path,
        *,
        num_candidates: int,
        feature_dim: int,
        slate_size: int,
        seed: int = 0,
    ) -> None:
        self._df = pd.read_parquet(parquet_path)
        required = {"request_id", "user_id", "clicks", "slate"}
        missing = required - set(self._df.columns)
        if missing:
            raise ValueError(
                f"finn-no-slate parquet missing columns: {sorted(missing)}"
            )
        first_slate = np.asarray(self._df.iloc[0]["slate"], dtype=np.int64)
        slate_len = int(first_slate.shape[0])
        if num_candidates < slate_len:
            raise ValueError(
                f"num_candidates={num_candidates} must be >= "
                f"logged slate length={slate_len}"
            )
        if feature_dim < 1:
            raise ValueError("feature_dim must be at least 1")

        self._num_candidates = int(num_candidates)
        self._feature_dim = int(feature_dim)
        self._slate_size = int(slate_size)
        self._seed = int(seed)

        # Build item universe via pyarrow.compute.unique on the flattened slate
        # column. ~7x faster than a Python set on the 28M-row finn-no-slate
        # parquet (~8s vs ~60s) and avoids materialising a multi-GB intermediate.
        slate_table = pq.read_table(parquet_path, columns=["slate"])
        flat_items = slate_table["slate"].combine_chunks().flatten()
        self._item_universe = np.sort(pc.unique(flat_items).to_numpy()).astype(np.int64)

    def iter_sessions(
        self, *, max_sessions: int | None = None, seed: int | None = None
    ) -> Iterator[Session]:
        ordered = self._df.sort_values(["user_id", "request_id"], kind="stable")
        groups = ordered.groupby("user_id", sort=False)
        rng = np.random.default_rng(self._seed if seed is None else seed)

        user_ids = list(groups.groups.keys())
        if seed is not None:
            rng.shuffle(user_ids)

        emitted = 0
        for user_id in user_ids:
            if max_sessions is not None and emitted >= max_sessions:
                break
            group = groups.get_group(user_id)
            steps: list[TrajectoryStep] = []
            for _, row in group.iterrows():
                logged_slate = np.asarray(row["slate"], dtype=np.int64)
                clicked_id = int(logged_slate[int(row["clicks"])])
                candidate_ids = self._build_candidate_ids(logged_slate, rng)
                candidate_features = np.stack(
                    [hashed_vector("item", int(i), self._feature_dim) for i in candidate_ids]
                )
                user_features = hashed_vector("user", int(user_id), self._feature_dim)
                obs = RecObs(
                    user_features=user_features,
                    candidate_features=candidate_features,
                    candidate_ids=candidate_ids,
                )
                steps.append(
                    TrajectoryStep(
                        obs=obs,
                        logged_slate=logged_slate,
                        logged_clicked_id=clicked_id,
                        logged_reward=1.0,
                    )
                )
            yield Session(session_id=int(user_id), steps=steps)
            emitted += 1

    def _build_candidate_ids(
        self, logged_slate: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        n_pad = self._num_candidates - len(logged_slate)
        if n_pad == 0:
            ids = logged_slate.copy()
        else:
            pool = self._item_universe[~np.isin(self._item_universe, logged_slate)]
            if len(pool) < n_pad:
                extra = np.arange(
                    self._item_universe.max() + 1,
                    self._item_universe.max() + 1 + (n_pad - len(pool)),
                    dtype=np.int64,
                )
                pad_pool = np.concatenate([pool, extra])
                pad = pad_pool[:n_pad]
            else:
                pad = rng.choice(pool, size=n_pad, replace=False)
            ids = np.concatenate([logged_slate, pad.astype(np.int64)])
        # Shuffle so logged-slate items don't always occupy positions 0..slate_size-1
        # — that would let a positional-prior agent trivially achieve coverage.
        rng.shuffle(ids)
        return ids
