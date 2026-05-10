from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from rl_recsys.environments.base import HistoryStep, RecObs
from rl_recsys.evaluation.behavior_policy import BehaviorPolicy
from rl_recsys.evaluation.ope_trajectory import LoggedTrajectoryStep


class RL4RSTrajectoryOPESource:
    """LoggedTrajectorySource over RL4RS dataset B sessions_b.parquet.

    Groups rows by session_id ordered by sequence_id and yields one
    LoggedTrajectoryStep per row. Reward = sum(user_feedback). Propensity is
    computed by a pre-fitted BehaviorPolicy.

    The candidate universe (sorted unique item ids + per-item feature vectors)
    is derived from the parquet at init time and cached as instance attributes.
    No per-row duplication — sessions_b.parquet stores only per-row data.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        behavior_policy: BehaviorPolicy,
        *,
        slate_size: int,
        session_filter: set[int] | None = None,
    ) -> None:
        self._df = pd.read_parquet(parquet_path)
        self._policy = behavior_policy
        self._slate_size = int(slate_size)
        self._session_filter = (
            None if session_filter is None else {int(s) for s in session_filter}
        )

        # Build the candidate universe from the slate column. Using
        # pyarrow.compute.unique on the flat list-array is ~7x faster than a
        # Python set for large parquets, and mirrors FinnNoSlateTrajectoryLoader.
        slate_table = pq.read_table(parquet_path, columns=["slate"])
        flat_items = slate_table["slate"].combine_chunks().flatten()
        universe = np.sort(pc.unique(flat_items).to_numpy()).astype(np.int64)

        # Build (item_id → features) by scanning slate+item_features once.
        # We stop as soon as every item in the universe has been seen.
        feature_for: dict[int, list[float]] = {}
        for slate, item_feats in zip(self._df["slate"], self._df["item_features"]):
            for item_id, feat in zip(slate, item_feats):
                if int(item_id) not in feature_for:
                    feature_for[int(item_id)] = list(feat)
            if len(feature_for) == len(universe):
                break  # done — saw every item at least once

        self._candidate_ids: np.ndarray = universe
        self._candidate_features: np.ndarray = np.array(
            [feature_for[int(i)] for i in universe], dtype=np.float64
        )
        self._cand_id_to_idx: dict[int, int] = {
            int(cid): k for k, cid in enumerate(universe)
        }

        # Sort + filter once. Row-positions in self._ordered are the indices
        # into self._propensities.
        ordered = self._df.sort_values(
            ["session_id", "sequence_id"], kind="stable",
        )
        if self._session_filter is not None:
            ordered = ordered[ordered["session_id"].isin(self._session_filter)]
        self._ordered = ordered.reset_index(drop=True)

        # Precompute propensities for every row in self._ordered in one batched
        # pass. Empty filter → zero-length _propensities; iter_trajectories
        # raises on iteration as before.
        if len(self._ordered) > 0:
            users = np.stack([
                np.array(list(u), dtype=np.float64)
                for u in self._ordered["user_state"]
            ])
            try:
                slate_indices = np.stack([
                    np.array(
                        [self._cand_id_to_idx[int(x)] for x in s], dtype=np.int64,
                    )
                    for s in self._ordered["slate"]
                ])
            except KeyError as exc:
                raise ValueError(
                    f"logged slate item not found in candidate universe — {exc}"
                ) from exc
            started = perf_counter()
            log_props = self._policy.slate_log_propensities_batch(
                users, slate_indices, self._candidate_features,
            )
            self._propensities = np.exp(log_props).astype(np.float64)
            elapsed = perf_counter() - started
            print(
                f"propensity precompute: {len(self._propensities)} rows "
                f"in {elapsed:.1f}s",
                flush=True,
            )
            if (self._propensities <= 0).any():
                raise ValueError("zero propensity in logged slate")
        else:
            self._propensities = np.zeros(0, dtype=np.float64)

    def iter_trajectories(
        self, *, max_trajectories: int | None = None, seed: int | None = None,
    ) -> Iterator[list[LoggedTrajectoryStep]]:
        groups = self._ordered.groupby("session_id", sort=False)
        session_ids = list(groups.groups.keys())
        rng = np.random.default_rng(0 if seed is None else seed)
        if seed is not None:
            rng.shuffle(session_ids)

        if self._session_filter is not None and not session_ids:
            raise ValueError(
                "session_filter excludes every session in the parquet — "
                "no trajectories to emit"
            )

        emitted = 0
        for sid in session_ids:
            if max_trajectories is not None and emitted >= max_trajectories:
                break
            group = groups.get_group(sid)
            history: list[HistoryStep] = []
            steps: list[LoggedTrajectoryStep] = []
            for row_pos, row in zip(group.index, group.itertuples(index=False)):
                user_features = np.array(list(row.user_state), dtype=np.float64)
                logged_slate_ids = np.array(list(row.slate), dtype=np.int64)
                logged_reward = float(np.sum(row.user_feedback))
                logged_clicks = np.array(
                    list(row.user_feedback), dtype=np.int64,
                )

                slate_indices = np.array(
                    [self._cand_id_to_idx[int(x)] for x in logged_slate_ids],
                    dtype=np.int64,
                )

                propensity = float(self._propensities[row_pos])
                obs = RecObs(
                    user_features=user_features,
                    candidate_features=self._candidate_features,
                    candidate_ids=self._candidate_ids,
                    history=tuple(history),
                    logged_action=slate_indices,
                    logged_clicks=logged_clicks,
                )
                steps.append(
                    LoggedTrajectoryStep(
                        obs=obs,
                        logged_action=slate_indices,
                        logged_reward=logged_reward,
                        logged_clicks=logged_clicks,
                        propensity=propensity,
                    )
                )
                history.append(
                    HistoryStep(slate=slate_indices, clicks=logged_clicks)
                )
            yield steps
            emitted += 1
