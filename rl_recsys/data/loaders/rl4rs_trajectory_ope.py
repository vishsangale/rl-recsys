from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from rl_recsys.environments.base import RecObs
from rl_recsys.evaluation.behavior_policy import BehaviorPolicy
from rl_recsys.evaluation.ope_trajectory import LoggedTrajectoryStep


class RL4RSTrajectoryOPESource:
    """LoggedTrajectorySource over RL4RS dataset B sessions_b.parquet.

    Groups rows by session_id ordered by sequence_id and yields one
    LoggedTrajectoryStep per row. Reward = sum(user_feedback). Propensity is
    computed by a pre-fitted BehaviorPolicy.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        behavior_policy: BehaviorPolicy,
        *,
        slate_size: int,
    ) -> None:
        self._df = pd.read_parquet(parquet_path)
        self._policy = behavior_policy
        self._slate_size = int(slate_size)

    def iter_trajectories(
        self, *, max_trajectories: int | None = None, seed: int | None = None
    ) -> Iterator[list[LoggedTrajectoryStep]]:
        ordered = self._df.sort_values(["session_id", "sequence_id"], kind="stable")
        groups = ordered.groupby("session_id", sort=False)
        session_ids = list(groups.groups.keys())
        rng = np.random.default_rng(0 if seed is None else seed)
        if seed is not None:
            rng.shuffle(session_ids)

        emitted = 0
        for sid in session_ids:
            if max_trajectories is not None and emitted >= max_trajectories:
                break
            group = groups.get_group(sid)
            steps: list[LoggedTrajectoryStep] = []
            for _, row in group.iterrows():
                user_features = np.array(list(row["user_state"]), dtype=np.float64)
                candidate_features = np.array(
                    list(row["candidate_features"]), dtype=np.float64
                )
                candidate_ids = np.array(list(row["candidate_ids"]), dtype=np.int64)
                logged_slate_ids = np.array(list(row["slate"]), dtype=np.int64)
                logged_reward = float(np.sum(row["user_feedback"]))

                # slate_propensity expects candidate indices (position in
                # candidate_ids), not item ids. Convert using the candidate list.
                cand_ids_list = candidate_ids.tolist()
                try:
                    slate_indices = np.array(
                        [cand_ids_list.index(int(x)) for x in logged_slate_ids],
                        dtype=np.int64,
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"session {sid}: logged slate item not found in "
                        f"candidate_ids — {exc}"
                    ) from exc

                propensity = self._policy.slate_propensity(
                    user_features, candidate_features, slate_indices,
                )
                obs = RecObs(
                    user_features=user_features,
                    candidate_features=candidate_features,
                    candidate_ids=candidate_ids,
                )
                steps.append(
                    LoggedTrajectoryStep(
                        obs=obs,
                        logged_action=logged_slate_ids,
                        logged_reward=logged_reward,
                        propensity=propensity,
                    )
                )
            yield steps
            emitted += 1
