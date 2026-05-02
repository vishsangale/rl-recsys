from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from rl_recsys.environments.base import RecEnv, RecObs, RecStep
from rl_recsys.environments.features import hashed_vector


class BanditDatasetEnv(RecEnv, ABC):
    def __init__(
        self,
        interactions: pd.DataFrame,
        *,
        slate_size: int = 1,
        num_candidates: int = 50,
        feature_dim: int = 16,
        feature_source: str = "native",
        seed: int = 0,
    ) -> None:
        if interactions.empty:
            raise ValueError("interactions DataFrame is empty")
        missing = {"user_id", "item_id"} - set(interactions.columns)
        if missing:
            raise ValueError(f"interactions missing columns: {sorted(missing)}")
        if feature_dim < 3:
            raise ValueError("feature_dim must be at least 3")
        n_items = interactions["item_id"].nunique()
        if num_candidates > n_items:
            raise ValueError(
                f"num_candidates={num_candidates} exceeds unique item count={n_items}"
            )
        if slate_size > num_candidates:
            raise ValueError("slate_size must be <= num_candidates")
        if feature_source not in ("native", "hashed"):
            raise ValueError(
                f"feature_source must be 'native' or 'hashed', got {feature_source!r}"
            )

        self._df = interactions.reset_index(drop=True)
        self._slate_size = slate_size
        self._num_candidates = num_candidates
        self._feature_dim = feature_dim
        self._feature_source = feature_source
        self._rng = np.random.default_rng(seed)
        self._all_items = np.array(
            sorted(interactions["item_id"].unique()), dtype=np.int64
        )
        self._user_positive_items: dict[int, set[int]] = {
            int(uid): set(grp["item_id"].astype(int))
            for uid, grp in interactions.groupby("user_id")
        }
        self._current_row: pd.Series | None = None
        self._current_candidate_ids: np.ndarray | None = None

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
        idx = int(self._rng.integers(0, len(self._df)))
        self._current_row = self._df.iloc[idx]
        positive_id = int(self._current_row["item_id"])
        user_id = int(self._current_row["user_id"])
        self._current_candidate_ids = self._build_candidate_ids(
            self._current_row, positive_id, user_id
        )
        return RecObs(
            user_features=self._get_user_features(self._current_row).astype(np.float32),
            candidate_features=self._get_item_features(
                self._current_row, self._current_candidate_ids
            ).astype(np.float32),
            candidate_ids=self._current_candidate_ids.copy(),
        )

    def step(self, slate: np.ndarray) -> RecStep:
        if self._current_row is None or self._current_candidate_ids is None:
            raise RuntimeError("reset() must be called before step()")
        slate = np.asarray(slate, dtype=np.int64)
        selected_ids = self._current_candidate_ids[slate]
        positive_id = int(self._current_row["item_id"])
        clicks = (selected_ids == positive_id).astype(np.float32)
        reward = self._compute_reward(self._current_row, clicks)
        obs = RecObs(
            user_features=self._get_user_features(self._current_row).astype(np.float32),
            candidate_features=self._get_item_features(
                self._current_row, self._current_candidate_ids
            ).astype(np.float32),
            candidate_ids=self._current_candidate_ids.copy(),
        )
        return RecStep(obs=obs, reward=reward, clicks=clicks, done=True)

    def _build_candidate_ids(
        self, row: pd.Series, positive_id: int, user_id: int
    ) -> np.ndarray:
        excluded = self._user_positive_items.get(user_id, set()) | {positive_id}
        pool = np.array(
            [iid for iid in self._all_items if int(iid) not in excluded],
            dtype=np.int64,
        )
        needed = self._num_candidates - 1
        if len(pool) < needed:
            pool = self._all_items[self._all_items != positive_id]
        negatives = self._rng.choice(pool, size=needed, replace=False)
        candidate_ids = np.concatenate([[positive_id], negatives])
        self._rng.shuffle(candidate_ids)
        return candidate_ids

    def _get_user_features(self, row: pd.Series) -> np.ndarray:
        return hashed_vector("user", int(row["user_id"]), self._feature_dim)

    def _get_item_features(
        self, row: pd.Series, candidate_ids: np.ndarray
    ) -> np.ndarray:
        return np.stack(
            [hashed_vector("item", int(iid), self._feature_dim) for iid in candidate_ids]
        ).astype(np.float32)

    @abstractmethod
    def _compute_reward(self, row: pd.Series, clicks: np.ndarray) -> float: ...


class SessionDatasetEnv(RecEnv, ABC):
    def __init__(
        self,
        sessions: dict[int, pd.DataFrame],
        *,
        slate_size: int = 1,
        num_candidates: int = 50,
        feature_dim: int = 16,
        feature_source: str = "native",
        seed: int = 0,
    ) -> None:
        if not sessions:
            raise ValueError("sessions dict is empty")
        if feature_dim < 3:
            raise ValueError("feature_dim must be at least 3")
        if feature_source not in ("native", "hashed"):
            raise ValueError(
                f"feature_source must be 'native' or 'hashed', got {feature_source!r}"
            )

        self._sessions = sessions
        self._session_ids = list(sessions.keys())
        self._slate_size = slate_size
        self._num_candidates = num_candidates
        self._feature_dim = feature_dim
        self._feature_source = feature_source
        self._rng = np.random.default_rng(seed)
        self._current_session: pd.DataFrame | None = None
        self._current_session_id: int | None = None
        self._cursor: int = 0

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
        idx = int(self._rng.integers(0, len(self._session_ids)))
        self._current_session_id = self._session_ids[idx]
        self._current_session = self._sessions[self._current_session_id].reset_index(drop=True)
        self._cursor = 0
        return self._obs_at_cursor()

    def step(self, slate: np.ndarray) -> RecStep:
        if self._current_session is None:
            raise RuntimeError("reset() must be called before step()")
        row = self._current_session.iloc[self._cursor]
        candidate_ids = np.array(row["slate"], dtype=np.int64)
        slate = np.asarray(slate, dtype=np.int64)
        selected_ids = candidate_ids[slate]
        logged_clicks = np.array(row["clicks"], dtype=np.float32)
        clicked_ids = candidate_ids[logged_clicks > 0]
        clicks = np.isin(selected_ids, clicked_ids).astype(np.float32)
        reward = self._compute_reward(row, clicks)
        self._cursor += 1
        done = self._cursor >= len(self._current_session)
        next_obs = self._obs_at_cursor() if not done else self._obs_for_row(row, candidate_ids)
        return RecStep(obs=next_obs, reward=reward, clicks=clicks, done=done)

    def _obs_at_cursor(self) -> RecObs:
        row = self._current_session.iloc[self._cursor]
        candidate_ids = np.array(row["slate"], dtype=np.int64)
        return self._obs_for_row(row, candidate_ids)

    def _obs_for_row(self, row: pd.Series, candidate_ids: np.ndarray) -> RecObs:
        return RecObs(
            user_features=self._get_user_features(row).astype(np.float32),
            candidate_features=self._get_item_features(row, candidate_ids).astype(np.float32),
            candidate_ids=candidate_ids.copy(),
        )

    def _get_user_features(self, row: pd.Series) -> np.ndarray:
        sid = self._current_session_id if self._current_session_id is not None else -1
        return hashed_vector("session", sid, self._feature_dim)

    def _get_item_features(
        self, row: pd.Series, candidate_ids: np.ndarray
    ) -> np.ndarray:
        return np.stack(
            [hashed_vector("item", int(iid), self._feature_dim) for iid in candidate_ids]
        ).astype(np.float32)

    @abstractmethod
    def _compute_reward(self, row: pd.Series, clicks: np.ndarray) -> float: ...
