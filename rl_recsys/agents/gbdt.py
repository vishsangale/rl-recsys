from __future__ import annotations

import warnings

import numpy as np

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecObs


class GBDTAgent(Agent):
    """LightGBM regressor on (concat(user, item), click)."""

    def __init__(
        self,
        slate_size: int,
        candidate_features: np.ndarray,
        *,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.05,
    ) -> None:
        self._slate_size = int(slate_size)
        self._candidate_features = np.asarray(candidate_features, dtype=np.float64)
        self._n_estimators = int(n_estimators)
        self._max_depth = int(max_depth)
        self._learning_rate = float(learning_rate)
        self._model = None

    def train_offline(self, source, *, seed: int = 0) -> dict[str, float]:
        try:
            import lightgbm as lgb  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "GBDTAgent requires lightgbm — pip install lightgbm"
            ) from exc

        rows_x: list[np.ndarray] = []
        rows_y: list[float] = []
        for traj in source.iter_trajectories(seed=seed):
            for step in traj:
                u = step.obs.user_features
                items = step.obs.candidate_features[step.logged_action]
                for item, click in zip(items, step.logged_clicks):
                    rows_x.append(np.concatenate([u, item]))
                    rows_y.append(float(click))
        if not rows_x:
            return {"n_train_rows": 0.0}
        x = np.stack(rows_x)
        y = np.asarray(rows_y, dtype=np.float64)
        self._model = lgb.LGBMRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            random_state=seed,
            verbose=-1,
        )
        self._model.fit(x, y)
        return {"n_train_rows": float(len(x))}

    def score_items(self, obs: RecObs) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("GBDTAgent.score_items called before train_offline")
        u = np.broadcast_to(
            obs.user_features,
            (len(obs.candidate_features), len(obs.user_features)),
        )
        x = np.concatenate([u, obs.candidate_features], axis=1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            return self._model.predict(x).astype(np.float64)

    def select_slate(self, obs: RecObs) -> np.ndarray:
        return np.argsort(self.score_items(obs))[-self._slate_size:][::-1].astype(
            np.int64
        )

    def update(self, obs, slate, reward, clicks, next_obs) -> dict[str, float]:
        return {}
