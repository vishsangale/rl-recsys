from __future__ import annotations

import numpy as np
import pytest

from rl_recsys.environments.base import HistoryStep, RecObs


def test_history_step_is_a_dataclass_with_slate_and_clicks():
    step = HistoryStep(
        slate=np.array([1, 2, 3], dtype=np.int64),
        clicks=np.array([0, 1, 0], dtype=np.int64),
    )
    assert step.slate.tolist() == [1, 2, 3]
    assert step.clicks.tolist() == [0, 1, 0]


def test_recobs_legacy_constructor_still_works():
    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.zeros((10, 3)),
        candidate_ids=np.arange(10, dtype=np.int64),
    )
    assert obs.history == ()
    assert obs.logged_action is None
    assert obs.logged_clicks is None


def test_recobs_accepts_history_and_logged_fields():
    h = (HistoryStep(np.array([0]), np.array([1])),)
    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.zeros((10, 3)),
        candidate_ids=np.arange(10, dtype=np.int64),
        history=h,
        logged_action=np.array([1, 2, 3]),
        logged_clicks=np.array([0, 1, 0]),
    )
    assert len(obs.history) == 1
    assert obs.logged_action.tolist() == [1, 2, 3]
    assert obs.logged_clicks.tolist() == [0, 1, 0]
