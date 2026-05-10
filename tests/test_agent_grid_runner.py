from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rl_recsys.environments.base import RecObs
from rl_recsys.evaluation.ope_trajectory import LoggedTrajectoryStep
from rl_recsys.training.agent_grid_runner import GridRun, run_grid


class _FakeSource:
    def __init__(self, num_candidates=8, slate_size=3):
        self.num_candidates = num_candidates
        self.slate_size = slate_size
        self._candidate_features = np.eye(num_candidates, 3)

    def iter_trajectories(self, *, max_trajectories=None, seed=0):
        obs = RecObs(
            user_features=np.zeros(4),
            candidate_features=self._candidate_features,
            candidate_ids=np.arange(self.num_candidates, dtype=np.int64),
            logged_action=np.array([0, 1, 2], dtype=np.int64),
            logged_clicks=np.array([1, 0, 1], dtype=np.int64),
        )
        step = LoggedTrajectoryStep(
            obs=obs,
            logged_action=np.array([0, 1, 2], dtype=np.int64),
            logged_reward=2.0,
            logged_clicks=np.array([1, 0, 1], dtype=np.int64),
            propensity=0.1,
        )
        for _ in range(2):
            yield [step, step]


def test_run_grid_writes_one_artifact_per_run(tmp_path):
    runs = [
        GridRun(agent_name="random", seed=0, pretrained=False),
        GridRun(agent_name="random", seed=1, pretrained=False),
    ]
    src = _FakeSource()
    written = run_grid(
        runs,
        train_source_factory=lambda seed: src,
        eval_source_factory=lambda seed: src,
        env_kwargs=dict(slate_size=3, user_dim=4, item_dim=3, num_candidates=8),
        output_dir=tmp_path,
        max_trajectories=2,
    )
    assert len(written) == 2
    for path in written:
        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        assert data["agent"] == "random"
        assert "metrics" in data


def test_run_grid_resume_skips_existing(tmp_path):
    src = _FakeSource()
    runs = [GridRun(agent_name="random", seed=0, pretrained=False)]
    run_grid(
        runs,
        train_source_factory=lambda s: src,
        eval_source_factory=lambda s: src,
        env_kwargs=dict(slate_size=3, user_dim=4, item_dim=3, num_candidates=8),
        output_dir=tmp_path, max_trajectories=2,
    )
    written = run_grid(
        runs,
        train_source_factory=lambda s: src,
        eval_source_factory=lambda s: src,
        env_kwargs=dict(slate_size=3, user_dim=4, item_dim=3, num_candidates=8),
        output_dir=tmp_path, max_trajectories=2, resume=True,
    )
    assert written == []


def test_run_grid_failed_run_writes_failed_json(tmp_path):
    """logged_replay against a source whose obs HAS logged_action works fine.
    To trigger a failure, point at an unknown agent name."""
    src = _FakeSource()
    runs = [GridRun(agent_name="not_a_real_agent", seed=0, pretrained=False)]
    written = run_grid(
        runs,
        train_source_factory=lambda s: src,
        eval_source_factory=lambda s: src,
        env_kwargs=dict(slate_size=3, user_dim=4, item_dim=3, num_candidates=8),
        output_dir=tmp_path, max_trajectories=2,
    )
    assert written == []
    failed = list(Path(tmp_path).glob("*.failed.json"))
    assert len(failed) == 1
    data = json.loads(failed[0].read_text())
    assert "error" in data
