from __future__ import annotations

import numpy as np
import pytest

from rl_recsys.agents.random import RandomAgent
from rl_recsys.config import (
    AgentConfig, EnvConfig, ExperimentConfig,
    MlflowConfig, RuntimeConfig, TrainConfig, WandbConfig,
)
from rl_recsys.environments.base import RecEnv, RecObs, RecStep
from rl_recsys.training.trainer import train


def _minimal_cfg(tmp_path) -> ExperimentConfig:
    base = str(tmp_path)
    return ExperimentConfig(
        env=EnvConfig(),
        agent=AgentConfig(),
        train=TrainConfig(num_episodes=1, log_every=1, seed=0),
        wandb=WandbConfig(enabled=False),
        mlflow=MlflowConfig(enabled=False),
        runtime=RuntimeConfig(
            repo_root=base,
            workspace_root=base,
            results_root=base,
            project_results_dir=base,
            run_dir=base,
            hydra_dir=base,
            wandb_dir=base,
            tb_dir=base,
            mlflow_dir=base,
            logs_dir=base,
            checkpoints_dir=base,
            exports_dir=base,
            mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
            project_manifest_path=str(tmp_path / "project.yaml"),
            run_manifest_path=str(tmp_path / "run.yaml"),
        ),
    )


def _obs() -> RecObs:
    return RecObs(
        user_features=np.zeros(4, dtype=np.float32),
        candidate_features=np.zeros((3, 4), dtype=np.float32),
        candidate_ids=np.arange(3),
    )


class _ThreeStepEnv(RecEnv):
    """Session env that requires 3 steps per episode, reward=1.0 each step."""

    def __init__(self):
        self._steps = 0

    @property
    def slate_size(self) -> int: return 2
    @property
    def num_candidates(self) -> int: return 3
    @property
    def user_dim(self) -> int: return 4
    @property
    def item_dim(self) -> int: return 4

    def reset(self, seed: int | None = None) -> RecObs:
        self._steps = 0
        return _obs()

    def step(self, slate: np.ndarray) -> RecStep:
        self._steps += 1
        return RecStep(obs=_obs(), reward=1.0, clicks=np.array([1, 0]), done=self._steps >= 3)


def test_trainer_single_step_env_still_works(tmp_path):
    """Bandit env (done=True on first step) works after loop change."""
    class _BanditEnv(_ThreeStepEnv):
        def step(self, slate):
            return RecStep(obs=_obs(), reward=2.0, clicks=np.array([1, 0]), done=True)

    history = train(_BanditEnv(), RandomAgent(slate_size=2, seed=0), _minimal_cfg(tmp_path))
    assert history[0]["reward"] == pytest.approx(2.0)


def test_trainer_multi_step_accumulates_all_rewards(tmp_path):
    """Session env: reward must sum across 3 steps."""
    history = train(_ThreeStepEnv(), RandomAgent(slate_size=2, seed=0), _minimal_cfg(tmp_path))
    assert history[0]["reward"] == pytest.approx(3.0)


def test_trainer_returns_one_entry_per_episode(tmp_path):
    """history length equals num_episodes regardless of steps per episode."""
    cfg = _minimal_cfg(tmp_path)
    cfg.train.num_episodes = 4
    history = train(_ThreeStepEnv(), RandomAgent(slate_size=2, seed=0), cfg)
    assert len(history) == 4
