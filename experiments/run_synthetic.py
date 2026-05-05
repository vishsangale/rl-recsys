# DEPRECATED: Use experiments/run.py instead.
# Equivalent: python experiments/run.py env=synthetic agent=<agent>
"""Run an agent baseline on the synthetic environment."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from rl_recsys.agents import build_agent
from rl_recsys.config import ExperimentConfig, to_experiment_config
from rl_recsys.environments.synthetic import SyntheticEnv
from rl_recsys.training.trainer import train


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
OmegaConf.register_new_resolver("workspace_root", lambda: str(WORKSPACE_ROOT), replace=True)
OmegaConf.register_new_resolver("workspace_run_id", lambda: WORKSPACE_RUN_ID, replace=True)


@hydra.main(version_base="1.3", config_path="../conf", config_name="train")
def main(cfg: DictConfig) -> None:
    exp_cfg: ExperimentConfig = to_experiment_config(cfg)

    env = SyntheticEnv(exp_cfg.env)
    agent = build_agent(exp_cfg.agent, exp_cfg.env)

    history = train(env, agent, exp_cfg)
    avg_reward = np.mean([h["reward"] for h in history])
    print(f"\nDone. Average reward over {len(history)} episodes: {avg_reward:.3f}")


if __name__ == "__main__":
    main()
