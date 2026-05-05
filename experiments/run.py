"""Unified benchmark runner — works with all env types and agents.

Replaces experiments/run_synthetic.py and experiments/run_dataset_bandit.py.

Single run (MLflow disabled by default):
    python experiments/run.py env=synthetic agent=linucb

Single run with MLflow tracking:
    python experiments/run.py mlflow=local env=kuairec agent=linucb

Full benchmark matrix — 8 runs:
    python experiments/run.py --multirun \\
        env=synthetic,kuairec,finn_no_slate,rl4rs \\
        agent=random,linucb

View results:
    mlflow ui --backend-store-uri sqlite:///results/rl-recsys/mlflow/mlflow.db
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from rl_recsys.agents import build_agent
from rl_recsys.config import EnvConfig, to_experiment_config
from rl_recsys.environments.factory import build_env
from rl_recsys.training.trainer import train


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
OmegaConf.register_new_resolver("workspace_root", lambda: str(WORKSPACE_ROOT), replace=True)
OmegaConf.register_new_resolver("workspace_run_id", lambda: WORKSPACE_RUN_ID, replace=True)


@hydra.main(version_base="1.3", config_path="../conf", config_name="train")
def main(cfg: DictConfig) -> None:
    env = build_env(cfg.env)

    # Build EnvConfig from live env properties so dataset envs work correctly.
    # We also patch cfg.env to contain only the fields known to EnvConfig —
    # env yamls include extra keys like 'type' and 'processed_dir' that
    # to_experiment_config() would reject during strict OmegaConf merging.
    env_cfg = EnvConfig(
        slate_size=env.slate_size,
        num_candidates=env.num_candidates,
        user_dim=env.user_dim,
        item_dim=env.item_dim,
        num_items=int(cfg.env.get("num_items", 1000)),
        position_bias_decay=float(cfg.env.get("position_bias_decay", 0.5)),
    )
    cfg_for_exp = OmegaConf.merge(
        cfg,
        OmegaConf.create({"env": OmegaConf.structured(env_cfg)}),
    )
    exp_cfg = to_experiment_config(cfg_for_exp)
    agent = build_agent(exp_cfg.agent, env_cfg)

    env_type = str(cfg.env["type"])
    print(f"\nenv={env_type}  agent={exp_cfg.agent.name}  episodes={exp_cfg.train.num_episodes}")

    history = train(env, agent, exp_cfg)
    avg_reward = float(np.mean([h["reward"] for h in history]))
    print(f"Done. avg_reward={avg_reward:.4f}")


if __name__ == "__main__":
    main()
