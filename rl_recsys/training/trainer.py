from __future__ import annotations

import numpy as np

from rl_recsys.agents.base import Agent
from rl_recsys.config import ExperimentConfig
from rl_recsys.environments.base import RecEnv
from rl_recsys.runtime import (
    TrackingIds,
    ensure_runtime_dirs,
    now_timestamp,
    write_project_manifest,
    write_run_manifest,
)
from rl_recsys.training.metrics import ctr, mrr, ndcg_at_k
from rl_recsys.training.mlflow_logger import finish_mlflow, init_mlflow, log_mlflow_metrics
from rl_recsys.training.wandb_logger import finish_wandb, init_wandb, log_wandb_metrics


def train(env: RecEnv, agent: Agent, cfg: ExperimentConfig) -> list[dict[str, float]]:
    """Main training loop.

    Returns a list of per-episode metric dicts.
    """
    rng = np.random.default_rng(cfg.train.seed)
    history: list[dict[str, float]] = []
    started_at = now_timestamp()
    ensure_runtime_dirs(cfg)
    write_project_manifest(cfg)
    wandb_run = init_wandb(cfg)
    mlflow_run = init_mlflow(cfg)
    tracking_ids = TrackingIds(
        workspace_run_id=cfg.runtime.workspace_run_id,
        wandb_run_id=getattr(wandb_run, "id", None),
        mlflow_run_id=getattr(getattr(mlflow_run, "info", None), "run_id", None),
    )
    write_run_manifest(cfg, tracking_ids=tracking_ids, status="running", started_at=started_at)

    for ep in range(cfg.train.num_episodes):
        obs = env.reset(seed=int(rng.integers(0, 2**31)))
        episode_rewards: list[float] = []
        episode_clicks: list[np.ndarray] = []

        # single-step episodes (re-rank a fresh candidate set each episode)
        slate = agent.select_slate(obs)
        step = env.step(slate)

        agent.update(obs, slate, step.reward, step.clicks, step.obs)

        episode_rewards.append(step.reward)
        episode_clicks.append(step.clicks)

        all_clicks = np.concatenate(episode_clicks)
        metrics = {
            "episode": float(ep),
            "reward": float(np.sum(episode_rewards)),
            "ndcg": ndcg_at_k(all_clicks),
            "mrr": mrr(all_clicks),
            "ctr": ctr(all_clicks),
        }
        history.append(metrics)
        log_wandb_metrics(wandb_run, metrics)
        log_mlflow_metrics(mlflow_run, metrics, step=ep)

        if ep % cfg.train.log_every == 0:
            print(
                f"[ep {ep:4d}] reward={metrics['reward']:.2f}  "
                f"ndcg={metrics['ndcg']:.3f}  ctr={metrics['ctr']:.3f}"
            )

    if history:
        summary = {
            "avg_reward": float(np.mean([entry["reward"] for entry in history])),
            "final_ctr": float(history[-1]["ctr"]),
        }
        finish_wandb(
            wandb_run,
            summary=summary,
        )
        finish_mlflow(
            mlflow_run,
            cfg=cfg,
            summary=summary,
            history=history,
            artifact_path=cfg.mlflow.artifact_path,
        )
    else:
        finish_wandb(wandb_run)
        finish_mlflow(mlflow_run, cfg=cfg)

    write_run_manifest(
        cfg,
        tracking_ids=tracking_ids,
        status="finished",
        started_at=started_at,
        finished_at=now_timestamp(),
    )

    return history
