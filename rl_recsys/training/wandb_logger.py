from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
from typing import Any

from rl_recsys.config import ExperimentConfig


def init_wandb(cfg: ExperimentConfig):
    if not cfg.wandb.enabled:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb logging is enabled but the wandb package is not installed."
        ) from exc

    if cfg.wandb.base_url:
        os.environ["WANDB_BASE_URL"] = cfg.wandb.base_url

    Path(cfg.runtime.wandb_dir).mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        id=cfg.runtime.workspace_run_id,
        name=cfg.runtime.workspace_run_id,
        resume="allow",
        group=cfg.wandb.group,
        job_type=cfg.wandb.job_type,
        dir=cfg.runtime.wandb_dir,
        tags=cfg.wandb.tags,
        config=asdict(cfg),
    )


def log_wandb_metrics(run: Any, metrics: dict[str, float]) -> None:
    if run is None:
        return
    run.log(metrics)


def finish_wandb(run: Any, summary: dict[str, float] | None = None) -> None:
    if run is None:
        return
    if summary:
        for key, value in summary.items():
            run.summary[key] = value
    run.finish()
