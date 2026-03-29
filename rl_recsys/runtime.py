from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from rl_recsys.config import ExperimentConfig


@dataclass
class TrackingIds:
    workspace_run_id: str
    wandb_run_id: str | None = None
    mlflow_run_id: str | None = None


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def ensure_runtime_dirs(cfg: ExperimentConfig) -> None:
    for path_value in (
        cfg.runtime.results_root,
        cfg.runtime.project_results_dir,
        cfg.runtime.run_dir,
        cfg.runtime.hydra_dir,
        cfg.runtime.wandb_dir,
        cfg.runtime.tb_dir,
        cfg.runtime.mlflow_dir,
        cfg.runtime.logs_dir,
        cfg.runtime.checkpoints_dir,
        cfg.runtime.exports_dir,
        str(Path(cfg.runtime.project_results_dir) / "mlflow"),
    ):
        Path(path_value).mkdir(parents=True, exist_ok=True)


def write_project_manifest(cfg: ExperimentConfig) -> None:
    payload = {
        "project_name": cfg.runtime.project_name,
        "repo_root": cfg.runtime.repo_root,
        "workspace_root": cfg.runtime.workspace_root,
        "results_root": cfg.runtime.results_root,
        "project_results_dir": cfg.runtime.project_results_dir,
        "sources": {
            "mlflow": {
                "tracking_uri": cfg.mlflow.tracking_uri,
                "experiment_name": cfg.mlflow.experiment_name,
            },
            "wandb_offline": {
                "paths": [str(Path(cfg.runtime.project_results_dir) / "runs")],
                "project": cfg.wandb.project,
            },
            "tensorboard": {
                "paths": [str(Path(cfg.runtime.project_results_dir) / "runs")],
            },
        },
    }
    _write_yaml(Path(cfg.runtime.project_manifest_path), payload)


def write_run_manifest(
    cfg: ExperimentConfig,
    *,
    tracking_ids: TrackingIds,
    status: str,
    started_at: str,
    finished_at: str | None = None,
) -> None:
    payload = {
        "workspace_run_id": tracking_ids.workspace_run_id,
        "project_name": cfg.runtime.project_name,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "repo_root": cfg.runtime.repo_root,
        "paths": {
            "run_dir": cfg.runtime.run_dir,
            "hydra_dir": cfg.runtime.hydra_dir,
            "wandb_dir": cfg.runtime.wandb_dir,
            "tb_dir": cfg.runtime.tb_dir,
            "mlflow_dir": cfg.runtime.mlflow_dir,
            "logs_dir": cfg.runtime.logs_dir,
            "checkpoints_dir": cfg.runtime.checkpoints_dir,
            "exports_dir": cfg.runtime.exports_dir,
        },
        "tracking": {
            "mlflow_tracking_uri": cfg.mlflow.tracking_uri,
            "mlflow_experiment_name": cfg.mlflow.experiment_name,
            "mlflow_run_id": tracking_ids.mlflow_run_id,
            "wandb_project": cfg.wandb.project,
            "wandb_run_id": tracking_ids.wandb_run_id,
        },
        "config": asdict(cfg),
    }
    _write_yaml(Path(cfg.runtime.run_manifest_path), payload)


def now_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
