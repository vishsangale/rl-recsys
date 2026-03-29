from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from rl_recsys.config import ExperimentConfig


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in data.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, dotted))
        elif isinstance(value, list):
            flat[dotted] = json.dumps(value)
        elif value is None:
            flat[dotted] = "null"
        else:
            flat[dotted] = str(value)
    return flat


def init_mlflow(cfg: ExperimentConfig):
    if not cfg.mlflow.enabled:
        return None

    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise RuntimeError(
            "mlflow logging is enabled but the mlflow package is not installed."
        ) from exc

    artifact_base = Path(cfg.runtime.project_results_dir) / "mlflow" / "artifacts"
    artifact_base.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    client = MlflowClient(tracking_uri=cfg.mlflow.tracking_uri)
    experiment = client.get_experiment_by_name(cfg.mlflow.experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            cfg.mlflow.experiment_name,
            artifact_location=artifact_base.resolve().as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id

    run = mlflow.start_run(
        experiment_id=experiment_id,
        run_name=cfg.mlflow.run_name or cfg.runtime.workspace_run_id,
    )
    mlflow.log_params(_flatten(asdict(cfg)))
    mlflow.set_tags(
        {
            **cfg.mlflow.tags,
            "workspace.project_name": cfg.runtime.project_name,
            "workspace.repo_root": cfg.runtime.repo_root,
            "workspace.results_root": cfg.runtime.results_root,
            "workspace.run_id": cfg.runtime.workspace_run_id,
            "workspace.run_dir": cfg.runtime.run_dir,
            "workspace.mlflow_dir": cfg.runtime.mlflow_dir,
            "workspace.hydra_dir": cfg.runtime.hydra_dir,
            "workspace.wandb_dir": cfg.runtime.wandb_dir,
            "workspace.tb_dir": cfg.runtime.tb_dir,
        }
    )
    return run


def log_mlflow_metrics(run: Any, metrics: dict[str, float], *, step: int) -> None:
    if run is None:
        return

    import mlflow

    mlflow.log_metrics(metrics, step=step)


def finish_mlflow(
    run: Any,
    cfg: ExperimentConfig | None = None,
    *,
    summary: dict[str, float] | None = None,
    history: list[dict[str, float]] | None = None,
    artifact_path: str = "artifacts",
) -> None:
    if run is None:
        return

    import mlflow

    if summary:
        mlflow.log_metrics(summary)

    if history:
        artifact_root = Path(cfg.runtime.mlflow_dir) if cfg is not None else (Path("outputs") / "mlflow-artifacts")
        artifact_root.mkdir(parents=True, exist_ok=True)
        history_path = artifact_root / f"{run.info.run_id}-history.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(history_path), artifact_path=artifact_path)

    mlflow.end_run()
