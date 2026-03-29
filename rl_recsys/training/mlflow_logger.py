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
    except ImportError as exc:
        raise RuntimeError(
            "mlflow logging is enabled but the mlflow package is not installed."
        ) from exc

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    run = mlflow.start_run(run_name=cfg.mlflow.run_name)
    mlflow.log_params(_flatten(asdict(cfg)))
    if cfg.mlflow.tags:
        mlflow.set_tags(cfg.mlflow.tags)
    return run


def log_mlflow_metrics(run: Any, metrics: dict[str, float], *, step: int) -> None:
    if run is None:
        return

    import mlflow

    mlflow.log_metrics(metrics, step=step)


def finish_mlflow(
    run: Any,
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
        artifact_root = Path("outputs") / "mlflow-artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        history_path = artifact_root / f"{run.info.run_id}-history.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(history_path), artifact_path=artifact_path)

    mlflow.end_run()
