from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf


@dataclass
class EnvConfig:
    num_items: int = 1000
    num_candidates: int = 50
    slate_size: int = 10
    user_dim: int = 32
    item_dim: int = 32
    position_bias_decay: float = 0.5


@dataclass
class AgentConfig:
    name: str = "random"
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon: float = 0.1
    # LinUCB
    alpha: float = 1.0
    # Replay buffer
    buffer_size: int = 10000
    batch_size: int = 64


@dataclass
class TrainConfig:
    num_episodes: int = 500
    eval_every: int = 50
    log_every: int = 10
    seed: int = 42


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "rl-recsys"
    entity: str | None = None
    mode: str = "offline"
    base_url: str | None = None
    group: str | None = None
    job_type: str = "train"
    dir: str = "wandb"
    tags: list[str] = field(default_factory=list)


@dataclass
class MlflowConfig:
    enabled: bool = False
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "rl-recsys"
    run_name: str | None = None
    artifact_path: str = "artifacts"
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)


def to_experiment_config(
    raw_cfg: DictConfig | Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    """Merge a Hydra/OmegaConf config into the structured experiment dataclasses."""
    structured = OmegaConf.structured(ExperimentConfig)
    merged = OmegaConf.merge(structured, raw_cfg or {})
    cfg = OmegaConf.to_object(merged)
    if not isinstance(cfg, ExperimentConfig):
        raise TypeError(f"Expected ExperimentConfig, got {type(cfg)!r}")
    return cfg
