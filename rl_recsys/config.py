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
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


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
