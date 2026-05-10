# rl_recsys/agents/factory.py
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from rl_recsys.agents.base import Agent
from rl_recsys.agents.bc import BCAgent
from rl_recsys.agents.boltzmann_linear import BoltzmannLinearAgent
from rl_recsys.agents.gbdt import GBDTAgent
from rl_recsys.agents.eps_greedy_linear import EpsGreedyLinearAgent
from rl_recsys.agents.lin_ts import LinTSAgent
from rl_recsys.agents.linucb import LinUCBAgent
from rl_recsys.agents.logged_replay import LoggedReplayAgent
from rl_recsys.agents.most_popular import MostPopularAgent
from rl_recsys.agents.neural_linear import NeuralLinearAgent
from rl_recsys.agents.oracle_click import OracleClickAgent
from rl_recsys.agents.random import RandomAgent
from rl_recsys.config import AgentConfig, EnvConfig

AgentBuilder = Callable[[AgentConfig, EnvConfig], Agent]


def _safe_device(agent_cfg: AgentConfig) -> str:
    """Downgrade 'cuda' to 'cpu' when CUDA isn't available so factory builders
    work both on dev boxes and CI."""
    requested = getattr(agent_cfg, "device", "cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def _build_random(agent_cfg: AgentConfig, env_cfg: EnvConfig) -> Agent:
    return RandomAgent(slate_size=env_cfg.slate_size)


def _build_linucb(agent_cfg: AgentConfig, env_cfg: EnvConfig) -> Agent:
    return LinUCBAgent(
        slate_size=env_cfg.slate_size,
        user_dim=env_cfg.user_dim,
        item_dim=env_cfg.item_dim,
        alpha=agent_cfg.alpha,
    )


def _build_most_popular(agent_cfg: AgentConfig, env_cfg: EnvConfig) -> Agent:
    return MostPopularAgent(
        slate_size=env_cfg.slate_size,
        num_candidates=env_cfg.num_candidates,
    )


def _build_logged_replay(agent_cfg: AgentConfig, env_cfg: EnvConfig) -> Agent:
    return LoggedReplayAgent(slate_size=env_cfg.slate_size)


def _build_oracle_click(agent_cfg: AgentConfig, env_cfg: EnvConfig) -> Agent:
    return OracleClickAgent(slate_size=env_cfg.slate_size)


def _build_lin_ts(agent_cfg: AgentConfig, env_cfg: EnvConfig) -> Agent:
    return LinTSAgent(
        slate_size=env_cfg.slate_size,
        user_dim=env_cfg.user_dim,
        item_dim=env_cfg.item_dim,
        sigma=getattr(agent_cfg, "sigma", 1.0),
    )


def _build_eps_greedy_linear(agent_cfg: AgentConfig, env_cfg: EnvConfig) -> Agent:
    return EpsGreedyLinearAgent(
        slate_size=env_cfg.slate_size,
        user_dim=env_cfg.user_dim,
        item_dim=env_cfg.item_dim,
        epsilon=getattr(agent_cfg, "epsilon", 0.1),
    )


def _build_boltzmann_linear(agent_cfg: AgentConfig, env_cfg: EnvConfig) -> Agent:
    return BoltzmannLinearAgent(
        slate_size=env_cfg.slate_size,
        user_dim=env_cfg.user_dim,
        item_dim=env_cfg.item_dim,
        temperature=getattr(agent_cfg, "temperature", 1.0),
    )


def _build_neural_linear(agent_cfg: AgentConfig, env_cfg: EnvConfig) -> Agent:
    return NeuralLinearAgent(
        slate_size=env_cfg.slate_size,
        user_dim=env_cfg.user_dim,
        item_dim=env_cfg.item_dim,
        hidden_dim=getattr(agent_cfg, "hidden_dim", 64),
        embedding_dim=getattr(agent_cfg, "embedding_dim", 32),
        mlp_epochs=getattr(agent_cfg, "epochs", 5),
        alpha=getattr(agent_cfg, "alpha", 1.0),
        device=_safe_device(agent_cfg),
    )


def _build_bc(agent_cfg: AgentConfig, env_cfg: EnvConfig) -> Agent:
    # candidate_features and behavior_policy are injected by the runner
    # (Task 20) before train_offline. The factory constructs a stub.
    return BCAgent(
        slate_size=env_cfg.slate_size,
        candidate_features=np.zeros((env_cfg.num_candidates, env_cfg.item_dim)),
    )


def _build_gbdt(agent_cfg: AgentConfig, env_cfg: EnvConfig) -> Agent:
    return GBDTAgent(
        slate_size=env_cfg.slate_size,
        candidate_features=np.zeros((env_cfg.num_candidates, env_cfg.item_dim)),
        n_estimators=getattr(agent_cfg, "n_estimators", 100),
        max_depth=getattr(agent_cfg, "max_depth", 6),
        learning_rate=getattr(agent_cfg, "learning_rate", 0.05),
    )


AGENT_REGISTRY: dict[str, AgentBuilder] = {
    "bc": _build_bc,
    "boltzmann_linear": _build_boltzmann_linear,
    "gbdt": _build_gbdt,
    "eps_greedy_linear": _build_eps_greedy_linear,
    "lin_ts": _build_lin_ts,
    "linucb": _build_linucb,
    "logged_replay": _build_logged_replay,
    "most_popular": _build_most_popular,
    "neural_linear": _build_neural_linear,
    "oracle_click": _build_oracle_click,
    "random": _build_random,
}


def build_agent(agent_cfg: AgentConfig, env_cfg: EnvConfig) -> Agent:
    """Build an agent from structured experiment config via AGENT_REGISTRY."""
    name = agent_cfg.name.lower()
    try:
        builder = AGENT_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown agent: {agent_cfg.name}") from exc
    return builder(agent_cfg, env_cfg)
