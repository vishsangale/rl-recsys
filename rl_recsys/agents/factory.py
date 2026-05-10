# rl_recsys/agents/factory.py
from __future__ import annotations

from typing import Callable

from rl_recsys.agents.base import Agent
from rl_recsys.agents.eps_greedy_linear import EpsGreedyLinearAgent
from rl_recsys.agents.lin_ts import LinTSAgent
from rl_recsys.agents.linucb import LinUCBAgent
from rl_recsys.agents.logged_replay import LoggedReplayAgent
from rl_recsys.agents.most_popular import MostPopularAgent
from rl_recsys.agents.oracle_click import OracleClickAgent
from rl_recsys.agents.random import RandomAgent
from rl_recsys.config import AgentConfig, EnvConfig

AgentBuilder = Callable[[AgentConfig, EnvConfig], Agent]


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


AGENT_REGISTRY: dict[str, AgentBuilder] = {
    "eps_greedy_linear": _build_eps_greedy_linear,
    "lin_ts": _build_lin_ts,
    "linucb": _build_linucb,
    "logged_replay": _build_logged_replay,
    "most_popular": _build_most_popular,
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
