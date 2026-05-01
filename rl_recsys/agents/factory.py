from __future__ import annotations

from rl_recsys.agents.base import Agent
from rl_recsys.agents.linucb import LinUCBAgent
from rl_recsys.agents.random import RandomAgent
from rl_recsys.config import AgentConfig, EnvConfig


def build_agent(agent_cfg: AgentConfig, env_cfg: EnvConfig) -> Agent:
    """Build an agent from structured experiment config."""
    name = agent_cfg.name.lower()
    if name == "random":
        return RandomAgent(slate_size=env_cfg.slate_size)
    if name == "linucb":
        return LinUCBAgent(
            slate_size=env_cfg.slate_size,
            user_dim=env_cfg.user_dim,
            item_dim=env_cfg.item_dim,
            alpha=agent_cfg.alpha,
        )
    raise ValueError(f"Unknown agent: {agent_cfg.name}")
