from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecEnv
from rl_recsys.evaluation.bandit import evaluate_bandit_agent

_SCALAR_KEYS = ("avg_reward", "hit_rate", "ctr", "ndcg", "mrr", "discounted_return")


@dataclass
class VarianceEvaluation:
    mean: dict[str, float]
    std: dict[str, float]
    n_seeds: int


def evaluate_with_variance(
    make_env: Callable[[], RecEnv],
    make_agent: Callable[[], Agent],
    *,
    agent_name: str,
    episodes: int,
    n_seeds: int = 5,
    base_seed: int = 42,
    gamma: float = 0.95,
) -> VarianceEvaluation:
    """Run evaluate_bandit_agent n_seeds times and return mean ± std per metric.

    make_env and make_agent are called fresh each seed to prevent state leakage.
    Default n_seeds=5 matches the RL4RS paper's reporting convention.
    """
    runs: dict[str, list[float]] = {k: [] for k in _SCALAR_KEYS}

    for i in range(n_seeds):
        env = make_env()
        agent = make_agent()
        result = evaluate_bandit_agent(
            env,
            agent,
            agent_name=agent_name,
            episodes=episodes,
            seed=base_seed + i,
            gamma=gamma,
        )
        for k in _SCALAR_KEYS:
            runs[k].append(getattr(result, k))

    return VarianceEvaluation(
        mean={k: float(np.mean(v)) for k, v in runs.items()},
        std={k: float(np.std(v)) for k, v in runs.items()},
        n_seeds=n_seeds,
    )
