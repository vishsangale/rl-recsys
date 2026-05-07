from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable

import numpy as np

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecEnv
from rl_recsys.evaluation.trajectory import (
    TrajectoryDataset,
    evaluate_trajectory_agent,
)
from rl_recsys.evaluation.bandit import evaluate_bandit_agent


@dataclass
class VarianceEvaluation:
    mean: dict[str, float]
    std: dict[str, float]
    n_seeds: int


def _aggregate_runs(
    results: list[Any],
) -> tuple[dict[str, float], dict[str, float]]:
    """Mean and std over scalar (int/float) fields shared by all dataclass results.

    Skips fields tagged with `metadata={"aggregate": False}` — used to exclude
    run-config (episodes, sessions) and runtime (seconds) from metric output.
    np.std uses ddof=0 (population std).
    """
    if not results:
        return {}, {}
    numeric_keys: list[str] = []
    for f in fields(results[0]):
        if f.metadata.get("aggregate") is False:
            continue
        ftype = f.type
        if ftype in (float, int) or ftype in ("float", "int"):
            numeric_keys.append(f.name)
    runs = {k: [getattr(r, k) for r in results] for k in numeric_keys}
    mean = {k: float(np.mean(v)) for k, v in runs.items()}
    std = {k: float(np.std(v)) for k, v in runs.items()}
    return mean, std


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
    """Run evaluate_bandit_agent n_seeds times; return mean ± std per metric.

    make_env and make_agent are called fresh each seed to prevent state leakage.
    Default n_seeds=5 matches the RL4RS paper's reporting convention.
    """
    results = [
        evaluate_bandit_agent(
            make_env(),
            make_agent(),
            agent_name=agent_name,
            episodes=episodes,
            seed=base_seed + i,
            gamma=gamma,
        )
        for i in range(n_seeds)
    ]
    mean, std = _aggregate_runs(results)
    return VarianceEvaluation(mean=mean, std=std, n_seeds=n_seeds)


def evaluate_trajectory_with_variance(
    make_dataset: Callable[[], TrajectoryDataset],
    make_agent: Callable[[], Agent],
    *,
    agent_name: str,
    max_sessions: int,
    n_seeds: int = 5,
    base_seed: int = 42,
    gamma: float = 0.95,
) -> VarianceEvaluation:
    """Run evaluate_trajectory_agent n_seeds times; return mean ± std per metric."""
    results = [
        evaluate_trajectory_agent(
            make_dataset(),
            make_agent(),
            agent_name=agent_name,
            max_sessions=max_sessions,
            seed=base_seed + i,
            gamma=gamma,
        )
        for i in range(n_seeds)
    ]
    mean, std = _aggregate_runs(results)
    return VarianceEvaluation(mean=mean, std=std, n_seeds=n_seeds)
