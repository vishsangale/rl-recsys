from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_recsys.agents.base import Agent
    from rl_recsys.evaluation.ope_trajectory import LoggedTrajectorySource


def pretrain_agent_on_logged(
    agent: "Agent",
    source: "LoggedTrajectorySource",
    *,
    max_trajectories: int | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Single offline pass over logged trajectories, calling agent.update().

    For each LoggedTrajectoryStep, calls
        agent.update(
            obs=step.obs,
            slate=step.logged_action,
            reward=step.logged_reward,
            clicks=step.logged_clicks,
            next_obs=step.obs,
        )

    `next_obs == obs` because LinUCB ignores it (contextual bandit) and
    Random ignores everything; we don't fabricate state evolution.

    Returns aggregate metrics: trajectories, total_steps, mean_click_rate,
    seconds.

    Raises ValueError if source yields zero trajectories.
    """
    started = perf_counter()
    n_traj = 0
    n_steps = 0
    total_clicks = 0.0
    total_slate_positions = 0

    for traj in source.iter_trajectories(
        max_trajectories=max_trajectories, seed=seed
    ):
        if not traj:
            continue
        n_traj += 1
        for step in traj:
            agent.update(
                obs=step.obs,
                slate=step.logged_action,
                reward=step.logged_reward,
                clicks=step.logged_clicks,
                next_obs=step.obs,
            )
            n_steps += 1
            total_clicks += float(step.logged_clicks.sum())
            total_slate_positions += int(step.logged_clicks.shape[0])

    if n_traj == 0:
        raise ValueError("source produced zero trajectories")

    return {
        "trajectories": float(n_traj),
        "total_steps": float(n_steps),
        "mean_click_rate": (
            total_clicks / total_slate_positions if total_slate_positions else 0.0
        ),
        "seconds": float(perf_counter() - started),
    }
