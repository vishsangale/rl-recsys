from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator, Protocol

import numpy as np

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecObs
from rl_recsys.training.metrics import ctr, discounted_return, mrr, ndcg_at_k


@dataclass(frozen=True)
class TrajectoryStep:
    obs: RecObs
    logged_slate: np.ndarray  # (slate_size,) item IDs the user actually saw — provenance only, not used by the replay reward rule
    logged_clicked_id: int
    logged_reward: float


@dataclass(frozen=True)
class Session:
    session_id: int
    steps: list[TrajectoryStep]


class TrajectoryDataset(Protocol):
    def iter_sessions(
        self, *, max_sessions: int | None = None, seed: int | None = None
    ) -> Iterator[Session]:
        ...


@dataclass
class TrajectoryEvaluation:
    agent: str
    sessions: int = field(metadata={"aggregate": False})
    total_steps: int = field(metadata={"aggregate": False})
    avg_session_reward: float
    avg_discounted_return: float
    avg_session_length: float
    avg_session_hit_rate: float
    avg_per_step_ctr: float
    avg_per_step_ndcg: float
    avg_per_step_mrr: float
    seconds: float = field(metadata={"aggregate": False})

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "agent": self.agent,
            "sessions": self.sessions,
            "total_steps": self.total_steps,
            "avg_session_reward": self.avg_session_reward,
            "avg_discounted_return": self.avg_discounted_return,
            "avg_session_length": self.avg_session_length,
            "avg_session_hit_rate": self.avg_session_hit_rate,
            "avg_per_step_ctr": self.avg_per_step_ctr,
            "avg_per_step_ndcg": self.avg_per_step_ndcg,
            "avg_per_step_mrr": self.avg_per_step_mrr,
            "seconds": self.seconds,
        }


def evaluate_trajectory_agent(
    dataset: TrajectoryDataset,
    agent: Agent,
    *,
    agent_name: str,
    max_sessions: int,
    seed: int,
    gamma: float = 0.95,
) -> TrajectoryEvaluation:
    """Replay-mode trajectory evaluator.

    For each step, the agent picks a slate from the candidate pool. Reward
    equals logged_reward if the agent's slate covers logged_clicked_id;
    otherwise zero. agent.update() is NOT called — the agent's state is
    frozen for the duration of evaluation.
    """
    if max_sessions <= 0:
        raise ValueError("max_sessions must be positive")

    started = perf_counter()
    session_rewards: list[float] = []
    session_disc_returns: list[float] = []
    session_lengths: list[int] = []
    session_hits: list[float] = []
    per_step_ctrs: list[float] = []
    per_step_ndcgs: list[float] = []
    per_step_mrrs: list[float] = []
    total_steps = 0

    for session in dataset.iter_sessions(max_sessions=max_sessions, seed=seed):
        rewards_per_step: list[float] = []
        for step in session.steps:
            slate_indices = np.asarray(agent.select_slate(step.obs), dtype=np.int64)
            slate_ids = step.obs.candidate_ids[slate_indices]
            covered = (
                step.logged_clicked_id != -1
                and bool(np.any(slate_ids == step.logged_clicked_id))
            )
            if covered:
                clicks = (slate_ids == step.logged_clicked_id).astype(np.float64)
                r = float(step.logged_reward)
            else:
                clicks = np.zeros(len(slate_indices), dtype=np.float64)
                r = 0.0
            rewards_per_step.append(r)
            per_step_ctrs.append(ctr(clicks))
            per_step_ndcgs.append(ndcg_at_k(clicks))
            per_step_mrrs.append(mrr(clicks))
            total_steps += 1
        rewards_arr = np.asarray(rewards_per_step, dtype=np.float64)
        session_rewards.append(float(rewards_arr.sum()))
        session_disc_returns.append(discounted_return(rewards_arr, gamma=gamma))
        session_lengths.append(len(session.steps))
        session_hits.append(float(rewards_arr.sum() > 0.0))

    sessions_count = len(session_rewards)
    if sessions_count == 0:
        raise ValueError("dataset produced zero sessions")

    return TrajectoryEvaluation(
        agent=agent_name,
        sessions=sessions_count,
        total_steps=total_steps,
        avg_session_reward=float(np.mean(session_rewards)),
        avg_discounted_return=float(np.mean(session_disc_returns)),
        avg_session_length=float(np.mean(session_lengths)),
        avg_session_hit_rate=float(np.mean(session_hits)),
        avg_per_step_ctr=float(np.mean(per_step_ctrs)),
        avg_per_step_ndcg=float(np.mean(per_step_ndcgs)),
        avg_per_step_mrr=float(np.mean(per_step_mrrs)),
        seconds=float(perf_counter() - started),
    )
