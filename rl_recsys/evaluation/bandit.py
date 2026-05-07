from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecEnv
from rl_recsys.training.metrics import ctr, discounted_return, mrr, ndcg_at_k


@dataclass
class BanditEvaluation:
    agent: str
    episodes: int
    avg_reward: float
    hit_rate: float
    ctr: float
    ndcg: float
    mrr: float
    discounted_return: float
    seconds: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "agent": self.agent,
            "episodes": self.episodes,
            "avg_reward": self.avg_reward,
            "hit_rate": self.hit_rate,
            "ctr": self.ctr,
            "ndcg": self.ndcg,
            "mrr": self.mrr,
            "discounted_return": self.discounted_return,
            "seconds": self.seconds,
        }


def evaluate_bandit_agent(
    env: RecEnv,
    agent: Agent,
    *,
    agent_name: str,
    episodes: int,
    seed: int,
    gamma: float = 0.95,
) -> BanditEvaluation:
    # gamma has no effect in bandit mode: single-step episodes mean gamma^0=1 always.
    # Accepted for API parity with multi-step evaluate_* functions.
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    hits: list[float] = []
    ctrs: list[float] = []
    ndcgs: list[float] = []
    mrrs: list[float] = []
    disc_returns: list[float] = []
    started = perf_counter()

    for _ in range(episodes):
        obs = env.reset(seed=int(rng.integers(0, 2**31)))
        slate = agent.select_slate(obs)
        step = env.step(slate)
        agent.update(obs, slate, step.reward, step.clicks, step.obs)
        rewards.append(step.reward)
        hits.append(float(step.reward > 0.0))
        ctrs.append(ctr(step.clicks))
        ndcgs.append(ndcg_at_k(step.clicks))
        mrrs.append(mrr(step.clicks))
        disc_returns.append(discounted_return(np.array([step.reward]), gamma=gamma))

    seconds = perf_counter() - started
    return BanditEvaluation(
        agent=agent_name,
        episodes=episodes,
        avg_reward=float(np.mean(rewards)),
        hit_rate=float(np.mean(hits)),
        ctr=float(np.mean(ctrs)),
        ndcg=float(np.mean(ndcgs)),
        mrr=float(np.mean(mrrs)),
        discounted_return=float(np.mean(disc_returns)),
        seconds=float(seconds),
    )
