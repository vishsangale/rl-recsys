"""Run a random-agent baseline on the synthetic environment."""
from __future__ import annotations

import argparse

import numpy as np

from rl_recsys.agents.base import Agent
from rl_recsys.config import ExperimentConfig
from rl_recsys.environments.base import RecObs
from rl_recsys.environments.synthetic import SyntheticEnv
from rl_recsys.training.trainer import train


class RandomAgent(Agent):
    """Uniformly random slate selection (baseline)."""

    def __init__(self, slate_size: int) -> None:
        self._slate_size = slate_size

    def select_slate(self, obs: RecObs) -> np.ndarray:
        n = len(obs.candidate_features)
        return np.random.choice(n, size=self._slate_size, replace=False)

    def update(
        self,
        obs: RecObs,
        slate: np.ndarray,
        reward: float,
        clicks: np.ndarray,
        next_obs: RecObs,
    ) -> dict[str, float]:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="rl-recsys synthetic experiment")
    parser.add_argument("--episodes", type=int, default=100, help="number of episodes")
    parser.add_argument("--slate-size", type=int, default=10, help="slate size")
    parser.add_argument("--num-candidates", type=int, default=50, help="candidates per step")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    cfg = ExperimentConfig()
    cfg.env.slate_size = args.slate_size
    cfg.env.num_candidates = args.num_candidates
    cfg.train.num_episodes = args.episodes
    cfg.train.seed = args.seed

    env = SyntheticEnv(cfg.env)
    agent = RandomAgent(slate_size=cfg.env.slate_size)

    history = train(env, agent, cfg)
    avg_reward = np.mean([h["reward"] for h in history])
    print(f"\nDone. Average reward over {len(history)} episodes: {avg_reward:.3f}")


if __name__ == "__main__":
    main()
