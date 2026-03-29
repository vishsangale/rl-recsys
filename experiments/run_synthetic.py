"""Run a random-agent baseline on the synthetic environment."""
from __future__ import annotations

import hydra
import numpy as np
from omegaconf import DictConfig

from rl_recsys.agents.base import Agent
from rl_recsys.config import ExperimentConfig, to_experiment_config
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


@hydra.main(version_base="1.3", config_path="../conf", config_name="train")
def main(cfg: DictConfig) -> None:
    exp_cfg: ExperimentConfig = to_experiment_config(cfg)

    env = SyntheticEnv(exp_cfg.env)
    agent = RandomAgent(slate_size=exp_cfg.env.slate_size)

    history = train(env, agent, exp_cfg)
    avg_reward = np.mean([h["reward"] for h in history])
    print(f"\nDone. Average reward over {len(history)} episodes: {avg_reward:.3f}")


if __name__ == "__main__":
    main()
