import numpy as np
import pytest

from rl_recsys.config import EnvConfig
from rl_recsys.environments.synthetic import SyntheticEnv


class TestSyntheticEnv:
    def test_reset_returns_obs(self, env: SyntheticEnv, env_config: EnvConfig) -> None:
        obs = env.reset(seed=0)
        assert obs.user_features.shape == (env_config.user_dim,)
        assert obs.candidate_features.shape == (
            env_config.num_candidates,
            env_config.item_dim,
        )
        assert obs.candidate_ids.shape == (env_config.num_candidates,)

    def test_step_returns_correct_shapes(
        self, env: SyntheticEnv, env_config: EnvConfig
    ) -> None:
        obs = env.reset(seed=1)
        slate = np.arange(env_config.slate_size)
        step = env.step(slate)

        assert step.clicks.shape == (env_config.slate_size,)
        assert step.obs.user_features.shape == (env_config.user_dim,)
        assert isinstance(step.reward, float)
        assert step.done is False

    def test_clicks_are_binary(
        self, env: SyntheticEnv, env_config: EnvConfig
    ) -> None:
        env.reset(seed=2)
        slate = np.arange(env_config.slate_size)
        step = env.step(slate)
        assert set(step.clicks.tolist()).issubset({0.0, 1.0})

    def test_deterministic_with_seed(self, env_config: EnvConfig) -> None:
        env1 = SyntheticEnv(env_config)
        env2 = SyntheticEnv(env_config)

        obs1 = env1.reset(seed=42)
        obs2 = env2.reset(seed=42)

        np.testing.assert_array_equal(obs1.user_features, obs2.user_features)
        np.testing.assert_array_equal(obs1.candidate_ids, obs2.candidate_ids)

    def test_properties(self, env: SyntheticEnv, env_config: EnvConfig) -> None:
        assert env.slate_size == env_config.slate_size
        assert env.num_candidates == env_config.num_candidates
        assert env.user_dim == env_config.user_dim
        assert env.item_dim == env_config.item_dim
