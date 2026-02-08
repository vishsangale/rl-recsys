import pytest

from rl_recsys.config import EnvConfig
from rl_recsys.environments.synthetic import SyntheticEnv


@pytest.fixture
def env_config() -> EnvConfig:
    return EnvConfig(
        num_items=100,
        num_candidates=20,
        slate_size=5,
        user_dim=8,
        item_dim=8,
    )


@pytest.fixture
def env(env_config: EnvConfig) -> SyntheticEnv:
    return SyntheticEnv(env_config)
