from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rl_recsys.agents import LinUCBAgent, RandomAgent, build_agent
from rl_recsys.config import AgentConfig, EnvConfig
from rl_recsys.environments.base import RecObs


def _obs() -> RecObs:
    return RecObs(
        user_features=np.array([1.0, 0.0], dtype=np.float32),
        candidate_features=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        ),
        candidate_ids=np.array([10, 20, 30]),
    )


def test_random_agent_score_items_is_uniform() -> None:
    agent = RandomAgent(slate_size=2)
    scores = agent.score_items(_obs())

    assert scores.shape == (3,)
    assert np.allclose(scores, scores[0])  # all equal → uniform softmax


def test_random_agent_selects_unique_slate() -> None:
    agent = RandomAgent(slate_size=2)
    slate = agent.select_slate(_obs())

    assert slate.shape == (2,)
    assert len(set(slate.tolist())) == 2
    assert set(slate.tolist()).issubset({0, 1, 2})


def test_linucb_selects_unique_slate() -> None:
    agent = LinUCBAgent(slate_size=2, user_dim=2, item_dim=2, alpha=1.0)
    slate = agent.select_slate(_obs())

    assert slate.shape == (2,)
    assert len(set(slate.tolist())) == 2
    assert set(slate.tolist()).issubset({0, 1, 2})


def test_linucb_update_changes_parameters() -> None:
    agent = LinUCBAgent(slate_size=2, user_dim=2, item_dim=2, alpha=1.0)
    before_a = agent._a_matrix.copy()
    before_b = agent._b_vector.copy()

    metrics = agent.update(
        _obs(),
        slate=np.array([0, 1]),
        reward=1.0,
        clicks=np.array([1.0, 0.0]),
        next_obs=_obs(),
    )

    assert metrics["agent_updates"] == 2.0
    assert not np.array_equal(agent._a_matrix, before_a)
    assert not np.array_equal(agent._b_vector, before_b)


def test_linucb_positive_feedback_improves_clicked_candidate_score() -> None:
    agent = LinUCBAgent(slate_size=1, user_dim=2, item_dim=2, alpha=0.0)
    obs = _obs()

    before_scores = agent.score_items(obs)
    for _ in range(5):
        agent.update(
            obs,
            slate=np.array([0]),
            reward=1.0,
            clicks=np.array([1.0]),
            next_obs=obs,
        )
    after_scores = agent.score_items(obs)

    assert after_scores[0] > before_scores[0]
    assert after_scores[0] > after_scores[1]
    assert agent.select_slate(obs).tolist() == [0]


def test_linucb_works_with_mismatched_user_and_item_dims() -> None:
    # Interaction term uses min(user_dim, item_dim) dims, so unequal dims are fine.
    agent = LinUCBAgent(slate_size=1, user_dim=4, item_dim=3)
    rng = np.random.default_rng(0)
    obs = RecObs(
        user_features=rng.standard_normal(4).astype(np.float32),
        candidate_features=rng.standard_normal((3, 3)).astype(np.float32),
        candidate_ids=np.array([0, 1, 2]),
    )
    slate = agent.select_slate(obs)
    assert slate.shape == (1,)


def test_build_agent_supports_random_and_linucb() -> None:
    env_cfg = EnvConfig(slate_size=2, user_dim=2, item_dim=2)

    random_agent = build_agent(AgentConfig(name="random"), env_cfg)
    linucb_agent = build_agent(AgentConfig(name="linucb", alpha=0.5), env_cfg)

    assert isinstance(random_agent, RandomAgent)
    assert isinstance(linucb_agent, LinUCBAgent)


def test_build_agent_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown agent"):
        build_agent(AgentConfig(name="missing"), EnvConfig())


def test_hydra_can_load_linucb_agent_config() -> None:
    config_dir = str(Path(__file__).resolve().parents[1] / "conf")
    OmegaConf.register_new_resolver("workspace_root", lambda: "/tmp", replace=True)
    OmegaConf.register_new_resolver("workspace_run_id", lambda: "test", replace=True)

    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="train", overrides=["agent=linucb"])

    assert cfg.agent.name == "linucb"
    assert cfg.agent.alpha == 1.0


def test_agent_score_items_default_returns_zeros():
    from rl_recsys.agents.base import Agent
    from rl_recsys.environments.base import RecObs

    class _MinimalAgent(Agent):
        def select_slate(self, obs):
            return np.zeros(3, dtype=np.int64)

        def update(self, obs, slate, reward, clicks, next_obs):
            return {}

    agent = _MinimalAgent()
    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.zeros((10, 3)),
        candidate_ids=np.arange(10, dtype=np.int64),
    )
    scores = agent.score_items(obs)
    assert scores.shape == (10,)
    assert np.allclose(scores, 0.0)


def test_agent_train_offline_default_calls_pretrain_helper():
    # The default delegates to pretrain_agent_on_logged, which calls
    # agent.update for every step. We assert by counting metrics returned
    # against a fake source that yields one trajectory of two steps.
    from rl_recsys.agents.linucb import LinUCBAgent
    from rl_recsys.environments.base import RecObs
    from rl_recsys.evaluation.ope_trajectory import LoggedTrajectoryStep

    class _FakeSource:
        def iter_trajectories(self, *, max_trajectories=None, seed=0):
            obs = RecObs(
                user_features=np.zeros(4),
                candidate_features=np.zeros((10, 3)),
                candidate_ids=np.arange(10, dtype=np.int64),
            )
            step = LoggedTrajectoryStep(
                obs=obs,
                logged_action=np.array([0, 1, 2], dtype=np.int64),
                logged_reward=1.0,
                logged_clicks=np.array([1, 0, 0], dtype=np.int64),
                propensity=0.1,
            )
            yield [step, step]

    agent = LinUCBAgent(slate_size=3, user_dim=4, item_dim=3)
    metrics = agent.train_offline(_FakeSource(), seed=0)
    assert metrics["total_steps"] == 2.0
    assert metrics["trajectories"] == 1.0
