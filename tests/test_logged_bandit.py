import numpy as np
import pandas as pd

from rl_recsys.agents import LinUCBAgent, RandomAgent
from rl_recsys.environments.logged import LoggedInteractionEnv
from rl_recsys.evaluation.bandit import evaluate_bandit_agent


def _interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 2, 2, 3, 3],
            "item_id": [0, 1, 2, 3, 4, 5, 6, 7],
            "rating": [5.0, 1.0, 4.0, 2.0, 5.0, 1.0, 4.0, 2.0],
            "timestamp": list(range(8)),
        }
    )


def test_logged_interaction_env_returns_sampled_candidate_set() -> None:
    env = LoggedInteractionEnv(
        _interactions(),
        slate_size=2,
        num_candidates=4,
        feature_dim=8,
        rating_threshold=4.0,
    )

    obs = env.reset(seed=0)

    assert obs.user_features.shape == (8,)
    assert obs.candidate_features.shape == (4, 8)
    assert obs.candidate_ids.shape == (4,)


def test_logged_interaction_env_rewards_selected_positive() -> None:
    env = LoggedInteractionEnv(
        _interactions(),
        slate_size=1,
        num_candidates=4,
        feature_dim=8,
        rating_threshold=4.0,
    )
    obs = env.reset(seed=1)
    positive_index = int(np.where(obs.candidate_ids == env._current_positive_item_id)[0][0])

    step = env.step(np.array([positive_index]))

    assert step.reward == 1.0
    assert step.clicks.tolist() == [1.0]
    assert step.done is True


def test_evaluate_bandit_agent_returns_summary() -> None:
    env = LoggedInteractionEnv(
        _interactions(),
        slate_size=2,
        num_candidates=4,
        feature_dim=8,
        rating_threshold=4.0,
    )
    agent = RandomAgent(slate_size=2)

    result = evaluate_bandit_agent(
        env,
        agent,
        agent_name="random",
        episodes=5,
        seed=2,
    )

    assert result.agent == "random"
    assert result.episodes == 5
    assert 0.0 <= result.hit_rate <= 1.0


def test_linucb_runs_on_logged_interaction_env() -> None:
    env = LoggedInteractionEnv(
        _interactions(),
        slate_size=2,
        num_candidates=4,
        feature_dim=8,
        rating_threshold=4.0,
    )
    agent = LinUCBAgent(slate_size=2, user_dim=8, item_dim=8, alpha=1.0)

    result = evaluate_bandit_agent(
        env,
        agent,
        agent_name="linucb",
        episodes=5,
        seed=3,
    )

    assert result.agent == "linucb"
    assert result.episodes == 5
