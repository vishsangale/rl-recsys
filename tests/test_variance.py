from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rl_recsys.agents import RandomAgent
from rl_recsys.environments.logged import LoggedInteractionEnv
from rl_recsys.evaluation.variance import VarianceEvaluation, evaluate_with_variance


def _interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 2, 2, 3, 3],
            "item_id": [0, 1, 2, 3, 4, 5, 6, 7],
            "rating": [5.0, 1.0, 4.0, 2.0, 5.0, 1.0, 4.0, 2.0],
            "timestamp": list(range(8)),
        }
    )


def _single_interaction() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [0],
            "item_id": [42],
            "rating": [5.0],
            "timestamp": [0],
        }
    )


def test_evaluate_with_variance_returns_finite_mean_and_std() -> None:
    df = _interactions()
    result = evaluate_with_variance(
        make_env=lambda: LoggedInteractionEnv(
            df, slate_size=2, num_candidates=4, feature_dim=8, rating_threshold=4.0
        ),
        make_agent=lambda: RandomAgent(slate_size=2, seed=0),
        agent_name="random",
        episodes=10,
        n_seeds=3,
        base_seed=0,
    )

    assert isinstance(result, VarianceEvaluation)
    assert result.n_seeds == 3
    for key in ("avg_reward", "hit_rate", "ctr", "ndcg", "mrr", "discounted_return"):
        assert key in result.mean
        assert key in result.std
        assert np.isfinite(result.mean[key])
        assert np.isfinite(result.std[key])


def test_evaluate_with_variance_uses_fresh_instances() -> None:
    df = _interactions()
    env_count = [0]
    agent_count = [0]

    def make_env() -> LoggedInteractionEnv:
        env_count[0] += 1
        return LoggedInteractionEnv(
            df, slate_size=2, num_candidates=4, feature_dim=8, rating_threshold=4.0
        )

    def make_agent() -> RandomAgent:
        agent_count[0] += 1
        return RandomAgent(slate_size=2, seed=0)

    evaluate_with_variance(
        make_env, make_agent, agent_name="random", episodes=5, n_seeds=4, base_seed=0
    )

    assert env_count[0] == 4
    assert agent_count[0] == 4


def test_variance_std_is_zero_for_deterministic_env() -> None:
    # 1 item, always positive → reward=1.0 every episode, std across seeds = 0
    df = _single_interaction()
    result = evaluate_with_variance(
        make_env=lambda: LoggedInteractionEnv(
            df, slate_size=1, num_candidates=1, feature_dim=4, rating_threshold=4.0
        ),
        make_agent=lambda: RandomAgent(slate_size=1, seed=0),
        agent_name="random",
        episodes=10,
        n_seeds=3,
        base_seed=0,
    )

    assert result.std["avg_reward"] == pytest.approx(0.0, abs=1e-10)
    assert result.mean["avg_reward"] == pytest.approx(1.0)


from dataclasses import dataclass


@dataclass
class _FakeResult:
    agent: str
    score: float
    count: int
    seconds: float


def test_evaluate_with_variance_introspects_dataclass_fields() -> None:
    from rl_recsys.evaluation.variance import _aggregate_runs

    runs = [
        _FakeResult(agent="x", score=1.0, count=10, seconds=0.5),
        _FakeResult(agent="x", score=3.0, count=12, seconds=0.6),
        _FakeResult(agent="x", score=5.0, count=14, seconds=0.7),
    ]
    mean, std = _aggregate_runs(runs)

    assert "agent" not in mean
    assert "agent" not in std
    assert set(mean.keys()) == {"score", "count", "seconds"}
    assert mean["score"] == pytest.approx(3.0)
    assert mean["count"] == pytest.approx(12.0)
    assert std["score"] == pytest.approx(np.std([1.0, 3.0, 5.0]))
