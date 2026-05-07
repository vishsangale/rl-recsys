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


def test_evaluate_trajectory_with_variance_returns_finite_mean_and_std(tmp_path) -> None:
    import pandas as pd

    from rl_recsys.agents import RandomAgent
    from rl_recsys.data.loaders.finn_no_slate_trajectory import (
        FinnNoSlateTrajectoryLoader,
    )
    from rl_recsys.evaluation.variance import evaluate_trajectory_with_variance

    df = pd.DataFrame(
        {
            "request_id": [0, 1, 2, 3, 4, 5],
            "user_id": [10, 10, 11, 11, 12, 12],
            "clicks": [0, 1, 2, 0, 1, 3],
            "slate": [
                [100, 101, 102, 103, 104],
                [200, 201, 202, 203, 204],
                [300, 301, 302, 303, 304],
                [400, 401, 402, 403, 404],
                [500, 501, 502, 503, 504],
                [600, 601, 602, 603, 604],
            ],
        }
    )
    path = tmp_path / "slates.parquet"
    df.to_parquet(path, index=False)

    result = evaluate_trajectory_with_variance(
        make_dataset=lambda: FinnNoSlateTrajectoryLoader(
            path, num_candidates=8, feature_dim=4, slate_size=3, seed=0
        ),
        make_agent=lambda: RandomAgent(slate_size=3, seed=0),
        agent_name="random",
        max_sessions=3,
        n_seeds=3,
        base_seed=0,
    )

    assert result.n_seeds == 3
    metric_keys = (
        "avg_session_reward",
        "avg_discounted_return",
        "avg_session_length",
        "avg_session_hit_rate",
        "avg_per_step_ctr",
        "avg_per_step_ndcg",
        "avg_per_step_mrr",
    )
    for key in metric_keys:
        assert key in result.mean
        assert key in result.std
        assert np.isfinite(result.mean[key])
        assert np.isfinite(result.std[key])
    # Run-config and runtime fields are excluded via aggregate-skip metadata
    for skipped in ("sessions", "total_steps", "seconds"):
        assert skipped not in result.mean
        assert skipped not in result.std
