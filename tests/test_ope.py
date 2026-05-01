import numpy as np
import pandas as pd

from rl_recsys.agents import LinUCBAgent, RandomAgent
from rl_recsys.environments.open_bandit import OpenBanditEventSampler
from rl_recsys.evaluation.ope import (
    evaluate_ope_agent,
    ips_value,
    replay_value,
    snips_value,
)


def _open_bandit_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 2, 2],
            "item_id": [10, 20, 30, 40, 50, 60],
            "rating": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "timestamp": list(range(6)),
            "propensity_score": [0.5, 0.25, 0.5, 0.25, 0.5, 0.25],
        }
    )


def test_replay_value_averages_exact_matches_only() -> None:
    rewards = np.array([1.0, 0.0, 1.0])
    matches = np.array([True, False, True])

    assert replay_value(rewards, matches) == 1.0


def test_ips_and_snips_use_importance_weights() -> None:
    rewards = np.array([1.0, 0.0, 1.0])
    target_probabilities = np.array([0.5, 0.5, 0.0])
    propensities = np.array([0.25, 0.5, 0.5])

    assert ips_value(rewards, target_probabilities, propensities) == 2.0 / 3.0
    assert snips_value(rewards, target_probabilities, propensities) == 2.0 / 3.0


def test_ips_rejects_invalid_propensities() -> None:
    rewards = np.array([1.0])
    target_probabilities = np.array([1.0])
    propensities = np.array([0.0])

    try:
        ips_value(rewards, target_probabilities, propensities)
    except ValueError as exc:
        assert "propensities" in str(exc)
    else:
        raise AssertionError("expected invalid propensity to raise")


def test_open_bandit_event_sampler_contains_logged_item_once() -> None:
    sampler = OpenBanditEventSampler(
        _open_bandit_rows(),
        num_candidates=4,
        feature_dim=8,
        seed=0,
    )

    event = sampler.sample_event(seed=123)

    assert event.obs.user_features.shape == (8,)
    assert event.obs.candidate_features.shape == (4, 8)
    assert event.obs.candidate_ids.shape == (4,)
    assert int(np.sum(event.obs.candidate_ids == event.logged_item_id)) == 1
    assert event.obs.candidate_ids[event.logged_action] == event.logged_item_id


def test_open_bandit_event_sampler_is_seed_deterministic() -> None:
    first = OpenBanditEventSampler(
        _open_bandit_rows(),
        num_candidates=4,
        feature_dim=8,
        seed=0,
    ).sample_event(seed=123)
    second = OpenBanditEventSampler(
        _open_bandit_rows(),
        num_candidates=4,
        feature_dim=8,
        seed=999,
    ).sample_event(seed=123)

    assert first.logged_action == second.logged_action
    assert first.logged_item_id == second.logged_item_id
    assert first.obs.candidate_ids.tolist() == second.obs.candidate_ids.tolist()


def test_evaluate_ope_agent_returns_finite_metrics_for_random_and_linucb() -> None:
    rows = _open_bandit_rows()
    random_result = evaluate_ope_agent(
        OpenBanditEventSampler(rows, num_candidates=4, feature_dim=8, seed=0),
        RandomAgent(slate_size=1, seed=0),
        agent_name="random",
        episodes=8,
        seed=0,
    )
    linucb_result = evaluate_ope_agent(
        OpenBanditEventSampler(rows, num_candidates=4, feature_dim=8, seed=0),
        LinUCBAgent(slate_size=1, user_dim=8, item_dim=8, alpha=1.0),
        agent_name="linucb",
        episodes=8,
        seed=0,
    )

    for result in (random_result, linucb_result):
        assert result.episodes == 8
        assert 0 <= result.matches <= 8
        assert 0.0 <= result.match_rate <= 1.0
        assert np.isfinite(result.replay_value)
        assert np.isfinite(result.ips_value)
        assert np.isfinite(result.snips_value)


def test_ope_summary_serializes_to_csv(tmp_path) -> None:
    result = evaluate_ope_agent(
        OpenBanditEventSampler(_open_bandit_rows(), num_candidates=4, feature_dim=8),
        RandomAgent(slate_size=1, seed=0),
        agent_name="random",
        episodes=4,
        seed=0,
    )
    path = tmp_path / "summary.csv"

    pd.DataFrame([result.as_dict()]).to_csv(path, index=False)

    assert path.read_text().startswith("agent,episodes,matches")
