import numpy as np
import pandas as pd
import pytest

from rl_recsys.agents import LinUCBAgent, RandomAgent
from rl_recsys.environments.open_bandit import OpenBanditEventSampler
from rl_recsys.evaluation.ope import (
    dr_value,
    evaluate_ope_agent,
    ips_value,
    replay_value,
    snips_value,
    swis_value,
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


def _native_open_bandit_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [0, 0, 0, 0],
            "item_id": [0, 1, 0, 1],
            "rating": [1.0, 0.0, 0.0, 1.0],
            "timestamp": [1, 2, 3, 4],
            "propensity_score": [0.5, 0.5, 0.5, 0.5],
            "policy": ["random", "random", "random", "random"],
            "campaign": ["all", "all", "men", "men"],
            "user_feature_0": ["u0", "u0", "u1", "u1"],
            "item_feature_0": [0.1, 0.2, 0.9, 0.8],
            "item_feature_1": ["i0a", "i1a", "i0m", "i1m"],
            "user_item_affinity_0": [0.9, 0.9, 0.2, 0.2],
            "user_item_affinity_1": [0.1, 0.1, 0.8, 0.8],
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


def test_open_bandit_event_sampler_uses_native_context_when_available() -> None:
    sampler = OpenBanditEventSampler(
        _native_open_bandit_rows(),
        num_candidates=2,
        feature_dim=8,
        feature_source="native",
        seed=0,
    )

    event = sampler.sample_event(seed=2)

    assert event.obs.user_features.shape == (8,)
    assert event.obs.candidate_features.shape == (2, 8)
    assert event.campaign in {"all", "men"}
    assert set(event.obs.candidate_ids.tolist()) == {0, 1}
    assert not np.allclose(event.obs.candidate_features[0], event.obs.candidate_features[1])


def test_open_bandit_event_sampler_supports_hashed_feature_source() -> None:
    sampler = OpenBanditEventSampler(
        _native_open_bandit_rows(),
        num_candidates=2,
        feature_dim=8,
        feature_source="hashed",
        seed=0,
    )

    event = sampler.sample_event(seed=2)

    assert event.obs.user_features.shape == (8,)
    assert event.obs.candidate_features.shape == (2, 8)


def test_open_bandit_event_sampler_rejects_unknown_feature_source() -> None:
    try:
        OpenBanditEventSampler(
            _open_bandit_rows(),
            num_candidates=4,
            feature_dim=8,
            feature_source="missing",
        )
    except ValueError as exc:
        assert "feature_source" in str(exc)
    else:
        raise AssertionError("expected invalid feature_source to raise")


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


def test_swis_clips_extreme_ratios() -> None:
    # ratio for episode 0: 0.5/0.01 = 50 → clipped to 10
    rewards = np.array([1.0, 0.0, 1.0])
    target_probabilities = np.array([0.5, 0.5, 0.5])
    propensities = np.array([0.01, 0.5, 0.25])

    result = swis_value(rewards, target_probabilities, propensities)

    # clipped weights: [10.0, 1.0, 2.0]; weighted rewards: [10.0, 0.0, 2.0]; mean=4.0
    assert result == pytest.approx(4.0)
    # unclipped IPS would give mean([50.0, 0.0, 2.0]) ≠ 4.0
    assert result != pytest.approx(ips_value(rewards, target_probabilities, propensities))


def test_dr_uses_mean_reward_when_no_model() -> None:
    # equal weights → DR collapses to mean(rewards)
    rewards = np.array([2.0, 4.0])
    target_probabilities = np.array([0.5, 0.5])
    propensities = np.array([0.5, 0.5])  # ratio=1.0, no clipping

    result = dr_value(rewards, target_probabilities, propensities, reward_model=None)

    assert result == pytest.approx(float(np.mean(rewards)))


def test_dr_uses_provided_reward_model() -> None:
    rewards = np.array([1.0, 0.0])
    target_probabilities = np.array([0.5, 0.5])
    propensities = np.array([1.0, 0.5])  # weights: [0.5, 1.0]
    # r_hat = 0.0 for all; dr = mean([0.5*(1-0)+0, 1.0*(0-0)+0]) = mean([0.5, 0]) = 0.25
    result = dr_value(
        rewards, target_probabilities, propensities, reward_model=lambda i: 0.0
    )
    assert result == pytest.approx(0.25)


def test_evaluate_ope_agent_populates_swis_and_dr() -> None:
    rows = _open_bandit_rows()
    result = evaluate_ope_agent(
        OpenBanditEventSampler(rows, num_candidates=4, feature_dim=8, seed=0),
        RandomAgent(slate_size=1, seed=0),
        agent_name="random",
        episodes=8,
        seed=0,
    )
    assert np.isfinite(result.swis_value)
    assert np.isfinite(result.dr_value)
