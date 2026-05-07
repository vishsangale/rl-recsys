from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Protocol

import numpy as np

from rl_recsys.agents.base import Agent
from rl_recsys.agents.random import RandomAgent
from rl_recsys.environments.open_bandit import LoggedBanditEvent


class LoggedBanditEventSource(Protocol):
    def sample_event(self, seed: int | None = None) -> LoggedBanditEvent:
        ...


@dataclass(frozen=True)
class OPERecord:
    reward: float
    propensity: float
    target_probability: float
    target_match: bool


@dataclass
class OPEEvaluation:
    agent: str
    episodes: int = field(metadata={"aggregate": False})
    matches: int
    match_rate: float
    replay_value: float
    ips_value: float
    snips_value: float
    swis_value: float
    dr_value: float
    avg_logged_reward: float
    seconds: float = field(metadata={"aggregate": False})

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "agent": self.agent,
            "episodes": self.episodes,
            "matches": self.matches,
            "match_rate": self.match_rate,
            "replay_value": self.replay_value,
            "ips_value": self.ips_value,
            "snips_value": self.snips_value,
            "swis_value": self.swis_value,
            "dr_value": self.dr_value,
            "avg_logged_reward": self.avg_logged_reward,
            "seconds": self.seconds,
        }


def replay_value(rewards: np.ndarray, target_matches: np.ndarray) -> float:
    rewards = _validate_rewards(rewards)
    matches = np.asarray(target_matches, dtype=bool)
    _validate_length("target_matches", matches, len(rewards))
    if not np.any(matches):
        return 0.0
    return float(np.mean(rewards[matches]))


def ips_value(
    rewards: np.ndarray,
    target_probabilities: np.ndarray,
    propensities: np.ndarray,
) -> float:
    rewards, target_probabilities, propensities = _validate_ope_arrays(
        rewards, target_probabilities, propensities
    )
    weights = target_probabilities / propensities
    return float(np.mean(weights * rewards))


def snips_value(
    rewards: np.ndarray,
    target_probabilities: np.ndarray,
    propensities: np.ndarray,
) -> float:
    rewards, target_probabilities, propensities = _validate_ope_arrays(
        rewards, target_probabilities, propensities
    )
    weights = target_probabilities / propensities
    denominator = float(np.sum(weights))
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(weights * rewards) / denominator)


def swis_value(
    rewards: np.ndarray,
    target_probabilities: np.ndarray,
    propensities: np.ndarray,
    clip: tuple[float, float] = (0.1, 10.0),
) -> float:
    """Step-Wise Importance Sampling with propensity ratio clipping."""
    rewards, target_probabilities, propensities = _validate_ope_arrays(
        rewards, target_probabilities, propensities
    )
    weights = np.clip(target_probabilities / propensities, clip[0], clip[1])
    return float(np.mean(weights * rewards))


def dr_value(
    rewards: np.ndarray,
    target_probabilities: np.ndarray,
    propensities: np.ndarray,
    reward_model: Callable[[int], float] | None = None,
    clip: tuple[float, float] = (0.1, 10.0),
) -> float:
    """Doubly Robust OPE estimator."""
    rewards, target_probabilities, propensities = _validate_ope_arrays(
        rewards, target_probabilities, propensities
    )
    weights = np.clip(target_probabilities / propensities, clip[0], clip[1])
    if reward_model is None:
        reward_hat = np.full(len(rewards), float(np.mean(rewards)))
    else:
        reward_hat = np.array(
            [reward_model(i) for i in range(len(rewards))], dtype=np.float64
        )
    return float(np.mean(weights * (rewards - reward_hat) + reward_hat))


def evaluate_ope_agent(
    event_source: LoggedBanditEventSource,
    agent: Agent,
    *,
    agent_name: str,
    episodes: int,
    seed: int,
) -> OPEEvaluation:
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    rng = np.random.default_rng(seed)
    records: list[OPERecord] = []
    started = perf_counter()

    for _ in range(episodes):
        event = event_source.sample_event(seed=int(rng.integers(0, 2**31)))
        slate = np.asarray(agent.select_slate(event.obs), dtype=np.int64)
        if len(slate) == 0:
            raise ValueError("agent returned an empty slate")
        top_action = int(slate[0])
        if top_action < 0 or top_action >= len(event.obs.candidate_ids):
            raise ValueError(f"agent selected invalid candidate index {top_action}")
        target_match = top_action == event.logged_action
        records.append(
            OPERecord(
                reward=event.logged_reward,
                propensity=event.propensity,
                target_probability=_target_probability(agent, event, top_action),
                target_match=target_match,
            )
        )

    seconds = perf_counter() - started
    rewards = np.array([record.reward for record in records], dtype=np.float64)
    propensities = np.array([record.propensity for record in records], dtype=np.float64)
    target_probabilities = np.array(
        [record.target_probability for record in records], dtype=np.float64
    )
    target_matches = np.array([record.target_match for record in records], dtype=bool)
    matches = int(target_matches.sum())

    return OPEEvaluation(
        agent=agent_name,
        episodes=episodes,
        matches=matches,
        match_rate=float(matches / episodes),
        replay_value=replay_value(rewards, target_matches),
        ips_value=ips_value(rewards, target_probabilities, propensities),
        snips_value=snips_value(rewards, target_probabilities, propensities),
        swis_value=swis_value(rewards, target_probabilities, propensities),
        dr_value=dr_value(rewards, target_probabilities, propensities),
        avg_logged_reward=float(np.mean(rewards)),
        seconds=float(seconds),
    )


def _target_probability(
    agent: Agent, event: LoggedBanditEvent, top_action: int
) -> float:
    if isinstance(agent, RandomAgent):
        return float(1.0 / len(event.obs.candidate_ids))
    return float(top_action == event.logged_action)


def _validate_ope_arrays(
    rewards: np.ndarray,
    target_probabilities: np.ndarray,
    propensities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rewards = _validate_rewards(rewards)
    target_probabilities = np.asarray(target_probabilities, dtype=np.float64)
    propensities = np.asarray(propensities, dtype=np.float64)
    _validate_length("target_probabilities", target_probabilities, len(rewards))
    _validate_length("propensities", propensities, len(rewards))
    if not np.all(np.isfinite(target_probabilities)):
        raise ValueError("target_probabilities must be finite")
    if np.any(target_probabilities < 0.0) or np.any(target_probabilities > 1.0):
        raise ValueError("target_probabilities must be probabilities in [0, 1]")
    if not np.all(np.isfinite(propensities)):
        raise ValueError("propensities must be finite")
    if np.any(propensities <= 0.0) or np.any(propensities > 1.0):
        raise ValueError("propensities must be probabilities in (0, 1]")
    return rewards, target_probabilities, propensities


def _validate_rewards(rewards: np.ndarray) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 1:
        raise ValueError("rewards must be one-dimensional")
    if len(rewards) == 0:
        raise ValueError("at least one reward is required")
    if not np.all(np.isfinite(rewards)):
        raise ValueError("rewards must be finite")
    return rewards


def _validate_length(name: str, values: np.ndarray, expected: int) -> None:
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(values) != expected:
        raise ValueError(
            f"{name} length {len(values)} does not match rewards length {expected}"
        )
