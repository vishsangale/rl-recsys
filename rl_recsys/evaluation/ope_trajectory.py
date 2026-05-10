from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Iterator, Protocol

import numpy as np

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecObs
from rl_recsys.evaluation.ope import _validate_ope_arrays
from rl_recsys.training.metrics import discounted_return


def seq_dr_value(
    rewards: np.ndarray,
    target_probabilities: np.ndarray,
    propensities: np.ndarray,
    *,
    gamma: float = 0.95,
    reward_model: Callable[[int], float] | None = None,
    clip: tuple[float, float] = (0.1, 10.0),
) -> float:
    """Sequential Doubly Robust on a single trajectory.

    V_DR(τ) = Σ_t γ^t · [ W_t · (r_t − b_t) + b_t ]
    where W_t = Π_{u≤t} clip(π/μ) and b_t = reward_model(t) or mean(rewards).
    """
    rewards, target_probabilities, propensities = _validate_ope_arrays(
        rewards, target_probabilities, propensities
    )
    weights = np.clip(target_probabilities / propensities, clip[0], clip[1])
    cumulative_weights = np.cumprod(weights)
    if reward_model is None:
        baseline = np.full(len(rewards), float(np.mean(rewards)))
    else:
        baseline = np.array(
            [reward_model(i) for i in range(len(rewards))], dtype=np.float64
        )
    discounts = gamma ** np.arange(len(rewards), dtype=np.float64)
    per_step = cumulative_weights * (rewards - baseline) + baseline
    return float(np.sum(discounts * per_step))


@dataclass(frozen=True)
class LoggedTrajectoryStep:
    obs: RecObs
    logged_action: np.ndarray  # shape (slate_size,) — candidate indices (positions in obs.candidate_ids)
    logged_reward: float
    logged_clicks: np.ndarray  # shape (slate_size,) — per-position binary feedback
    propensity: float          # μ(slate | obs) = Π_k μ(slate[k] | obs, k)


class LoggedTrajectorySource(Protocol):
    def iter_trajectories(
        self, *, max_trajectories: int | None = None, seed: int | None = None
    ) -> Iterator[list[LoggedTrajectoryStep]]:
        ...


@dataclass
class TrajectoryOPEEvaluation:
    agent: str
    trajectories: int = field(metadata={"aggregate": False})
    total_steps: int = field(metadata={"aggregate": False})
    avg_seq_dr_value: float
    avg_logged_discounted_return: float
    ess: float = field(metadata={"aggregate": False})
    seconds: float = field(metadata={"aggregate": False})

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "agent": self.agent,
            "trajectories": self.trajectories,
            "total_steps": self.total_steps,
            "avg_seq_dr_value": self.avg_seq_dr_value,
            "avg_logged_discounted_return": self.avg_logged_discounted_return,
            "ess": self.ess,
            "seconds": self.seconds,
        }


def _target_probability(
    agent: Agent,
    obs: RecObs,
    logged_slate: np.ndarray,
    *,
    temperature: float,
) -> float:
    """π_target(logged_slate | obs) under a temperature-softmax shim of agent.

    π(item_k | obs) = softmax(agent.score_items(obs) / T)[item_k]
    π(slate | obs) = Π_k π(slate[k] | obs)
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    scores = np.asarray(agent.score_items(obs), dtype=np.float64)
    # Numerically stable softmax: subtract max before exp.
    z = scores / temperature
    z = z - z.max()
    exp_z = np.exp(z)
    probs = exp_z / exp_z.sum()
    logged = np.asarray(logged_slate, dtype=np.int64)
    return float(np.prod(probs[logged]))


def evaluate_trajectory_ope_agent(
    source: LoggedTrajectorySource,
    agent: Agent,
    *,
    agent_name: str,
    max_trajectories: int,
    seed: int,
    gamma: float = 0.95,
    reward_model: Callable[[int], float] | None = None,
    clip: tuple[float, float] = (0.1, 10.0),
    temperature: float = 1.0,
) -> TrajectoryOPEEvaluation:
    """Sequential DR off-policy evaluator with slate-as-action.

    For each trajectory, the agent picks a slate per step. Per-step target
    probability uses a Boltzmann shim over agent.score_items: π(slate | obs) =
    Π_k softmax(scores / T)[slate[k]]. agent.update() is NOT called.

    Empty trajectories yielded by ``source`` are silently skipped — they have
    no steps to score and would make ``seq_dr_value`` raise. If every yielded
    trajectory is empty (or ``source`` yields nothing), the run raises
    ``ValueError`` because the resulting averages would be undefined.
    """
    if max_trajectories <= 0:
        raise ValueError("max_trajectories must be positive")
    if not hasattr(agent, "score_items"):
        raise AttributeError(
            f"agent {agent_name!r} must implement score_items(obs) -> np.ndarray "
            "for slate OPE under the Boltzmann target-probability shim"
        )

    started = perf_counter()
    seq_dr_per_traj: list[float] = []
    logged_returns: list[float] = []
    total_steps = 0
    all_weights: list[np.ndarray] = []

    for traj in source.iter_trajectories(max_trajectories=max_trajectories, seed=seed):
        if not traj:
            continue
        rewards: list[float] = []
        target_probs: list[float] = []
        propensities: list[float] = []
        for step in traj:
            target_probs.append(
                _target_probability(
                    agent, step.obs,
                    logged_slate=np.asarray(step.logged_action, dtype=np.int64),
                    temperature=temperature,
                )
            )
            rewards.append(float(step.logged_reward))
            propensities.append(float(step.propensity))
        rewards_arr = np.asarray(rewards, dtype=np.float64)
        target_arr = np.asarray(target_probs, dtype=np.float64)
        prop_arr = np.asarray(propensities, dtype=np.float64)
        ratios = np.clip(target_arr / prop_arr, clip[0], clip[1])
        cum_weights = np.cumprod(ratios)
        all_weights.append(cum_weights)
        seq_dr_per_traj.append(
            seq_dr_value(
                rewards_arr, target_arr, prop_arr,
                gamma=gamma, reward_model=reward_model, clip=clip,
            )
        )
        logged_returns.append(discounted_return(rewards_arr, gamma=gamma))
        total_steps += len(traj)

    n = len(seq_dr_per_traj)
    if n == 0:
        raise ValueError("source produced zero trajectories")

    weights = np.concatenate(all_weights)
    ess = float((weights.sum() ** 2) / (np.sum(weights ** 2)))
    if ess / total_steps < 0.01:
        import warnings
        warnings.warn(
            f"effective sample size {ess:.2f} is < 1% of total steps "
            f"({total_steps}); estimator may be unreliable",
            stacklevel=2,
        )

    return TrajectoryOPEEvaluation(
        agent=agent_name,
        trajectories=n,
        total_steps=total_steps,
        avg_seq_dr_value=float(np.mean(seq_dr_per_traj)),
        avg_logged_discounted_return=float(np.mean(logged_returns)),
        ess=ess,
        seconds=float(perf_counter() - started),
    )
