from __future__ import annotations

from typing import Callable

import numpy as np

from rl_recsys.evaluation.ope import _validate_ope_arrays


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
