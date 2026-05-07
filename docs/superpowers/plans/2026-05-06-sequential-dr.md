# Sequential DR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Sequential Doubly Robust (`seq_dr_value`) and a trajectory-OPE evaluator (`evaluate_trajectory_ope_agent`) over a `LoggedTrajectorySource` Protocol. Verified end-to-end on a synthetic in-test trajectory source.

**Architecture:** New module `rl_recsys/evaluation/ope_trajectory.py` holds the pure formula, the source Protocol, the `LoggedTrajectoryStep` dataclass, the `TrajectoryOPEEvaluation` dataclass, and the orchestrator. `rl_recsys/evaluation/ope.py` is untouched — sequential pieces are isolated, mirroring the bandit (`bandit.py`) → trajectory (`trajectory.py`) split. Reuses `_validate_ope_arrays` from `ope.py`.

**Tech Stack:** Python 3.10+, NumPy, pytest. Reuses `RecObs`, `Agent`, and `discounted_return`.

---

## File Map

| File | Action | What changes |
|---|---|---|
| `rl_recsys/evaluation/ope_trajectory.py` | Create | `seq_dr_value`, `LoggedTrajectoryStep`, `LoggedTrajectorySource`, `TrajectoryOPEEvaluation`, `evaluate_trajectory_ope_agent` |
| `rl_recsys/evaluation/__init__.py` | Modify | Export new symbols |
| `tests/test_ope_trajectory.py` | Create | 6 tests + `_SyntheticTrajectorySource` fixture |

---

## Task 1: `seq_dr_value` pure function

**Files:**
- Create: `rl_recsys/evaluation/ope_trajectory.py`
- Create: `tests/test_ope_trajectory.py`

- [ ] **Step 1: Create `tests/test_ope_trajectory.py` with the 4 pure-function tests**

```python
from __future__ import annotations

import numpy as np
import pytest

from rl_recsys.evaluation.ope_trajectory import seq_dr_value


def test_seq_dr_value_collapses_to_logged_return_when_target_equals_behavior() -> None:
    # target == behavior → w=1, W_t=1 → result = sum(γ^t r_t)
    rewards = np.array([1.0, 0.5, 2.0])
    probs = np.array([0.3, 0.4, 0.5])
    gamma = 0.9
    expected = sum(gamma ** t * r for t, r in enumerate(rewards))

    result = seq_dr_value(rewards, probs, probs, gamma=gamma)

    assert result == pytest.approx(expected)


def test_seq_dr_value_collapses_to_per_decision_is_when_baseline_zero() -> None:
    # reward_model = 0 → baseline cancels → result = sum(γ^t W_t r_t)
    rewards = np.array([1.0, 1.0, 1.0])
    target_probs = np.array([0.4, 0.5, 0.6])
    propensities = np.array([0.2, 0.5, 0.3])
    # ratios: [2.0, 1.0, 2.0]; W = [2.0, 2.0, 4.0]
    gamma = 0.9
    expected = (1.0 * 2.0 * 1.0) + (0.9 * 2.0 * 1.0) + (0.81 * 4.0 * 1.0)  # = 7.04

    result = seq_dr_value(
        rewards, target_probs, propensities,
        gamma=gamma, reward_model=lambda i: 0.0,
    )

    assert result == pytest.approx(expected)


def test_seq_dr_value_clips_extreme_ratios() -> None:
    # ratio[0] = 0.99/0.01 = 99 → clipped to 10
    # ratio[1] = 0.5/0.5 = 1
    # Use reward_model=lambda i: 0.0 so clipping is detectable (mean(rewards)
    # baseline would cancel the W_t difference at γ=1.0).
    rewards = np.array([2.0, 0.0])
    target_probs = np.array([0.99, 0.5])
    propensities = np.array([0.01, 0.5])
    gamma = 1.0
    # Clipped: W = [10, 10]; V = 1*(10*2 + 0) + 1*(10*0 + 0) = 20
    clipped_expected = 20.0
    # Unclipped manually: W = [99, 99]; V_unclipped = 1*(99*2) + 1*(99*0) = 198

    result = seq_dr_value(
        rewards, target_probs, propensities,
        gamma=gamma, reward_model=lambda i: 0.0,
    )

    assert result == pytest.approx(clipped_expected)
    assert result != pytest.approx(198.0)


def test_seq_dr_value_uses_provided_reward_model() -> None:
    # reward_model returns i+1 → b = [1.0, 2.0]
    # target == behavior → W = [1, 1]
    # γ=0.5: V = 1*(1*(3-1) + 1) + 0.5*(1*(1-2) + 2) = 3 + 0.5 = 3.5
    rewards = np.array([3.0, 1.0])
    probs = np.array([0.5, 0.5])
    expected = 1.0 * (1 * (3.0 - 1.0) + 1.0) + 0.5 * (1 * (1.0 - 2.0) + 2.0)

    result = seq_dr_value(
        rewards, probs, probs,
        gamma=0.5, reward_model=lambda i: float(i + 1),
    )

    assert result == pytest.approx(expected)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/vishsangale/workspace/rl-recsys && .venv/bin/pytest tests/test_ope_trajectory.py -v
```

Expected: ImportError — `rl_recsys.evaluation.ope_trajectory` does not exist.

- [ ] **Step 3: Create `rl_recsys/evaluation/ope_trajectory.py` with `seq_dr_value`**

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /home/vishsangale/workspace/rl-recsys && .venv/bin/pytest tests/test_ope_trajectory.py -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
cd /home/vishsangale/workspace/rl-recsys && \
  git add rl_recsys/evaluation/ope_trajectory.py tests/test_ope_trajectory.py && \
  git commit -m "feat: add seq_dr_value sequential doubly robust estimator"
```

---

## Task 2: `LoggedTrajectoryStep`, Protocol, evaluator, and `TrajectoryOPEEvaluation`

**Files:**
- Modify: `rl_recsys/evaluation/ope_trajectory.py`
- Modify: `tests/test_ope_trajectory.py`

- [ ] **Step 1a: Update the top-of-file imports in `tests/test_ope_trajectory.py`**

Replace the existing import block at the top of the file with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pytest

from rl_recsys.agents import LinUCBAgent
from rl_recsys.environments.base import RecObs
from rl_recsys.evaluation.ope_trajectory import (
    LoggedTrajectoryStep,
    TrajectoryOPEEvaluation,
    evaluate_trajectory_ope_agent,
    seq_dr_value,
)
```

- [ ] **Step 1b: Append the orchestrator tests + synthetic fixture at the bottom of `tests/test_ope_trajectory.py`**

```python


def _make_obs(num_candidates: int = 4, feature_dim: int = 4) -> RecObs:
    return RecObs(
        user_features=np.zeros(feature_dim, dtype=np.float64),
        candidate_features=np.zeros((num_candidates, feature_dim), dtype=np.float64),
        candidate_ids=np.arange(num_candidates, dtype=np.int64),
    )


@dataclass
class _SyntheticTrajectorySource:
    trajectories: list[list[LoggedTrajectoryStep]]

    def iter_trajectories(
        self, *, max_trajectories: int | None = None, seed: int | None = None
    ) -> Iterator[list[LoggedTrajectoryStep]]:
        out = (
            self.trajectories
            if max_trajectories is None
            else self.trajectories[:max_trajectories]
        )
        for traj in out:
            yield traj


class _DetAgent:
    """Always picks slate=[0] — top action is candidate index 0."""

    def __init__(self, slate_size: int = 1) -> None:
        self._slate_size = slate_size

    def select_slate(self, obs: RecObs) -> np.ndarray:
        return np.arange(self._slate_size, dtype=np.int64)

    def update(
        self,
        obs: RecObs,
        slate: np.ndarray,
        reward: float,
        clicks: np.ndarray,
        next_obs: RecObs,
    ) -> dict[str, float]:
        return {}


def test_evaluate_trajectory_ope_agent_aggregates_per_trajectory() -> None:
    # _DetAgent picks index 0; we set logged_action=0 so target_prob=1.0 for non-Random agents.
    # Trajectory A (2 steps): rewards=[1.0, 0.0], propensity=[0.5, 0.5]
    #   target_prob = 1.0; w = [2.0, 2.0]; W = [2.0, 4.0]
    #   b = mean([1, 0]) = 0.5
    #   V_A(γ=0.9) = 1*(2*(1-0.5)+0.5) + 0.9*(4*(0-0.5)+0.5) = 1.5 + 0.9*(-1.5) = 0.15
    # Trajectory B (1 step): rewards=[2.0], propensity=[1.0]
    #   w = [1.0]; W = [1.0]; b = 2.0
    #   V_B = 1*(1*(2-2)+2) = 2.0
    # avg_seq_dr = (0.15 + 2.0) / 2 = 1.075
    # avg_logged_discounted_return:
    #   Logged_A(γ=0.9) = 1.0 + 0.9*0.0 = 1.0
    #   Logged_B = 2.0
    #   avg = (1.0 + 2.0) / 2 = 1.5
    obs = _make_obs(num_candidates=4)
    traj_a = [
        LoggedTrajectoryStep(obs=obs, logged_action=0, logged_reward=1.0, propensity=0.5),
        LoggedTrajectoryStep(obs=obs, logged_action=0, logged_reward=0.0, propensity=0.5),
    ]
    traj_b = [
        LoggedTrajectoryStep(obs=obs, logged_action=0, logged_reward=2.0, propensity=1.0),
    ]
    source = _SyntheticTrajectorySource(trajectories=[traj_a, traj_b])
    agent = _DetAgent(slate_size=1)

    result = evaluate_trajectory_ope_agent(
        source, agent, agent_name="det", max_trajectories=2, seed=0, gamma=0.9
    )

    assert isinstance(result, TrajectoryOPEEvaluation)
    assert result.trajectories == 2
    assert result.total_steps == 3
    assert result.avg_seq_dr_value == pytest.approx((0.15 + 2.0) / 2)
    assert result.avg_logged_discounted_return == pytest.approx((1.0 + 2.0) / 2)


def test_evaluate_trajectory_ope_agent_does_not_mutate_agent_state() -> None:
    # Confirm LinUCB internal matrices are unchanged after eval — agent.update must not run.
    obs = RecObs(
        user_features=np.ones(4, dtype=np.float64),
        candidate_features=np.eye(4, 4, dtype=np.float64),
        candidate_ids=np.arange(4, dtype=np.int64),
    )
    step = LoggedTrajectoryStep(
        obs=obs, logged_action=0, logged_reward=1.0, propensity=0.5
    )
    source = _SyntheticTrajectorySource(trajectories=[[step] * 5])
    agent = LinUCBAgent(slate_size=1, user_dim=4, item_dim=4, alpha=1.0)
    a_before = agent._a_matrix.copy()
    b_before = agent._b_vector.copy()

    evaluate_trajectory_ope_agent(
        source, agent, agent_name="linucb", max_trajectories=1, seed=0
    )

    assert np.array_equal(agent._a_matrix, a_before)
    assert np.array_equal(agent._b_vector, b_before)
```

- [ ] **Step 2: Run the new tests to confirm they fail**

```bash
cd /home/vishsangale/workspace/rl-recsys && .venv/bin/pytest tests/test_ope_trajectory.py::test_evaluate_trajectory_ope_agent_aggregates_per_trajectory tests/test_ope_trajectory.py::test_evaluate_trajectory_ope_agent_does_not_mutate_agent_state -v
```

Expected: ImportError — `LoggedTrajectoryStep`, `TrajectoryOPEEvaluation`, `evaluate_trajectory_ope_agent` not yet defined.

- [ ] **Step 3: Append types, helper, and orchestrator to `rl_recsys/evaluation/ope_trajectory.py`**

Update the imports section at the top of the file to:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Iterator, Protocol

import numpy as np

from rl_recsys.agents.base import Agent
from rl_recsys.agents.random import RandomAgent
from rl_recsys.environments.base import RecObs
from rl_recsys.evaluation.ope import _validate_ope_arrays
from rl_recsys.training.metrics import discounted_return
```

Append at the bottom of the file (after `seq_dr_value`):

```python


@dataclass(frozen=True)
class LoggedTrajectoryStep:
    obs: RecObs
    logged_action: int
    logged_reward: float
    propensity: float


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
    seconds: float = field(metadata={"aggregate": False})

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "agent": self.agent,
            "trajectories": self.trajectories,
            "total_steps": self.total_steps,
            "avg_seq_dr_value": self.avg_seq_dr_value,
            "avg_logged_discounted_return": self.avg_logged_discounted_return,
            "seconds": self.seconds,
        }


def _target_probability(
    agent: Agent, obs: RecObs, top_action: int, logged_action: int
) -> float:
    if isinstance(agent, RandomAgent):
        return float(1.0 / len(obs.candidate_ids))
    return float(top_action == logged_action)


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
) -> TrajectoryOPEEvaluation:
    """Sequential DR off-policy evaluator.

    For each trajectory, the agent picks a slate per step. Per-step target
    probability is 1/num_candidates for RandomAgent or 1.0/0.0 indicator for
    deterministic agents. agent.update() is NOT called.
    """
    if max_trajectories <= 0:
        raise ValueError("max_trajectories must be positive")

    started = perf_counter()
    seq_dr_per_traj: list[float] = []
    logged_returns: list[float] = []
    total_steps = 0

    for traj in source.iter_trajectories(max_trajectories=max_trajectories, seed=seed):
        if not traj:
            continue
        rewards: list[float] = []
        target_probs: list[float] = []
        propensities: list[float] = []
        for step in traj:
            slate = np.asarray(agent.select_slate(step.obs), dtype=np.int64)
            if len(slate) == 0:
                raise ValueError("agent returned an empty slate")
            top_action = int(slate[0])
            target_probs.append(
                _target_probability(agent, step.obs, top_action, step.logged_action)
            )
            rewards.append(float(step.logged_reward))
            propensities.append(float(step.propensity))
        rewards_arr = np.asarray(rewards, dtype=np.float64)
        target_arr = np.asarray(target_probs, dtype=np.float64)
        prop_arr = np.asarray(propensities, dtype=np.float64)
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

    return TrajectoryOPEEvaluation(
        agent=agent_name,
        trajectories=n,
        total_steps=total_steps,
        avg_seq_dr_value=float(np.mean(seq_dr_per_traj)),
        avg_logged_discounted_return=float(np.mean(logged_returns)),
        seconds=float(perf_counter() - started),
    )
```

- [ ] **Step 4: Run all tests in the file**

```bash
cd /home/vishsangale/workspace/rl-recsys && .venv/bin/pytest tests/test_ope_trajectory.py -v
```

Expected: 6 PASSED.

- [ ] **Step 5: Run the full suite to confirm no regressions**

```bash
cd /home/vishsangale/workspace/rl-recsys && .venv/bin/pytest tests/ -q --tb=short
```

Expected: 155 + 6 = 161 PASSED.

- [ ] **Step 6: Commit**

```bash
cd /home/vishsangale/workspace/rl-recsys && \
  git add rl_recsys/evaluation/ope_trajectory.py tests/test_ope_trajectory.py && \
  git commit -m "feat: add evaluate_trajectory_ope_agent for sequential DR"
```

---

## Task 3: Update `__init__.py` exports

**Files:**
- Modify: `rl_recsys/evaluation/__init__.py`

- [ ] **Step 1: Replace the entire content of `rl_recsys/evaluation/__init__.py` with**

```python
"""Evaluation helpers for offline and sampled recommender benchmarks."""

from rl_recsys.evaluation.ope import (
    OPEEvaluation,
    OPERecord,
    dr_value,
    evaluate_ope_agent,
    ips_value,
    replay_value,
    snips_value,
    swis_value,
)
from rl_recsys.evaluation.ope_trajectory import (
    LoggedTrajectoryStep,
    LoggedTrajectorySource,
    TrajectoryOPEEvaluation,
    evaluate_trajectory_ope_agent,
    seq_dr_value,
)
from rl_recsys.evaluation.trajectory import (
    Session,
    TrajectoryDataset,
    TrajectoryEvaluation,
    TrajectoryStep,
    evaluate_trajectory_agent,
)
from rl_recsys.evaluation.variance import (
    VarianceEvaluation,
    evaluate_trajectory_with_variance,
    evaluate_with_variance,
)

__all__ = [
    "LoggedTrajectorySource",
    "LoggedTrajectoryStep",
    "OPEEvaluation",
    "OPERecord",
    "Session",
    "TrajectoryDataset",
    "TrajectoryEvaluation",
    "TrajectoryOPEEvaluation",
    "TrajectoryStep",
    "VarianceEvaluation",
    "dr_value",
    "evaluate_ope_agent",
    "evaluate_trajectory_agent",
    "evaluate_trajectory_ope_agent",
    "evaluate_trajectory_with_variance",
    "evaluate_with_variance",
    "ips_value",
    "replay_value",
    "seq_dr_value",
    "snips_value",
    "swis_value",
]
```

- [ ] **Step 2: Verify the new exports**

```bash
cd /home/vishsangale/workspace/rl-recsys && .venv/bin/python -c "
from rl_recsys.evaluation import (
    seq_dr_value, LoggedTrajectoryStep, LoggedTrajectorySource,
    TrajectoryOPEEvaluation, evaluate_trajectory_ope_agent,
)
print('All sequential DR exports importable from rl_recsys.evaluation')
"
```

Expected: prints "All sequential DR exports importable from rl_recsys.evaluation".

- [ ] **Step 3: Run the full suite**

```bash
cd /home/vishsangale/workspace/rl-recsys && .venv/bin/pytest tests/ -q --tb=short
```

Expected: 161 PASSED.

- [ ] **Step 4: Commit**

```bash
cd /home/vishsangale/workspace/rl-recsys && \
  git add rl_recsys/evaluation/__init__.py && \
  git commit -m "chore: export sequential DR symbols from evaluation package"
```
