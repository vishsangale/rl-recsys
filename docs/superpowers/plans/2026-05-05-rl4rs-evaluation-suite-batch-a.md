# RL4RS Evaluation Suite — Batch A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SWIS and DR OPE estimators, `discounted_return` and `per_session_reward` metrics, and a multi-seed variance wrapper to the verification suite, as specified in the RL4RS paper (Wang et al. 2021).

**Architecture:** Extend `rl_recsys/training/metrics.py` with two new pure metric functions; extend `rl_recsys/evaluation/bandit.py` with a `discounted_return` field; extend `rl_recsys/evaluation/ope.py` with `swis_value` and `dr_value` estimator functions and two new `OPEEvaluation` fields; create `rl_recsys/evaluation/variance.py` with a fresh-instance multi-seed wrapper; update `rl_recsys/evaluation/__init__.py` exports.

**Tech Stack:** Python 3.10+, NumPy, pytest, existing `LoggedInteractionEnv`, `RandomAgent`, `OpenBanditEventSampler`.

---

## File Map

| File | Action | What changes |
|---|---|---|
| `rl_recsys/training/metrics.py` | Modify | Add `discounted_return`, `per_session_reward` |
| `rl_recsys/evaluation/bandit.py` | Modify | Add `discounted_return` field to `BanditEvaluation`; compute in `evaluate_bandit_agent` |
| `rl_recsys/evaluation/ope.py` | Modify | Add `swis_value`, `dr_value`; add two fields to `OPEEvaluation`; update `evaluate_ope_agent` |
| `rl_recsys/evaluation/variance.py` | Create | `VarianceEvaluation` dataclass + `evaluate_with_variance` |
| `rl_recsys/evaluation/__init__.py` | Modify | Export new symbols |
| `tests/test_metrics.py` | Create | 3 tests for `discounted_return` and `per_session_reward` |
| `tests/test_ope.py` | Modify | 4 new tests for `swis_value`, `dr_value`, end-to-end |
| `tests/test_variance.py` | Create | 3 tests for `evaluate_with_variance` |

---

## Task 1: Metric functions — `discounted_return` and `per_session_reward`

**Files:**
- Modify: `rl_recsys/training/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_metrics.py`:

```python
import numpy as np
import pytest

from rl_recsys.training.metrics import discounted_return, per_session_reward


def test_discounted_return_geometric_decay() -> None:
    rewards = np.array([1.0, 1.0, 1.0])
    # gamma=0.5: 1*1 + 1*0.5 + 1*0.25 = 1.75
    assert discounted_return(rewards, gamma=0.5) == pytest.approx(1.75)


def test_discounted_return_single_step() -> None:
    rewards = np.array([3.7])
    assert discounted_return(rewards, gamma=0.95) == pytest.approx(3.7)
    assert discounted_return(rewards, gamma=0.0) == pytest.approx(3.7)


def test_per_session_reward_averages_sessions() -> None:
    # session 1 total=3, session 2 total=5 → mean=4
    sessions = [np.array([1.0, 2.0]), np.array([2.0, 3.0])]
    assert per_session_reward(sessions) == pytest.approx(4.0)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/bin/pytest tests/test_metrics.py -v
```

Expected: `ImportError` or `AttributeError` — `discounted_return` and `per_session_reward` not yet defined.

- [ ] **Step 3: Add the two functions to `rl_recsys/training/metrics.py`**

Append after line 37 (end of `ctr`):

```python


def discounted_return(rewards: np.ndarray, gamma: float = 0.95) -> float:
    """Geometric sum of rewards: sum(gamma^i * r_i)."""
    rewards = np.asarray(rewards, dtype=np.float64)
    if len(rewards) == 0:
        return 0.0
    powers = np.arange(len(rewards), dtype=np.float64)
    return float(np.sum(rewards * gamma ** powers))


def per_session_reward(session_rewards: list[np.ndarray]) -> float:
    """Mean total reward across sessions (for trajectory-level evaluation)."""
    if not session_rewards:
        return 0.0
    return float(np.mean([np.sum(r) for r in session_rewards]))
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
.venv/bin/pytest tests/test_metrics.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add rl_recsys/training/metrics.py tests/test_metrics.py
git commit -m "feat: add discounted_return and per_session_reward metrics"
```

---

## Task 2: Add `discounted_return` to `BanditEvaluation`

**Files:**
- Modify: `rl_recsys/evaluation/bandit.py` (lines 1–74)

- [ ] **Step 1: Update the `BanditEvaluation` dataclass and `evaluate_bandit_agent`**

Replace the entire content of `rl_recsys/evaluation/bandit.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecEnv
from rl_recsys.training.metrics import ctr, discounted_return, mrr, ndcg_at_k


@dataclass
class BanditEvaluation:
    agent: str
    episodes: int
    avg_reward: float
    hit_rate: float
    ctr: float
    ndcg: float
    mrr: float
    discounted_return: float
    seconds: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "agent": self.agent,
            "episodes": self.episodes,
            "avg_reward": self.avg_reward,
            "hit_rate": self.hit_rate,
            "ctr": self.ctr,
            "ndcg": self.ndcg,
            "mrr": self.mrr,
            "discounted_return": self.discounted_return,
            "seconds": self.seconds,
        }


def evaluate_bandit_agent(
    env: RecEnv,
    agent: Agent,
    *,
    agent_name: str,
    episodes: int,
    seed: int,
    gamma: float = 0.95,
) -> BanditEvaluation:
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    hits: list[float] = []
    ctrs: list[float] = []
    ndcgs: list[float] = []
    mrrs: list[float] = []
    disc_returns: list[float] = []
    started = perf_counter()

    for _ in range(episodes):
        obs = env.reset(seed=int(rng.integers(0, 2**31)))
        slate = agent.select_slate(obs)
        step = env.step(slate)
        agent.update(obs, slate, step.reward, step.clicks, step.obs)
        rewards.append(step.reward)
        hits.append(float(step.reward > 0.0))
        ctrs.append(ctr(step.clicks))
        ndcgs.append(ndcg_at_k(step.clicks))
        mrrs.append(mrr(step.clicks))
        disc_returns.append(discounted_return(np.array([step.reward]), gamma=gamma))

    seconds = perf_counter() - started
    return BanditEvaluation(
        agent=agent_name,
        episodes=episodes,
        avg_reward=float(np.mean(rewards)),
        hit_rate=float(np.mean(hits)),
        ctr=float(np.mean(ctrs)),
        ndcg=float(np.mean(ndcgs)),
        mrr=float(np.mean(mrrs)),
        discounted_return=float(np.mean(disc_returns)),
        seconds=float(seconds),
    )
```

- [ ] **Step 2: Run the full test suite to confirm no regressions**

```bash
.venv/bin/pytest tests/ -v --tb=short -q
```

Expected: all previously passing tests still pass; `discounted_return` field now present in `BanditEvaluation`.

- [ ] **Step 3: Commit**

```bash
git add rl_recsys/evaluation/bandit.py
git commit -m "feat: add discounted_return field to BanditEvaluation"
```

---

## Task 3: OPE estimators — `swis_value` and `dr_value`

**Files:**
- Modify: `rl_recsys/evaluation/ope.py`
- Modify: `tests/test_ope.py`

- [ ] **Step 1: Write the failing tests**

Add to the bottom of `tests/test_ope.py`:

```python
from rl_recsys.evaluation.ope import dr_value, swis_value


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
```

Also add `import pytest` to the imports at the top of `tests/test_ope.py` if not already present.

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/bin/pytest tests/test_ope.py::test_swis_clips_extreme_ratios tests/test_ope.py::test_dr_uses_mean_reward_when_no_model tests/test_ope.py::test_dr_uses_provided_reward_model tests/test_ope.py::test_evaluate_ope_agent_populates_swis_and_dr -v
```

Expected: `ImportError` — `swis_value` and `dr_value` not yet defined.

- [ ] **Step 3: Add `swis_value` and `dr_value` to `rl_recsys/evaluation/ope.py`**

At the top of `ope.py`, update the imports line to add `Callable`:

```python
from typing import Callable, Protocol
```

Then insert after the `snips_value` function (after line 86), before `evaluate_ope_agent`:

```python

def swis_value(
    rewards: np.ndarray,
    target_probabilities: np.ndarray,
    propensities: np.ndarray,
    clip: tuple[float, float] = (0.1, 10.0),
) -> float:
    """Step-Wise Importance Sampling with propensity ratio clipping.

    Identical to IPS but clips each ratio target/propensity to [clip_lo, clip_hi]
    before averaging. Default clip range (0.1, 10.0) matches the RL4RS paper.
    """
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
    """Doubly Robust OPE estimator.

    Computes mean(w * (r - r_hat) + r_hat) where w = clip(target/propensity).
    When reward_model is None, r_hat defaults to mean(rewards) for all episodes.
    When provided, reward_model(i) returns the direct model estimate for episode i.
    """
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
.venv/bin/pytest tests/test_ope.py::test_swis_clips_extreme_ratios tests/test_ope.py::test_dr_uses_mean_reward_when_no_model tests/test_ope.py::test_dr_uses_provided_reward_model -v
```

Expected: 3 PASSED. (The end-to-end test still fails — fixed in Task 4.)

- [ ] **Step 5: Commit**

```bash
git add rl_recsys/evaluation/ope.py tests/test_ope.py
git commit -m "feat: add swis_value and dr_value OPE estimators"
```

---

## Task 4: Wire `swis_value` and `dr_value` into `OPEEvaluation`

**Files:**
- Modify: `rl_recsys/evaluation/ope.py`

- [ ] **Step 1: Update `OPEEvaluation` dataclass**

Replace the `OPEEvaluation` dataclass (lines 27–50 of `ope.py`) with:

```python
@dataclass
class OPEEvaluation:
    agent: str
    episodes: int
    matches: int
    match_rate: float
    replay_value: float
    ips_value: float
    snips_value: float
    swis_value: float
    dr_value: float
    avg_logged_reward: float
    seconds: float

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
```

- [ ] **Step 2: Update `evaluate_ope_agent` to compute and return the new fields**

Replace the `return OPEEvaluation(...)` block at the end of `evaluate_ope_agent` (lines 131–141) with:

```python
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
```

- [ ] **Step 3: Run all OPE tests**

```bash
.venv/bin/pytest tests/test_ope.py -v
```

Expected: all tests pass including `test_evaluate_ope_agent_populates_swis_and_dr`.

- [ ] **Step 4: Commit**

```bash
git add rl_recsys/evaluation/ope.py
git commit -m "feat: add swis_value and dr_value fields to OPEEvaluation"
```

---

## Task 5: Variance wrapper

**Files:**
- Create: `rl_recsys/evaluation/variance.py`
- Create: `tests/test_variance.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_variance.py`:

```python
from __future__ import annotations

import pandas as pd
import numpy as np
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

    evaluate_with_variance(make_env, make_agent, agent_name="random", episodes=5, n_seeds=4, base_seed=0)

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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/bin/pytest tests/test_variance.py -v
```

Expected: `ModuleNotFoundError` — `rl_recsys.evaluation.variance` does not exist yet.

- [ ] **Step 3: Create `rl_recsys/evaluation/variance.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecEnv
from rl_recsys.evaluation.bandit import evaluate_bandit_agent

_SCALAR_KEYS = ("avg_reward", "hit_rate", "ctr", "ndcg", "mrr", "discounted_return")


@dataclass
class VarianceEvaluation:
    mean: dict[str, float]
    std: dict[str, float]
    n_seeds: int


def evaluate_with_variance(
    make_env: Callable[[], RecEnv],
    make_agent: Callable[[], Agent],
    *,
    agent_name: str,
    episodes: int,
    n_seeds: int = 5,
    base_seed: int = 42,
    gamma: float = 0.95,
) -> VarianceEvaluation:
    """Run evaluate_bandit_agent n_seeds times and return mean ± std per metric.

    make_env and make_agent are called fresh each seed to prevent state leakage.
    Default n_seeds=5 matches the RL4RS paper's reporting convention.
    """
    runs: dict[str, list[float]] = {k: [] for k in _SCALAR_KEYS}

    for i in range(n_seeds):
        env = make_env()
        agent = make_agent()
        result = evaluate_bandit_agent(
            env,
            agent,
            agent_name=agent_name,
            episodes=episodes,
            seed=base_seed + i,
            gamma=gamma,
        )
        for k in _SCALAR_KEYS:
            runs[k].append(getattr(result, k))

    return VarianceEvaluation(
        mean={k: float(np.mean(v)) for k, v in runs.items()},
        std={k: float(np.std(v)) for k, v in runs.items()},
        n_seeds=n_seeds,
    )
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
.venv/bin/pytest tests/test_variance.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add rl_recsys/evaluation/variance.py tests/test_variance.py
git commit -m "feat: add evaluate_with_variance multi-seed wrapper"
```

---

## Task 6: Update `__init__.py` exports

**Files:**
- Modify: `rl_recsys/evaluation/__init__.py`

- [ ] **Step 1: Update exports**

Replace the entire content of `rl_recsys/evaluation/__init__.py` with:

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
from rl_recsys.evaluation.variance import VarianceEvaluation, evaluate_with_variance

__all__ = [
    "OPEEvaluation",
    "OPERecord",
    "dr_value",
    "evaluate_ope_agent",
    "ips_value",
    "replay_value",
    "snips_value",
    "swis_value",
    "VarianceEvaluation",
    "evaluate_with_variance",
]
```

- [ ] **Step 2: Run the full test suite**

```bash
.venv/bin/pytest tests/ -v --tb=short -q
```

Expected: all tests pass, no regressions.

- [ ] **Step 3: Commit**

```bash
git add rl_recsys/evaluation/__init__.py
git commit -m "chore: export swis_value, dr_value, VarianceEvaluation from evaluation package"
```
