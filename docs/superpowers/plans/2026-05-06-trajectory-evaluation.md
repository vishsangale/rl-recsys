# Trajectory Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add multi-step trajectory evaluation to the verification suite via a `TrajectoryDataset` Protocol, an `evaluate_trajectory_agent` function over `RecObs`, and a `FinnNoSlateTrajectoryLoader` that consumes `data/processed/finn-no-slate/slates.parquet`. Generalize the variance wrapper to introspect any dataclass's numeric fields and add a parallel `evaluate_trajectory_with_variance`.

**Architecture:** New module `rl_recsys/evaluation/trajectory.py` holds the types, the replay reward rule, and the evaluator. New module `rl_recsys/data/loaders/finn_no_slate_trajectory.py` is the first concrete `TrajectoryDataset`. The variance wrapper at `rl_recsys/evaluation/variance.py` is refactored from a hardcoded `_SCALAR_KEYS` constant to a `dataclasses.fields()`-based helper that works for any numeric-field dataclass. Existing bandit/OPE code is untouched.

**Tech Stack:** Python 3.10+, NumPy, pandas, pytest. Reuses `RecObs` from `rl_recsys.environments.base`, the per-step metrics (`ctr`, `ndcg_at_k`, `mrr`, `discounted_return`) from `rl_recsys.training.metrics`, and `hashed_vector` from `rl_recsys.environments.features`.

---

## File Map

| File | Action | What changes |
|---|---|---|
| `rl_recsys/evaluation/trajectory.py` | Create | Types (`TrajectoryStep`, `Session`, `TrajectoryDataset`, `TrajectoryEvaluation`) + `evaluate_trajectory_agent` |
| `rl_recsys/data/loaders/__init__.py` | Create | Empty package marker |
| `rl_recsys/data/loaders/finn_no_slate_trajectory.py` | Create | `FinnNoSlateTrajectoryLoader` |
| `rl_recsys/evaluation/variance.py` | Modify | Drop `_SCALAR_KEYS`; add `_aggregate_runs` helper; add `evaluate_trajectory_with_variance` |
| `rl_recsys/evaluation/__init__.py` | Modify | Export new symbols |
| `tests/test_trajectory.py` | Create | 5 tests for `evaluate_trajectory_agent` + 1 test for the loader |
| `tests/test_variance.py` | Modify | 2 new tests (introspection + trajectory variance smoke) |

---

## Task 1: Trajectory types, replay rule, and `evaluate_trajectory_agent`

**Files:**
- Create: `rl_recsys/evaluation/trajectory.py`
- Create: `tests/test_trajectory.py`

- [ ] **Step 1: Create `tests/test_trajectory.py` with the 5 evaluator tests**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pytest

from rl_recsys.agents import LinUCBAgent, RandomAgent
from rl_recsys.environments.base import RecObs
from rl_recsys.evaluation.trajectory import (
    Session,
    TrajectoryEvaluation,
    TrajectoryStep,
    evaluate_trajectory_agent,
)


def _make_step(*, clicked_id: int, candidate_ids: np.ndarray) -> TrajectoryStep:
    cand = np.asarray(candidate_ids, dtype=np.int64)
    feature_dim = 4
    obs = RecObs(
        user_features=np.zeros(feature_dim, dtype=np.float64),
        candidate_features=np.zeros((len(cand), feature_dim), dtype=np.float64),
        candidate_ids=cand,
    )
    return TrajectoryStep(
        obs=obs,
        logged_slate=cand[:3].copy(),
        logged_clicked_id=clicked_id,
        logged_reward=1.0 if clicked_id != -1 else 0.0,
    )


@dataclass
class _StaticDataset:
    sessions: list[Session]

    def iter_sessions(
        self, *, max_sessions: int | None = None, seed: int | None = None
    ) -> Iterator[Session]:
        out = self.sessions if max_sessions is None else self.sessions[:max_sessions]
        for s in out:
            yield s


class _DeterministicSlateAgent:
    """Always returns slate = [0, 1, 2] — first 3 candidates."""

    def __init__(self, slate_size: int = 3) -> None:
        self._slate_size = slate_size

    def select_slate(self, obs: RecObs) -> np.ndarray:
        return np.arange(self._slate_size, dtype=np.int64)

    def update(self, obs, slate, reward, clicks, next_obs):
        return {}


def test_replay_reward_when_slate_covers_logged_click() -> None:
    # candidate_ids = [10, 11, 12, 13]; agent picks indices [0, 1, 2] = [10, 11, 12].
    # logged_clicked_id = 11 → covered → reward = 1.0
    step = _make_step(clicked_id=11, candidate_ids=np.array([10, 11, 12, 13]))
    session = Session(session_id=1, steps=[step])
    dataset = _StaticDataset(sessions=[session])
    agent = _DeterministicSlateAgent(slate_size=3)

    result = evaluate_trajectory_agent(
        dataset, agent, agent_name="det", max_sessions=1, seed=0
    )

    assert result.avg_session_reward == pytest.approx(1.0)
    assert result.avg_session_hit_rate == pytest.approx(1.0)


def test_replay_reward_zero_when_slate_misses_logged_click() -> None:
    # candidate_ids = [10, 11, 12, 13]; agent picks [10, 11, 12].
    # logged_clicked_id = 13 → not in slate → reward = 0
    step = _make_step(clicked_id=13, candidate_ids=np.array([10, 11, 12, 13]))
    session = Session(session_id=1, steps=[step])
    dataset = _StaticDataset(sessions=[session])
    agent = _DeterministicSlateAgent(slate_size=3)

    result = evaluate_trajectory_agent(
        dataset, agent, agent_name="det", max_sessions=1, seed=0
    )

    assert result.avg_session_reward == pytest.approx(0.0)
    assert result.avg_session_hit_rate == pytest.approx(0.0)


def test_evaluate_trajectory_agent_aggregates_per_session() -> None:
    # Two sessions, 3 steps each, all covered. Each step reward = 1.0.
    # Per-session sum = 3.0; per-session discounted = 1 + 0.95 + 0.95^2 = 2.8525.
    sessions = []
    for sid in (1, 2):
        steps = [
            _make_step(clicked_id=11, candidate_ids=np.array([10, 11, 12, 13]))
            for _ in range(3)
        ]
        sessions.append(Session(session_id=sid, steps=steps))
    dataset = _StaticDataset(sessions=sessions)
    agent = _DeterministicSlateAgent(slate_size=3)

    result = evaluate_trajectory_agent(
        dataset, agent, agent_name="det", max_sessions=2, seed=0, gamma=0.95
    )

    assert isinstance(result, TrajectoryEvaluation)
    assert result.sessions == 2
    assert result.total_steps == 6
    assert result.avg_session_reward == pytest.approx(3.0)
    assert result.avg_session_length == pytest.approx(3.0)
    assert result.avg_discounted_return == pytest.approx(1 + 0.95 + 0.95 ** 2)
    assert result.avg_session_hit_rate == pytest.approx(1.0)


def test_evaluate_trajectory_agent_handles_uncovered_steps() -> None:
    # 1 session, 4 steps: covered, uncovered, covered, uncovered.
    # Reward sequence = [1, 0, 1, 0] → session sum = 2.0; hit_rate = 1.0.
    cands = np.array([10, 11, 12, 13])
    session = Session(
        session_id=1,
        steps=[
            _make_step(clicked_id=11, candidate_ids=cands),  # covered
            _make_step(clicked_id=13, candidate_ids=cands),  # missed
            _make_step(clicked_id=10, candidate_ids=cands),  # covered
            _make_step(clicked_id=13, candidate_ids=cands),  # missed
        ],
    )
    dataset = _StaticDataset(sessions=[session])
    agent = _DeterministicSlateAgent(slate_size=3)

    result = evaluate_trajectory_agent(
        dataset, agent, agent_name="det", max_sessions=1, seed=0
    )

    assert result.avg_session_reward == pytest.approx(2.0)
    assert result.avg_session_length == pytest.approx(4.0)
    assert result.total_steps == 4


def test_evaluate_trajectory_agent_does_not_mutate_agent_state() -> None:
    # LinUCB has internal state (_a_matrix, _b_vector). Eval must not touch them.
    cands = np.array([10, 11, 12, 13])
    feature_dim = 4
    obs = RecObs(
        user_features=np.ones(feature_dim, dtype=np.float64),
        candidate_features=np.eye(4, feature_dim, dtype=np.float64),
        candidate_ids=cands,
    )
    step = TrajectoryStep(
        obs=obs,
        logged_slate=cands[:3].copy(),
        logged_clicked_id=11,
        logged_reward=1.0,
    )
    session = Session(session_id=1, steps=[step] * 5)
    dataset = _StaticDataset(sessions=[session])
    agent = LinUCBAgent(slate_size=3, user_dim=feature_dim, item_dim=feature_dim, alpha=1.0)
    a_before = agent._a_matrix.copy()
    b_before = agent._b_vector.copy()

    evaluate_trajectory_agent(dataset, agent, agent_name="linucb", max_sessions=1, seed=0)

    assert np.array_equal(agent._a_matrix, a_before)
    assert np.array_equal(agent._b_vector, b_before)
```

- [ ] **Step 2: Run the 5 tests to confirm they fail**

```bash
cd /home/vishsangale/workspace/rl-recsys && .venv/bin/pytest tests/test_trajectory.py -v
```

Expected: `ImportError` — `rl_recsys.evaluation.trajectory` does not exist.

- [ ] **Step 3: Create `rl_recsys/evaluation/trajectory.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterator, Protocol

import numpy as np

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecObs
from rl_recsys.training.metrics import ctr, discounted_return, mrr, ndcg_at_k


@dataclass(frozen=True)
class TrajectoryStep:
    obs: RecObs
    logged_slate: np.ndarray
    logged_clicked_id: int
    logged_reward: float


@dataclass(frozen=True)
class Session:
    session_id: int
    steps: list[TrajectoryStep]


class TrajectoryDataset(Protocol):
    def iter_sessions(
        self, *, max_sessions: int | None = None, seed: int | None = None
    ) -> Iterator[Session]:
        ...


@dataclass
class TrajectoryEvaluation:
    agent: str
    sessions: int
    total_steps: int
    avg_session_reward: float
    avg_discounted_return: float
    avg_session_length: float
    avg_session_hit_rate: float
    avg_per_step_ctr: float
    avg_per_step_ndcg: float
    avg_per_step_mrr: float
    seconds: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "agent": self.agent,
            "sessions": self.sessions,
            "total_steps": self.total_steps,
            "avg_session_reward": self.avg_session_reward,
            "avg_discounted_return": self.avg_discounted_return,
            "avg_session_length": self.avg_session_length,
            "avg_session_hit_rate": self.avg_session_hit_rate,
            "avg_per_step_ctr": self.avg_per_step_ctr,
            "avg_per_step_ndcg": self.avg_per_step_ndcg,
            "avg_per_step_mrr": self.avg_per_step_mrr,
            "seconds": self.seconds,
        }


def evaluate_trajectory_agent(
    dataset: TrajectoryDataset,
    agent: Agent,
    *,
    agent_name: str,
    max_sessions: int,
    seed: int,
    gamma: float = 0.95,
) -> TrajectoryEvaluation:
    """Replay-mode trajectory evaluator.

    For each step, the agent picks a slate from the candidate pool. Reward
    equals the logged_reward if the agent's slate covers logged_clicked_id;
    otherwise zero. agent.update() is NOT called — the agent's state is frozen
    for the duration of evaluation.
    """
    if max_sessions <= 0:
        raise ValueError("max_sessions must be positive")

    started = perf_counter()
    session_rewards: list[float] = []
    session_disc_returns: list[float] = []
    session_lengths: list[int] = []
    session_hits: list[float] = []
    per_step_ctrs: list[float] = []
    per_step_ndcgs: list[float] = []
    per_step_mrrs: list[float] = []
    total_steps = 0

    for session in dataset.iter_sessions(max_sessions=max_sessions, seed=seed):
        rewards_per_step: list[float] = []
        for step in session.steps:
            slate_indices = np.asarray(agent.select_slate(step.obs), dtype=np.int64)
            slate_ids = step.obs.candidate_ids[slate_indices]
            covered = (
                step.logged_clicked_id != -1
                and bool(np.any(slate_ids == step.logged_clicked_id))
            )
            if covered:
                clicks = (slate_ids == step.logged_clicked_id).astype(np.float64)
                r = float(step.logged_reward)
            else:
                clicks = np.zeros(len(slate_indices), dtype=np.float64)
                r = 0.0
            rewards_per_step.append(r)
            per_step_ctrs.append(ctr(clicks))
            per_step_ndcgs.append(ndcg_at_k(clicks))
            per_step_mrrs.append(mrr(clicks))
            total_steps += 1
        rewards_arr = np.asarray(rewards_per_step, dtype=np.float64)
        session_rewards.append(float(rewards_arr.sum()))
        session_disc_returns.append(discounted_return(rewards_arr, gamma=gamma))
        session_lengths.append(len(session.steps))
        session_hits.append(float(rewards_arr.sum() > 0.0))

    sessions_count = len(session_rewards)
    if sessions_count == 0:
        raise ValueError("dataset produced zero sessions")

    return TrajectoryEvaluation(
        agent=agent_name,
        sessions=sessions_count,
        total_steps=total_steps,
        avg_session_reward=float(np.mean(session_rewards)),
        avg_discounted_return=float(np.mean(session_disc_returns)),
        avg_session_length=float(np.mean(session_lengths)),
        avg_session_hit_rate=float(np.mean(session_hits)),
        avg_per_step_ctr=float(np.mean(per_step_ctrs)),
        avg_per_step_ndcg=float(np.mean(per_step_ndcgs)),
        avg_per_step_mrr=float(np.mean(per_step_mrrs)),
        seconds=float(perf_counter() - started),
    )
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /home/vishsangale/workspace/rl-recsys && .venv/bin/pytest tests/test_trajectory.py -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
cd /home/vishsangale/workspace/rl-recsys && \
  git add rl_recsys/evaluation/trajectory.py tests/test_trajectory.py && \
  git commit -m "feat: add evaluate_trajectory_agent with replay-mode reward rule"
```

---

## Task 2: `FinnNoSlateTrajectoryLoader`

**Files:**
- Create: `rl_recsys/data/loaders/__init__.py` (empty)
- Create: `rl_recsys/data/loaders/finn_no_slate_trajectory.py`
- Modify: `tests/test_trajectory.py` (append loader test)

- [ ] **Step 1: Append the loader test to `tests/test_trajectory.py`**

```python
import pandas as pd

from rl_recsys.data.loaders.finn_no_slate_trajectory import FinnNoSlateTrajectoryLoader


def test_finn_no_slate_loader_emits_sessions(tmp_path) -> None:
    # Synthetic mini parquet: 3 users, varying session lengths.
    df = pd.DataFrame(
        {
            "request_id": [0, 1, 2, 3, 4, 5, 6],
            "user_id": [10, 10, 10, 11, 11, 12, 12],
            "clicks": [0, 2, 1, 3, 0, 4, 2],  # position within slate
            "slate": [
                [100, 101, 102, 103, 104],  # user 10, click position 0 → id 100
                [200, 201, 202, 203, 204],  # user 10, click position 2 → id 202
                [300, 301, 302, 303, 304],  # user 10, click position 1 → id 301
                [400, 401, 402, 403, 404],  # user 11, click position 3 → id 403
                [500, 501, 502, 503, 504],  # user 11, click position 0 → id 500
                [600, 601, 602, 603, 604],  # user 12, click position 4 → id 604
                [700, 701, 702, 703, 704],  # user 12, click position 2 → id 702
            ],
        }
    )
    parquet_path = tmp_path / "slates.parquet"
    df.to_parquet(parquet_path, index=False)

    loader = FinnNoSlateTrajectoryLoader(
        parquet_path,
        num_candidates=8,
        feature_dim=4,
        slate_size=3,
        seed=0,
    )
    sessions = list(loader.iter_sessions())

    # 3 distinct users → 3 sessions
    assert len(sessions) == 3
    sessions_by_id = {s.session_id: s for s in sessions}
    assert sessions_by_id[10].steps[0].logged_clicked_id == 100
    assert sessions_by_id[10].steps[1].logged_clicked_id == 202
    assert sessions_by_id[10].steps[2].logged_clicked_id == 301
    assert len(sessions_by_id[11].steps) == 2
    assert sessions_by_id[11].steps[0].logged_clicked_id == 403
    assert len(sessions_by_id[12].steps) == 2

    # Verify candidate pool always contains all logged_slate items
    for session in sessions:
        for step in session.steps:
            assert step.obs.candidate_ids.shape == (8,)
            assert step.obs.candidate_features.shape == (8, 4)
            assert step.obs.user_features.shape == (4,)
            assert set(step.logged_slate.tolist()).issubset(
                step.obs.candidate_ids.tolist()
            )

    # Verify max_sessions cap
    capped = list(loader.iter_sessions(max_sessions=2))
    assert len(capped) == 2


def test_finn_no_slate_loader_rejects_small_num_candidates(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "request_id": [0],
            "user_id": [10],
            "clicks": [0],
            "slate": [[100, 101, 102, 103, 104]],  # 5-item slate
        }
    )
    parquet_path = tmp_path / "slates.parquet"
    df.to_parquet(parquet_path, index=False)

    with pytest.raises(ValueError, match="num_candidates"):
        FinnNoSlateTrajectoryLoader(
            parquet_path,
            num_candidates=3,  # smaller than slate length 5 → invalid
            feature_dim=4,
            slate_size=3,
            seed=0,
        )
```

- [ ] **Step 2: Run loader tests to confirm they fail**

```bash
cd /home/vishsangale/workspace/rl-recsys && .venv/bin/pytest tests/test_trajectory.py::test_finn_no_slate_loader_emits_sessions tests/test_trajectory.py::test_finn_no_slate_loader_rejects_small_num_candidates -v
```

Expected: `ImportError` — `rl_recsys.data.loaders.finn_no_slate_trajectory` does not exist.

- [ ] **Step 3: Create the `loaders` package**

```bash
mkdir -p /home/vishsangale/workspace/rl-recsys/rl_recsys/data/loaders
```

Create `rl_recsys/data/loaders/__init__.py` with content `"""Trajectory dataset loaders."""\n` only.

- [ ] **Step 4: Create `rl_recsys/data/loaders/finn_no_slate_trajectory.py`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from rl_recsys.environments.base import RecObs
from rl_recsys.environments.features import hashed_vector
from rl_recsys.evaluation.trajectory import Session, TrajectoryStep


class FinnNoSlateTrajectoryLoader:
    """TrajectoryDataset over data/processed/finn-no-slate/slates.parquet.

    Schema (per pipeline finn_no_slate.py): request_id, user_id, clicks, slate.
    The pipeline pre-filters to clicked rows only — every parquet row has a
    real click, and the `clicks` column stores the position of the click
    within `slate`. Therefore logged_clicked_id = slate[clicks].
    """

    def __init__(
        self,
        parquet_path: str | Path,
        *,
        num_candidates: int,
        feature_dim: int,
        slate_size: int,
        seed: int = 0,
    ) -> None:
        self._df = pd.read_parquet(parquet_path)
        required = {"request_id", "user_id", "clicks", "slate"}
        missing = required - set(self._df.columns)
        if missing:
            raise ValueError(
                f"finn-no-slate parquet missing columns: {sorted(missing)}"
            )
        # Determine logged-slate length from the first row to validate num_candidates.
        first_slate = np.asarray(self._df.iloc[0]["slate"], dtype=np.int64)
        slate_len = int(first_slate.shape[0])
        if num_candidates < slate_len:
            raise ValueError(
                f"num_candidates={num_candidates} must be >= "
                f"logged slate length={slate_len}"
            )
        if feature_dim < 1:
            raise ValueError("feature_dim must be at least 1")

        self._num_candidates = int(num_candidates)
        self._feature_dim = int(feature_dim)
        self._slate_size = int(slate_size)
        self._seed = int(seed)

        # Item universe for padding candidate pools — union of all observed
        # slate item IDs. For a synthetic test parquet this is ~tens of items;
        # for the real 28M-row parquet it's ~hundreds of thousands.
        all_items = np.concatenate(
            [np.asarray(row, dtype=np.int64) for row in self._df["slate"].to_numpy()]
        )
        self._item_universe = np.unique(all_items)

    def iter_sessions(
        self, *, max_sessions: int | None = None, seed: int | None = None
    ) -> Iterator[Session]:
        ordered = self._df.sort_values(["user_id", "request_id"], kind="stable")
        groups = ordered.groupby("user_id", sort=False)
        rng = np.random.default_rng(self._seed if seed is None else seed)

        emitted = 0
        for user_id, group in groups:
            if max_sessions is not None and emitted >= max_sessions:
                break
            steps: list[TrajectoryStep] = []
            for _, row in group.iterrows():
                logged_slate = np.asarray(row["slate"], dtype=np.int64)
                clicked_id = int(logged_slate[int(row["clicks"])])
                candidate_ids = self._build_candidate_ids(logged_slate, rng)
                candidate_features = np.stack(
                    [hashed_vector("item", int(i), self._feature_dim) for i in candidate_ids]
                )
                user_features = hashed_vector("user", int(user_id), self._feature_dim)
                obs = RecObs(
                    user_features=user_features,
                    candidate_features=candidate_features,
                    candidate_ids=candidate_ids,
                )
                steps.append(
                    TrajectoryStep(
                        obs=obs,
                        logged_slate=logged_slate,
                        logged_clicked_id=clicked_id,
                        logged_reward=1.0,
                    )
                )
            yield Session(session_id=int(user_id), steps=steps)
            emitted += 1

    def _build_candidate_ids(
        self, logged_slate: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        n_pad = self._num_candidates - len(logged_slate)
        if n_pad == 0:
            return logged_slate.copy()
        pool = self._item_universe[~np.isin(self._item_universe, logged_slate)]
        if len(pool) < n_pad:
            # Universe too small (tiny test data) — pad with synthetic large IDs.
            extra = np.arange(
                self._item_universe.max() + 1,
                self._item_universe.max() + 1 + (n_pad - len(pool)),
                dtype=np.int64,
            )
            pad_pool = np.concatenate([pool, extra])
            pad = pad_pool[:n_pad]
        else:
            pad = rng.choice(pool, size=n_pad, replace=False)
        return np.concatenate([logged_slate, pad.astype(np.int64)])
```

- [ ] **Step 5: Run loader tests to confirm they pass**

```bash
cd /home/vishsangale/workspace/rl-recsys && .venv/bin/pytest tests/test_trajectory.py -v
```

Expected: all 7 tests PASSED (5 evaluator + 2 loader).

- [ ] **Step 6: Commit**

```bash
cd /home/vishsangale/workspace/rl-recsys && \
  git add rl_recsys/data/loaders/__init__.py \
          rl_recsys/data/loaders/finn_no_slate_trajectory.py \
          tests/test_trajectory.py && \
  git commit -m "feat: add FinnNoSlateTrajectoryLoader"
```

---

## Task 3: Generalize `evaluate_with_variance` via dataclass introspection

**Files:**
- Modify: `rl_recsys/evaluation/variance.py`
- Modify: `tests/test_variance.py`

- [ ] **Step 1: Append the introspection test to `tests/test_variance.py`**

```python
from dataclasses import dataclass


@dataclass
class _FakeResult:
    agent: str           # non-numeric — should be filtered
    score: float
    count: int
    seconds: float


def test_evaluate_with_variance_introspects_dataclass_fields(tmp_path) -> None:
    # Direct test of the _aggregate_runs helper via a synthetic list of results.
    from rl_recsys.evaluation.variance import _aggregate_runs

    runs = [
        _FakeResult(agent="x", score=1.0, count=10, seconds=0.5),
        _FakeResult(agent="x", score=3.0, count=12, seconds=0.6),
        _FakeResult(agent="x", score=5.0, count=14, seconds=0.7),
    ]
    mean, std = _aggregate_runs(runs)

    # String field 'agent' must be filtered out
    assert "agent" not in mean
    assert "agent" not in std
    # Numeric fields present
    assert set(mean.keys()) == {"score", "count", "seconds"}
    assert mean["score"] == pytest.approx(3.0)
    assert mean["count"] == pytest.approx(12.0)
    assert std["score"] == pytest.approx(np.std([1.0, 3.0, 5.0]))
```

- [ ] **Step 2: Run the new test to confirm it fails**

```bash
cd /home/vishsangale/workspace/rl-recsys && .venv/bin/pytest tests/test_variance.py::test_evaluate_with_variance_introspects_dataclass_fields -v
```

Expected: `ImportError` — `_aggregate_runs` does not exist yet.

- [ ] **Step 3: Refactor `rl_recsys/evaluation/variance.py`**

Replace the entire content with:

```python
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable

import numpy as np

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecEnv
from rl_recsys.evaluation.bandit import evaluate_bandit_agent


@dataclass
class VarianceEvaluation:
    mean: dict[str, float]
    std: dict[str, float]
    n_seeds: int


def _aggregate_runs(
    results: list[Any],
) -> tuple[dict[str, float], dict[str, float]]:
    """Mean and std over scalar (int/float) fields shared by all dataclass results."""
    if not results:
        return {}, {}
    numeric_keys: list[str] = []
    for f in fields(results[0]):
        ftype = f.type
        if ftype in (float, int) or ftype in ("float", "int"):
            numeric_keys.append(f.name)
    runs = {k: [getattr(r, k) for r in results] for k in numeric_keys}
    mean = {k: float(np.mean(v)) for k, v in runs.items()}
    std = {k: float(np.std(v)) for k, v in runs.items()}
    return mean, std


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
    """Run evaluate_bandit_agent n_seeds times; return mean ± std per metric.

    make_env and make_agent are called fresh each seed to prevent state leakage.
    Default n_seeds=5 matches the RL4RS paper's reporting convention.
    """
    results = [
        evaluate_bandit_agent(
            make_env(),
            make_agent(),
            agent_name=agent_name,
            episodes=episodes,
            seed=base_seed + i,
            gamma=gamma,
        )
        for i in range(n_seeds)
    ]
    mean, std = _aggregate_runs(results)
    return VarianceEvaluation(mean=mean, std=std, n_seeds=n_seeds)
```

- [ ] **Step 4: Run the full variance test file to confirm everything still passes**

```bash
cd /home/vishsangale/workspace/rl-recsys && .venv/bin/pytest tests/test_variance.py -v
```

Expected: 4 PASSED (3 existing + 1 new introspection test).

- [ ] **Step 5: Commit**

```bash
cd /home/vishsangale/workspace/rl-recsys && \
  git add rl_recsys/evaluation/variance.py tests/test_variance.py && \
  git commit -m "refactor: generalize evaluate_with_variance via dataclass introspection"
```

---

## Task 4: `evaluate_trajectory_with_variance`

**Files:**
- Modify: `rl_recsys/evaluation/variance.py`
- Modify: `tests/test_variance.py`

- [ ] **Step 1: Append the trajectory variance test to `tests/test_variance.py`**

```python
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
    for key in (
        "sessions",
        "total_steps",
        "avg_session_reward",
        "avg_discounted_return",
        "avg_session_length",
        "avg_session_hit_rate",
        "avg_per_step_ctr",
        "avg_per_step_ndcg",
        "avg_per_step_mrr",
        "seconds",
    ):
        assert key in result.mean
        assert key in result.std
        assert np.isfinite(result.mean[key])
        assert np.isfinite(result.std[key])
```

- [ ] **Step 2: Run the new test to confirm it fails**

```bash
cd /home/vishsangale/workspace/rl-recsys && .venv/bin/pytest tests/test_variance.py::test_evaluate_trajectory_with_variance_returns_finite_mean_and_std -v
```

Expected: `ImportError` — `evaluate_trajectory_with_variance` does not exist yet.

- [ ] **Step 3: Append `evaluate_trajectory_with_variance` to `rl_recsys/evaluation/variance.py`**

Add these imports at the top (alongside the existing imports):

```python
from rl_recsys.evaluation.trajectory import (
    TrajectoryDataset,
    evaluate_trajectory_agent,
)
```

Then append at the bottom of the file:

```python
def evaluate_trajectory_with_variance(
    make_dataset: Callable[[], TrajectoryDataset],
    make_agent: Callable[[], Agent],
    *,
    agent_name: str,
    max_sessions: int,
    n_seeds: int = 5,
    base_seed: int = 42,
    gamma: float = 0.95,
) -> VarianceEvaluation:
    """Run evaluate_trajectory_agent n_seeds times; return mean ± std per metric."""
    results = [
        evaluate_trajectory_agent(
            make_dataset(),
            make_agent(),
            agent_name=agent_name,
            max_sessions=max_sessions,
            seed=base_seed + i,
            gamma=gamma,
        )
        for i in range(n_seeds)
    ]
    mean, std = _aggregate_runs(results)
    return VarianceEvaluation(mean=mean, std=std, n_seeds=n_seeds)
```

- [ ] **Step 4: Run the full variance test file**

```bash
cd /home/vishsangale/workspace/rl-recsys && .venv/bin/pytest tests/test_variance.py -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
cd /home/vishsangale/workspace/rl-recsys && \
  git add rl_recsys/evaluation/variance.py tests/test_variance.py && \
  git commit -m "feat: add evaluate_trajectory_with_variance"
```

---

## Task 5: Update `__init__.py` exports

**Files:**
- Modify: `rl_recsys/evaluation/__init__.py`

- [ ] **Step 1: Read the current file to see what exports already exist**

```bash
cd /home/vishsangale/workspace/rl-recsys && cat rl_recsys/evaluation/__init__.py
```

- [ ] **Step 2: Replace the entire content of `rl_recsys/evaluation/__init__.py` with**

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
    "OPEEvaluation",
    "OPERecord",
    "Session",
    "TrajectoryDataset",
    "TrajectoryEvaluation",
    "TrajectoryStep",
    "VarianceEvaluation",
    "dr_value",
    "evaluate_ope_agent",
    "evaluate_trajectory_agent",
    "evaluate_trajectory_with_variance",
    "evaluate_with_variance",
    "ips_value",
    "replay_value",
    "snips_value",
    "swis_value",
]
```

- [ ] **Step 3: Run the full test suite**

```bash
cd /home/vishsangale/workspace/rl-recsys && .venv/bin/pytest tests/ -q --tb=short
```

Expected: all tests pass, no regressions. Total = previous 144 + 5 evaluator + 2 loader + 2 variance = 153.

- [ ] **Step 4: Commit**

```bash
cd /home/vishsangale/workspace/rl-recsys && \
  git add rl_recsys/evaluation/__init__.py && \
  git commit -m "chore: export trajectory evaluation symbols from evaluation package"
```
