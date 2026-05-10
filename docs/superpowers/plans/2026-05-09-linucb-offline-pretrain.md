# Offline LinUCB Pretraining for RL4RS-B Sequential DR — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give LinUCB an offline training pass over logged trajectories before Sequential DR evaluation, with a clean session-level train/eval split, so the RL4RS-B benchmark becomes discriminative between LinUCB and Random.

**Architecture:** Extend `LoggedTrajectoryStep` with per-position `logged_clicks` so `Agent.update()` can be replayed verbatim. Add `session_filter` to `RL4RSTrajectoryOPESource` for disjoint train/eval halves under a shared candidate universe + behavior policy. New generic helper `pretrain_agent_on_logged(agent, source)` does a single pass calling `agent.update(...)`. Benchmark script wires train→pretrain→eval per seed.

**Tech Stack:** Python 3, numpy, pandas, pyarrow, PyTorch (already in repo), pytest. All commands use `.venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-05-09-linucb-offline-pretrain-design.md`

---

## File map

- Modify: `rl_recsys/evaluation/ope_trajectory.py` — add `logged_clicks` field
- Modify: `rl_recsys/data/loaders/rl4rs_trajectory_ope.py` — populate `logged_clicks`, add `session_filter`
- Create: `rl_recsys/training/offline_pretrain.py` — new pretrain helper
- Modify: `rl_recsys/training/__init__.py` — re-export `pretrain_agent_on_logged`
- Create: `rl_recsys/training/session_split.py` — `split_session_ids` helper
- Modify: `scripts/benchmark_rl4rs_b_seq_dr.py` — wire train/eval/pretrain
- Modify: `tests/test_ope_trajectory.py` — backfill `logged_clicks` in fixtures
- Modify: `tests/test_variance.py` — backfill `logged_clicks` in fixtures
- Modify: `tests/test_rl4rs_trajectory_ope.py` — extend with new assertions
- Create: `tests/test_offline_pretrain.py`
- Create: `tests/test_session_split.py`

---

## Task 1: Extend `LoggedTrajectoryStep` with `logged_clicks`

**Files:**
- Modify: `rl_recsys/evaluation/ope_trajectory.py:45-50` (dataclass)
- Modify: `rl_recsys/data/loaders/rl4rs_trajectory_ope.py:106-112` (single construction site)
- Modify: `tests/test_ope_trajectory.py` (multiple fixture sites)
- Modify: `tests/test_variance.py:199-209`
- Modify: `tests/test_rl4rs_trajectory_ope.py` (already constructs via loader, just check)

The existing dataclass has `obs`, `logged_action`, `logged_reward`, `propensity`. We add `logged_clicks: np.ndarray` of shape `(slate_size,)` carrying per-position binary feedback. Loader constructions and unit-test fixtures must be backfilled. Existing OPE consumers (`seq_dr_value`, `evaluate_trajectory_ope_agent`) ignore the new field.

- [ ] **Step 1: Add the failing assertion to the loader test**

Edit `tests/test_rl4rs_trajectory_ope.py`. Add a new test below `test_loader_emits_trajectories_grouped_by_session`:

```python
def test_loader_emits_logged_clicks(tmp_path: Path) -> None:
    from rl_recsys.data.loaders.rl4rs_trajectory_ope import (
        RL4RSTrajectoryOPESource,
    )
    parquet = _fixture_b_parquet(tmp_path)

    model = BehaviorPolicy(
        user_dim=2, item_dim=2, slate_size=2, num_items=3,
        hidden_dim=4, seed=0,
    )
    source = RL4RSTrajectoryOPESource(
        parquet_path=parquet, behavior_policy=model, slate_size=2,
    )
    trajectories = list(source.iter_trajectories(max_trajectories=10, seed=0))

    s1 = next(t for t in trajectories if len(t) == 2)
    s2 = next(t for t in trajectories if len(t) == 1)

    # logged_clicks shape == (slate_size,) and matches user_feedback verbatim
    np.testing.assert_array_equal(s1[0].logged_clicks, np.array([1, 0]))
    np.testing.assert_array_equal(s1[1].logged_clicks, np.array([0, 1]))
    np.testing.assert_array_equal(s2[0].logged_clicks, np.array([0, 0]))
    assert s1[0].logged_clicks.dtype == np.int64
```

- [ ] **Step 2: Run the test to verify it fails**

```
.venv/bin/python -m pytest tests/test_rl4rs_trajectory_ope.py::test_loader_emits_logged_clicks -v
```

Expected: FAIL with `AttributeError: 'LoggedTrajectoryStep' object has no attribute 'logged_clicks'`.

- [ ] **Step 3: Add the field to the dataclass**

Edit `rl_recsys/evaluation/ope_trajectory.py`, replace lines 45-50:

```python
@dataclass(frozen=True)
class LoggedTrajectoryStep:
    obs: RecObs
    logged_action: np.ndarray  # shape (slate_size,) — candidate indices (positions in obs.candidate_ids)
    logged_reward: float
    logged_clicks: np.ndarray  # shape (slate_size,) — per-position binary feedback
    propensity: float          # μ(slate | obs) = Π_k μ(slate[k] | obs, k)
```

- [ ] **Step 4: Populate `logged_clicks` in the loader**

Edit `rl_recsys/data/loaders/rl4rs_trajectory_ope.py`. After the `logged_reward` line in `iter_trajectories` (around line 83), add:

```python
logged_clicks = np.array(
    list(row["user_feedback"]), dtype=np.int64
)
```

And in the `LoggedTrajectoryStep(...)` construction (around line 106), add the field:

```python
steps.append(
    LoggedTrajectoryStep(
        obs=obs,
        logged_action=slate_indices,
        logged_reward=logged_reward,
        logged_clicks=logged_clicks,
        propensity=propensity,
    )
)
```

- [ ] **Step 5: Backfill fixtures in `tests/test_ope_trajectory.py`**

Every site that constructs `LoggedTrajectoryStep(...)` directly needs `logged_clicks=np.zeros(slate_size, dtype=np.int64)` (slate_size matching the existing `logged_action` shape on that line). Sites to patch (use grep to confirm):

```
tests/test_ope_trajectory.py:149  — slate_size=2
tests/test_ope_trajectory.py:153  — slate_size=2
tests/test_ope_trajectory.py:159  — slate_size=2
tests/test_ope_trajectory.py:230  — slate_size=1
tests/test_ope_trajectory.py:231  — slate_size=1
tests/test_ope_trajectory.py:251  — read the surrounding code for slate_size
tests/test_ope_trajectory.py:318  — read the surrounding code for slate_size
tests/test_ope_trajectory.py:346  — read the surrounding code for slate_size
```

For each call site, read the existing `logged_action` argument to determine the slate_size, then add the new keyword. Example patch pattern:

```python
# Before
LoggedTrajectoryStep(
    obs=obs,
    logged_action=np.array([0], dtype=np.int64),
    logged_reward=1.0,
    propensity=0.5,
),
# After
LoggedTrajectoryStep(
    obs=obs,
    logged_action=np.array([0], dtype=np.int64),
    logged_reward=1.0,
    logged_clicks=np.zeros(1, dtype=np.int64),
    propensity=0.5,
),
```

- [ ] **Step 6: Backfill fixtures in `tests/test_variance.py`**

Patch the constructions around lines 199 and 205 the same way. Check the surrounding `logged_action` to determine slate_size.

- [ ] **Step 7: Run the full test suite to verify everything is green**

```
.venv/bin/python -m pytest -q
```

Expected: all tests pass, including the new `test_loader_emits_logged_clicks`.

- [ ] **Step 8: Commit**

```bash
git add rl_recsys/evaluation/ope_trajectory.py rl_recsys/data/loaders/rl4rs_trajectory_ope.py tests/test_ope_trajectory.py tests/test_variance.py tests/test_rl4rs_trajectory_ope.py
git commit -m "feat: add logged_clicks to LoggedTrajectoryStep

Per-position binary feedback needed for offline Agent.update()
replay. Loader populates from user_feedback; existing OPE consumers
ignore the new field. Test fixtures backfilled with zeros."
```

---

## Task 2: Add `session_filter` to `RL4RSTrajectoryOPESource`

**Files:**
- Modify: `rl_recsys/data/loaders/rl4rs_trajectory_ope.py`
- Modify: `tests/test_rl4rs_trajectory_ope.py`

A new optional constructor argument `session_filter: set[int] | None = None`. When provided, `iter_trajectories` skips sessions whose `session_id` is not in the filter. The candidate universe is built from the FULL parquet regardless of the filter — propensities and candidate features must remain comparable between train and eval halves.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rl4rs_trajectory_ope.py`:

```python
def test_loader_session_filter_yields_only_filtered_sessions(tmp_path: Path) -> None:
    from rl_recsys.data.loaders.rl4rs_trajectory_ope import (
        RL4RSTrajectoryOPESource,
    )
    parquet = _fixture_b_parquet(tmp_path)
    model = BehaviorPolicy(
        user_dim=2, item_dim=2, slate_size=2, num_items=3,
        hidden_dim=4, seed=0,
    )

    # Filter to only session 1
    source = RL4RSTrajectoryOPESource(
        parquet_path=parquet, behavior_policy=model, slate_size=2,
        session_filter={1},
    )
    trajectories = list(source.iter_trajectories(max_trajectories=10, seed=0))

    # Only session 1 emitted (it has 2 steps)
    assert len(trajectories) == 1
    assert len(trajectories[0]) == 2
    # Universe still includes all items from the full parquet (10, 11, 12)
    assert source._candidate_ids.tolist() == [10, 11, 12]


def test_loader_empty_session_filter_raises_on_iter(tmp_path: Path) -> None:
    from rl_recsys.data.loaders.rl4rs_trajectory_ope import (
        RL4RSTrajectoryOPESource,
    )
    import pytest
    parquet = _fixture_b_parquet(tmp_path)
    model = BehaviorPolicy(
        user_dim=2, item_dim=2, slate_size=2, num_items=3,
        hidden_dim=4, seed=0,
    )
    # Filter excludes everything
    source = RL4RSTrajectoryOPESource(
        parquet_path=parquet, behavior_policy=model, slate_size=2,
        session_filter={9999},
    )
    with pytest.raises(ValueError, match="session_filter"):
        list(source.iter_trajectories(max_trajectories=10, seed=0))
```

- [ ] **Step 2: Run tests to verify they fail**

```
.venv/bin/python -m pytest tests/test_rl4rs_trajectory_ope.py::test_loader_session_filter_yields_only_filtered_sessions tests/test_rl4rs_trajectory_ope.py::test_loader_empty_session_filter_raises_on_iter -v
```

Expected: both FAIL — first with `TypeError: unexpected keyword argument 'session_filter'`.

- [ ] **Step 3: Implement the constructor argument**

Edit `rl_recsys/data/loaders/rl4rs_trajectory_ope.py`. Modify `__init__`:

```python
def __init__(
    self,
    parquet_path: str | Path,
    behavior_policy: BehaviorPolicy,
    *,
    slate_size: int,
    session_filter: set[int] | None = None,
) -> None:
    self._df = pd.read_parquet(parquet_path)
    self._policy = behavior_policy
    self._slate_size = int(slate_size)
    self._session_filter = (
        None if session_filter is None else {int(s) for s in session_filter}
    )

    # ... rest unchanged (universe build over full parquet) ...
```

- [ ] **Step 4: Apply the filter in `iter_trajectories`**

In `iter_trajectories`, after `session_ids = list(groups.groups.keys())` and the optional shuffle, add the filter step. Then add an empty-emit guard at the end of the function. Replace the loop with:

```python
def iter_trajectories(
    self, *, max_trajectories: int | None = None, seed: int | None = None
) -> Iterator[list[LoggedTrajectoryStep]]:
    ordered = self._df.sort_values(["session_id", "sequence_id"], kind="stable")
    groups = ordered.groupby("session_id", sort=False)
    session_ids = list(groups.groups.keys())
    rng = np.random.default_rng(0 if seed is None else seed)
    if seed is not None:
        rng.shuffle(session_ids)

    if self._session_filter is not None:
        session_ids = [
            sid for sid in session_ids if int(sid) in self._session_filter
        ]
        if not session_ids:
            raise ValueError(
                "session_filter excludes every session in the parquet — "
                "no trajectories to emit"
            )

    emitted = 0
    for sid in session_ids:
        # ... existing per-session loop unchanged ...
```

(Keep the rest of the loop body identical.)

- [ ] **Step 5: Run the new tests + full suite**

```
.venv/bin/python -m pytest tests/test_rl4rs_trajectory_ope.py -v
.venv/bin/python -m pytest -q
```

Expected: new tests PASS, all existing tests PASS.

- [ ] **Step 6: Commit**

```bash
git add rl_recsys/data/loaders/rl4rs_trajectory_ope.py tests/test_rl4rs_trajectory_ope.py
git commit -m "feat: add session_filter to RL4RSTrajectoryOPESource

Optional constructor arg restricts iter_trajectories to a subset of
session_ids while keeping the candidate universe global. Enables a
disjoint train/eval split for offline pretraining without forking
the parquet."
```

---

## Task 3: Implement `split_session_ids`

**Files:**
- Create: `rl_recsys/training/session_split.py`
- Create: `tests/test_session_split.py`

Deterministic 50/50 (configurable) session-level partition. Hashes `(seed, sid)` and bucketed thresholding so runs are reproducible.

- [ ] **Step 1: Write failing tests**

Create `tests/test_session_split.py`:

```python
from pathlib import Path

import pandas as pd
import pytest

from rl_recsys.training.session_split import split_session_ids


def _make_parquet(tmp_path: Path, n_sessions: int = 1000) -> Path:
    rows = []
    for sid in range(n_sessions):
        rows.append({
            "session_id": sid, "sequence_id": 1,
            "user_state": [1.0, 0.0],
            "slate": [10, 11], "user_feedback": [1, 0],
            "item_features": [[0.0, 0.0], [1.0, 0.0]],
        })
    p = tmp_path / "sessions_b.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    return p


def test_split_is_deterministic(tmp_path: Path) -> None:
    p = _make_parquet(tmp_path, n_sessions=200)
    a_train, a_eval = split_session_ids(p, train_fraction=0.5, seed=42)
    b_train, b_eval = split_session_ids(p, train_fraction=0.5, seed=42)
    assert a_train == b_train
    assert a_eval == b_eval


def test_split_is_disjoint_and_complete(tmp_path: Path) -> None:
    p = _make_parquet(tmp_path, n_sessions=200)
    train, evl = split_session_ids(p, train_fraction=0.5, seed=42)
    assert train.isdisjoint(evl)
    assert train | evl == set(range(200))


def test_split_fraction_approximate(tmp_path: Path) -> None:
    p = _make_parquet(tmp_path, n_sessions=1000)
    train, evl = split_session_ids(p, train_fraction=0.5, seed=42)
    # ±5% tolerance at n=1000
    assert 450 <= len(train) <= 550
    assert 450 <= len(evl) <= 550


def test_split_rejects_invalid_fractions(tmp_path: Path) -> None:
    p = _make_parquet(tmp_path, n_sessions=10)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="train_fraction"):
            split_session_ids(p, train_fraction=bad, seed=0)


def test_split_rejects_empty_split(tmp_path: Path) -> None:
    # train_fraction=0.999 with only 2 sessions is likely to put both in train
    rows = [
        {"session_id": 1, "sequence_id": 1, "user_state": [1.0, 0.0],
         "slate": [10, 11], "user_feedback": [1, 0],
         "item_features": [[0.0, 0.0], [1.0, 0.0]]},
        {"session_id": 2, "sequence_id": 1, "user_state": [1.0, 0.0],
         "slate": [10, 11], "user_feedback": [0, 0],
         "item_features": [[0.0, 0.0], [1.0, 0.0]]},
    ]
    p = tmp_path / "sessions_b.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)

    # Try a few seeds to find one that produces an empty split — if none do
    # within reasonable attempts, skip; the API contract still holds when an
    # empty split occurs.
    raised = False
    for seed in range(50):
        try:
            split_session_ids(p, train_fraction=0.999, seed=seed)
        except ValueError as e:
            if "empty" in str(e):
                raised = True
                break
    assert raised, "expected at least one seed/fraction combo to trigger empty-split error"
```

- [ ] **Step 2: Run tests to verify they fail**

```
.venv/bin/python -m pytest tests/test_session_split.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'rl_recsys.training.session_split'`.

- [ ] **Step 3: Implement the helper**

Create `rl_recsys/training/session_split.py`:

```python
from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq


def _bucket(sid: int, seed: int) -> int:
    """Return a deterministic integer in [0, 1000) for (seed, sid)."""
    h = hashlib.blake2b(
        f"{seed}:{sid}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(h, "big") % 1000


def split_session_ids(
    parquet_path: str | Path,
    *,
    train_fraction: float = 0.5,
    seed: int = 42,
) -> tuple[set[int], set[int]]:
    """Deterministic session-level partition of a sessions_b.parquet.

    Reads only the session_id column. For each unique session_id,
    bucket = blake2b(seed:sid) % 1000; if bucket < int(1000 * train_fraction)
    the session is in the train half, else eval.

    Returns (train_ids, eval_ids). Raises ValueError if train_fraction is
    not in (0, 1) or if either side is empty.
    """
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(
            f"train_fraction must be in (0, 1); got {train_fraction!r}"
        )

    table = pq.read_table(parquet_path, columns=["session_id"])
    unique_ids = pc.unique(table["session_id"]).to_pylist()
    threshold = int(1000 * train_fraction)

    train: set[int] = set()
    evl: set[int] = set()
    for sid in unique_ids:
        if _bucket(int(sid), seed) < threshold:
            train.add(int(sid))
        else:
            evl.add(int(sid))

    if not train or not evl:
        raise ValueError(
            f"split produced an empty side (train={len(train)}, eval={len(evl)}); "
            "try a different train_fraction or seed"
        )
    return train, evl
```

- [ ] **Step 4: Run the new tests**

```
.venv/bin/python -m pytest tests/test_session_split.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add rl_recsys/training/session_split.py tests/test_session_split.py
git commit -m "feat: add deterministic session_id splitter

split_session_ids partitions sessions_b.parquet into train/eval halves
via blake2b(seed:sid) % 1000 bucketing. Stable across runs, disjoint by
construction, raises on degenerate fractions/empty sides."
```

---

## Task 4: Implement `pretrain_agent_on_logged`

**Files:**
- Create: `rl_recsys/training/offline_pretrain.py`
- Modify: `rl_recsys/training/__init__.py`
- Create: `tests/test_offline_pretrain.py`

Generic helper: walks `source.iter_trajectories(...)`, calls `agent.update(...)` on every step. Returns aggregate metrics. Single pass — no epochs.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_offline_pretrain.py`:

```python
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pytest

from rl_recsys.agents import LinUCBAgent, RandomAgent
from rl_recsys.environments.base import RecObs
from rl_recsys.evaluation.ope_trajectory import LoggedTrajectoryStep


class _SpyAgent:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def select_slate(self, obs):  # pragma: no cover — pretrain doesn't call this
        return np.array([0], dtype=np.int64)

    def score_items(self, obs):  # pragma: no cover — pretrain doesn't call this
        return np.zeros(len(obs.candidate_features), dtype=np.float64)

    def update(self, obs, slate, reward, clicks, next_obs):
        self.calls.append({
            "obs_id": id(obs),
            "next_obs_id": id(next_obs),
            "slate": np.array(slate, copy=True),
            "reward": float(reward),
            "clicks": np.array(clicks, copy=True),
        })
        return {}


class _ListSource:
    def __init__(self, trajectories: list[list[LoggedTrajectoryStep]]) -> None:
        self._t = trajectories

    def iter_trajectories(
        self, *, max_trajectories=None, seed=None
    ) -> Iterator[list[LoggedTrajectoryStep]]:
        for t in self._t[: max_trajectories if max_trajectories else len(self._t)]:
            yield t


def _make_step(slate=(0, 1), clicks=(1, 0), reward=1.0, propensity=0.25):
    obs = RecObs(
        user_features=np.array([1.0, 0.0]),
        candidate_features=np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]]),
        candidate_ids=np.array([10, 11, 12], dtype=np.int64),
    )
    return LoggedTrajectoryStep(
        obs=obs,
        logged_action=np.array(slate, dtype=np.int64),
        logged_reward=reward,
        logged_clicks=np.array(clicks, dtype=np.int64),
        propensity=propensity,
    )


def test_pretrain_calls_update_per_step() -> None:
    from rl_recsys.training.offline_pretrain import pretrain_agent_on_logged

    s1 = [_make_step(slate=(0, 1), clicks=(1, 0)),
          _make_step(slate=(1, 2), clicks=(0, 1))]
    s2 = [_make_step(slate=(0, 2), clicks=(0, 0))]
    source = _ListSource([s1, s2])
    agent = _SpyAgent()

    metrics = pretrain_agent_on_logged(agent, source)

    assert len(agent.calls) == 3
    np.testing.assert_array_equal(agent.calls[0]["clicks"], np.array([1, 0]))
    np.testing.assert_array_equal(agent.calls[1]["clicks"], np.array([0, 1]))
    np.testing.assert_array_equal(agent.calls[2]["clicks"], np.array([0, 0]))
    assert agent.calls[0]["obs_id"] == agent.calls[0]["next_obs_id"]
    assert metrics["trajectories"] == 2
    assert metrics["total_steps"] == 3
    assert np.isfinite(metrics["seconds"])


def test_pretrain_changes_linucb_state() -> None:
    from rl_recsys.training.offline_pretrain import pretrain_agent_on_logged

    s1 = [_make_step(slate=(0, 1), clicks=(1, 0)),
          _make_step(slate=(1, 2), clicks=(0, 1))]
    source = _ListSource([s1])
    agent = LinUCBAgent(slate_size=2, user_dim=2, item_dim=2, alpha=1.0)
    a_before = agent._a_matrix.copy()
    b_before = agent._b_vector.copy()

    pretrain_agent_on_logged(agent, source)

    assert not np.allclose(agent._a_matrix, a_before)
    assert not np.allclose(agent._b_vector, b_before)


def test_pretrain_random_is_noop_safe() -> None:
    from rl_recsys.training.offline_pretrain import pretrain_agent_on_logged

    s1 = [_make_step(slate=(0, 1), clicks=(1, 0))]
    source = _ListSource([s1])
    agent = RandomAgent(slate_size=2, seed=0)

    metrics = pretrain_agent_on_logged(agent, source)
    assert metrics["total_steps"] == 1
    assert np.isfinite(metrics["seconds"])


def test_pretrain_raises_on_empty_source() -> None:
    from rl_recsys.training.offline_pretrain import pretrain_agent_on_logged

    source = _ListSource([])
    agent = _SpyAgent()
    with pytest.raises(ValueError, match="zero trajectories"):
        pretrain_agent_on_logged(agent, source)
```

- [ ] **Step 2: Run tests to verify they fail**

```
.venv/bin/python -m pytest tests/test_offline_pretrain.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'rl_recsys.training.offline_pretrain'`.

- [ ] **Step 3: Implement the helper**

Create `rl_recsys/training/offline_pretrain.py`:

```python
from __future__ import annotations

from time import perf_counter

from rl_recsys.agents.base import Agent
from rl_recsys.evaluation.ope_trajectory import LoggedTrajectorySource


def pretrain_agent_on_logged(
    agent: Agent,
    source: LoggedTrajectorySource,
    *,
    max_trajectories: int | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Single offline pass over logged trajectories, calling agent.update().

    For each LoggedTrajectoryStep, calls
        agent.update(
            obs=step.obs,
            slate=step.logged_action,
            reward=step.logged_reward,
            clicks=step.logged_clicks,
            next_obs=step.obs,
        )

    `next_obs == obs` because LinUCB ignores it (contextual bandit) and
    Random ignores everything; we don't fabricate state evolution.

    Returns aggregate metrics: trajectories, total_steps, mean_click_rate,
    seconds.

    Raises ValueError if source yields zero trajectories.
    """
    started = perf_counter()
    n_traj = 0
    n_steps = 0
    total_clicks = 0.0
    total_slate_positions = 0

    for traj in source.iter_trajectories(
        max_trajectories=max_trajectories, seed=seed
    ):
        if not traj:
            continue
        n_traj += 1
        for step in traj:
            agent.update(
                obs=step.obs,
                slate=step.logged_action,
                reward=step.logged_reward,
                clicks=step.logged_clicks,
                next_obs=step.obs,
            )
            n_steps += 1
            total_clicks += float(step.logged_clicks.sum())
            total_slate_positions += int(step.logged_clicks.shape[0])

    if n_traj == 0:
        raise ValueError("source produced zero trajectories")

    return {
        "trajectories": float(n_traj),
        "total_steps": float(n_steps),
        "mean_click_rate": (
            total_clicks / total_slate_positions if total_slate_positions else 0.0
        ),
        "seconds": float(perf_counter() - started),
    }
```

- [ ] **Step 4: Re-export from the training package**

Edit `rl_recsys/training/__init__.py`. Add (preserving existing exports):

```python
from rl_recsys.training.offline_pretrain import pretrain_agent_on_logged
from rl_recsys.training.session_split import split_session_ids
```

If the file uses an `__all__` list, append `"pretrain_agent_on_logged"` and `"split_session_ids"` to it. If you're unsure of the existing structure, read the file first.

- [ ] **Step 5: Run the new tests**

```
.venv/bin/python -m pytest tests/test_offline_pretrain.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add rl_recsys/training/offline_pretrain.py rl_recsys/training/__init__.py tests/test_offline_pretrain.py
git commit -m "feat: add pretrain_agent_on_logged helper

Generic single-pass offline trainer that walks a LoggedTrajectorySource
and calls agent.update() per step using logged_clicks. Random no-ops
unchanged; LinUCB accumulates A/b sufficient stats."
```

---

## Task 5: End-to-end smoke — pretrain changes Sequential DR result

**Files:**
- Modify: `tests/test_rl4rs_trajectory_ope.py`

Add a single end-to-end test asserting that pretraining LinUCB on a train half changes `avg_seq_dr_value` versus a fresh LinUCB on the same eval half. This proves the wiring has an observable effect.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rl4rs_trajectory_ope.py`:

```python
def test_pretrained_linucb_diverges_from_fresh_linucb(tmp_path: Path) -> None:
    from rl_recsys.evaluation.behavior_policy import (
        fit_behavior_policy_with_calibration,
    )
    from rl_recsys.data.loaders.rl4rs_trajectory_ope import (
        RL4RSTrajectoryOPESource,
    )
    from rl_recsys.evaluation import evaluate_trajectory_ope_agent
    from rl_recsys.agents import LinUCBAgent
    from rl_recsys.training import (
        pretrain_agent_on_logged, split_session_ids,
    )

    rng = np.random.default_rng(0)
    rows = []
    for sid in range(60):
        for seq in range(2):
            rows.append({
                "session_id": sid,
                "sequence_id": seq,
                "user_state": rng.standard_normal(2).tolist(),
                "slate": [int(rng.integers(0, 3)), int(rng.integers(0, 3))],
                "user_feedback": [int(rng.integers(0, 2)), int(rng.integers(0, 2))],
                "item_features": [[0.0, 0.0], [1.0, 0.0]],
            })
    parquet = tmp_path / "sessions_b.parquet"
    pd.DataFrame(rows).to_parquet(parquet, index=False)

    model = fit_behavior_policy_with_calibration(
        parquet, user_dim=2, item_dim=2, slate_size=2, num_items=3,
        epochs=5, batch_size=16, seed=0, nll_threshold=10.0,
    )

    train_ids, eval_ids = split_session_ids(parquet, train_fraction=0.5, seed=42)

    train_source = RL4RSTrajectoryOPESource(
        parquet_path=parquet, behavior_policy=model, slate_size=2,
        session_filter=train_ids,
    )
    eval_source_fresh = RL4RSTrajectoryOPESource(
        parquet_path=parquet, behavior_policy=model, slate_size=2,
        session_filter=eval_ids,
    )
    eval_source_pretrained = RL4RSTrajectoryOPESource(
        parquet_path=parquet, behavior_policy=model, slate_size=2,
        session_filter=eval_ids,
    )

    fresh = LinUCBAgent(slate_size=2, user_dim=2, item_dim=2, alpha=1.0)
    pretrained = LinUCBAgent(slate_size=2, user_dim=2, item_dim=2, alpha=1.0)
    pretrain_agent_on_logged(pretrained, train_source)

    fresh_result = evaluate_trajectory_ope_agent(
        eval_source_fresh, fresh, agent_name="fresh",
        max_trajectories=60, seed=0, gamma=0.95, temperature=1.0,
    )
    pretrained_result = evaluate_trajectory_ope_agent(
        eval_source_pretrained, pretrained, agent_name="pretrained",
        max_trajectories=60, seed=0, gamma=0.95, temperature=1.0,
    )

    assert np.isfinite(fresh_result.avg_seq_dr_value)
    assert np.isfinite(pretrained_result.avg_seq_dr_value)
    # The two values must differ — pretraining changed agent.score_items.
    assert (
        abs(fresh_result.avg_seq_dr_value - pretrained_result.avg_seq_dr_value)
        > 1e-6
    )
```

- [ ] **Step 2: Run the test**

```
.venv/bin/python -m pytest tests/test_rl4rs_trajectory_ope.py::test_pretrained_linucb_diverges_from_fresh_linucb -v
```

Expected: PASS. (No code changes needed — all infra is already in from prior tasks.) If it fails because the values are identical, the pretrain pass is a no-op — investigate before proceeding.

- [ ] **Step 3: Run the full suite to confirm no regressions**

```
.venv/bin/python -m pytest -q
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_rl4rs_trajectory_ope.py
git commit -m "test: end-to-end pretrain changes Sequential DR estimate

Pretrains LinUCB on the train half and confirms its avg_seq_dr_value
differs from a fresh LinUCB on the same eval half. Guards the
pretrain→score_items→target_probability path against silent no-ops."
```

---

## Task 6: Wire the benchmark script

**Files:**
- Modify: `scripts/benchmark_rl4rs_b_seq_dr.py`

Replace the single-source flow with a train/eval split. LinUCB's `make_agent` constructs and pretrains; Random is unchanged.

- [ ] **Step 1: Modify the benchmark script**

Edit `scripts/benchmark_rl4rs_b_seq_dr.py`. After the dim-detection block (around line 56), replace the rest of `main()` with:

```python
    model = fit_behavior_policy_with_calibration(
        parquet, user_dim=USER_DIM, item_dim=ITEM_DIM,
        slate_size=SLATE_SIZE, num_items=num_items,
        epochs=5, batch_size=512, seed=0,
    )

    from rl_recsys.training import (
        pretrain_agent_on_logged, split_session_ids,
    )

    train_ids, eval_ids = split_session_ids(
        parquet, train_fraction=0.5, seed=42,
    )
    print(
        f"split: train={len(train_ids)} sessions, eval={len(eval_ids)} sessions",
        flush=True,
    )

    def make_train_source() -> RL4RSTrajectoryOPESource:
        return RL4RSTrajectoryOPESource(
            parquet_path=parquet, behavior_policy=model,
            slate_size=SLATE_SIZE, session_filter=train_ids,
        )

    def make_eval_source() -> RL4RSTrajectoryOPESource:
        return RL4RSTrajectoryOPESource(
            parquet_path=parquet, behavior_policy=model,
            slate_size=SLATE_SIZE, session_filter=eval_ids,
        )

    def make_linucb() -> LinUCBAgent:
        agent = LinUCBAgent(
            slate_size=SLATE_SIZE, user_dim=USER_DIM,
            item_dim=ITEM_DIM, alpha=1.0,
        )
        metrics = pretrain_agent_on_logged(agent, make_train_source())
        print(
            f"  pretrain: {metrics['trajectories']:.0f} traj, "
            f"{metrics['total_steps']:.0f} steps, "
            f"mean_click_rate={metrics['mean_click_rate']:.3f}, "
            f"{metrics['seconds']:.1f}s",
            flush=True,
        )
        return agent

    print("\n--- LinUCB (with offline pretrain) ---", flush=True)
    linucb_result = evaluate_trajectory_ope_with_variance(
        make_source=make_eval_source,
        make_agent=make_linucb,
        agent_name="linucb",
        max_trajectories=5000, n_seeds=3, base_seed=42,
        gamma=0.95, temperature=1.0,
    )
    print(linucb_result, flush=True)

    print("\n--- Random ---", flush=True)
    random_result = evaluate_trajectory_ope_with_variance(
        make_source=make_eval_source,
        make_agent=lambda: RandomAgent(slate_size=SLATE_SIZE, seed=0),
        agent_name="random",
        max_trajectories=5000, n_seeds=3, base_seed=42,
        gamma=0.95, temperature=1.0,
    )
    print(random_result, flush=True)
```

(Remove the old `make_source` definition; it's superseded by the train/eval pair.)

- [ ] **Step 2: Sanity-check the script imports**

```
.venv/bin/python -c "import scripts.benchmark_rl4rs_b_seq_dr"
```

Expected: no import error. (If the scripts dir isn't a package, run `.venv/bin/python -m py_compile scripts/benchmark_rl4rs_b_seq_dr.py` instead.)

- [ ] **Step 3: Run the full test suite once more**

```
.venv/bin/python -m pytest -q
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/benchmark_rl4rs_b_seq_dr.py
git commit -m "feat(benchmark): pretrain LinUCB on train half before eval

Splits sessions 50/50 (deterministic, seed=42), pretrains a fresh
LinUCB per seed on the train source via pretrain_agent_on_logged,
then evaluates Sequential DR on the disjoint eval half. Random is
unchanged."
```

---

## Task 7: Run the benchmark on real data and capture results

**Files:**
- (no source changes; results recorded in TODO.md or a new docs file)

This is a manual / observational task. The previous run showed agents collapsing to the behavior policy; we need to see whether pretraining moves the needle.

- [ ] **Step 1: Confirm processed data is present**

```
ls -la data/processed/rl4rs/sessions_b.parquet
```

Expected: file exists (~171MB). If missing, run the pipeline first:

```
.venv/bin/python -m rl_recsys.data.cli rl4rs --download
```

(or whatever invocation the dataset CLI exposes — see `rl_recsys/data/cli.py`.)

- [ ] **Step 2: Run the benchmark, tee output to a log**

```
.venv/bin/python scripts/benchmark_rl4rs_b_seq_dr.py 2>&1 | tee /tmp/seq_dr_pretrain_run.log
```

Expected sections in output: dim detection, behavior-policy fit + calibration, split sizes, "--- LinUCB (with offline pretrain) ---" with per-seed pretrain metrics, then VarianceEvaluation, then "--- Random ---" then VarianceEvaluation.

- [ ] **Step 3: Capture the result table in TODO.md**

Update `TODO.md`. Replace the existing benchmark table (the one showing 10.565 / 10.433) with a new comparison showing pre- vs post-pretrain for LinUCB, plus the persistent Random number. If the new LinUCB value diverges meaningfully from the logged baseline, mark cause #2 as resolved and pivot the "Next up" recommendation to attack #1 (behavior model) or #3 (temperature sweep). If LinUCB still collapses, add a note describing what was observed and propose the next investigation step.

- [ ] **Step 4: Commit the doc update**

```bash
git add TODO.md
git commit -m "docs: capture LinUCB-with-pretrain Sequential DR run

[summary of result — was the run discriminative? what's next?]"
```

---

## Self-review

**Spec coverage:**
- Component 1 (`logged_clicks`) → Task 1.
- Component 2 (loader `session_filter`) → Task 2.
- Component 3 (`pretrain_agent_on_logged`) → Task 4.
- Component 4 (`split_session_ids`) → Task 3.
- Component 5 (benchmark wiring) → Task 6.
- Error handling: zero-trajectory raise (Task 4), empty-filter raise (Task 2), invalid fraction / empty split (Task 3).
- Tests: all named test functions in spec map to a Task 1/2/3/4/5 test.
- End-to-end smoke: Task 5.
- Acceptance run: Task 7.

No gaps.

**Placeholder scan:** All steps include concrete code or commands. Task 7 has one bracketed placeholder in the commit message body — that's intentional, the engineer fills it from observed run output.

**Type consistency:** `logged_clicks` is `np.ndarray` of int64, shape `(slate_size,)`, populated identically in the loader (Task 1) and consumed in pretrain (Task 4). `session_filter` is `set[int] | None` everywhere. `split_session_ids` returns `tuple[set[int], set[int]]`, consumed as such in Task 6.
