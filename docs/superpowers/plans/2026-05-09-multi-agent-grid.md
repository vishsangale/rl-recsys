# Multi-Agent Ablation Grid on RL4RS-B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land 12 new agents under a stable, dataset-agnostic interface plus a (agent × seed × pretrained) ablation harness on RL4RS-B that aggregates results into a checked-in `summary.md`.

**Architecture:** Extend `RecObs` with optional `history` + replay-only `logged_action`/`logged_clicks`; extend `Agent` with a `train_offline` default; refactor `factory.py` to a registry so `--agents all` works; add the 12 agents in dependency order (heuristics → linear bandit family on a shared mixin → neural → imitation → DL); add a runner that fans out the grid and a per-run JSON artifact format; add an aggregator that writes `summary.csv` + `summary.md`.

**Tech Stack:** Python 3.12, PyTorch (CUDA when available), pandas/pyarrow, LightGBM (new dep), pytest. Always use `.venv/bin/python` and `.venv/bin/pytest`.

**Spec:** `docs/superpowers/specs/2026-05-09-multi-agent-grid-design.md`.

---

## File map

**New files:**
- `rl_recsys/agents/_linear_base.py` — `LinearBanditBase` mixin with shared feature pipeline + `update`.
- `rl_recsys/agents/most_popular.py`, `logged_replay.py`, `oracle_click.py`
- `rl_recsys/agents/lin_ts.py`, `eps_greedy_linear.py`, `boltzmann_linear.py`
- `rl_recsys/agents/neural_linear.py`
- `rl_recsys/agents/bc.py`, `gbdt.py`
- `rl_recsys/agents/sasrec.py`, `topk_reinforce.py`, `decision_transformer.py`
- `rl_recsys/training/agent_grid_runner.py` — internal runner (callable from CLI + tests).
- `rl_recsys/training/results_aggregator.py` — long-form DataFrame + summary table.
- `scripts/benchmark_agent_grid.py` — CLI front-end.
- `configs/agents/*.yaml` — one per agent (14 files).
- `tests/test_agents_heuristic.py`, `test_agents_linear_bandit.py`, `test_agents_neural.py`, `test_agents_imitation.py`, `test_agents_dl.py`
- `tests/test_agent_grid_runner.py`, `test_results_aggregator.py`, `test_agent_grid_smoke.py`
- `tests/test_recobs.py`, `tests/test_agents_dataset_agnostic.py`

**Modified files:**
- `rl_recsys/environments/base.py` — add `HistoryStep`, extend `RecObs`.
- `rl_recsys/agents/base.py` — add `train_offline` default + `score_items` default.
- `rl_recsys/agents/factory.py` — refactor to `AGENT_REGISTRY` dict.
- `rl_recsys/data/loaders/rl4rs_trajectory_ope.py` — populate `history`/`logged_action`/`logged_clicks` on `RecObs`.
- `rl_recsys/agents/__init__.py` — re-export new agents.
- `rl_recsys/config.py` — add optional fields used by new agents.
- `requirements.txt` — add `lightgbm`.
- `tests/test_rl4rs_trajectory_ope.py` — add three loader-history tests.

---

## Task 1: Extend `RecObs` with `HistoryStep`, `history`, `logged_action`, `logged_clicks`

**Files:**
- Modify: `rl_recsys/environments/base.py`
- Test: `tests/test_recobs.py` (new)

- [ ] **Step 1: Write failing tests for `HistoryStep` and extended `RecObs`**

```python
# tests/test_recobs.py
from __future__ import annotations

import numpy as np
import pytest

from rl_recsys.environments.base import HistoryStep, RecObs


def test_history_step_is_a_dataclass_with_slate_and_clicks():
    step = HistoryStep(
        slate=np.array([1, 2, 3], dtype=np.int64),
        clicks=np.array([0, 1, 0], dtype=np.int64),
    )
    assert step.slate.tolist() == [1, 2, 3]
    assert step.clicks.tolist() == [0, 1, 0]


def test_recobs_legacy_constructor_still_works():
    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.zeros((10, 3)),
        candidate_ids=np.arange(10, dtype=np.int64),
    )
    assert obs.history == ()
    assert obs.logged_action is None
    assert obs.logged_clicks is None


def test_recobs_accepts_history_and_logged_fields():
    h = (HistoryStep(np.array([0]), np.array([1])),)
    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.zeros((10, 3)),
        candidate_ids=np.arange(10, dtype=np.int64),
        history=h,
        logged_action=np.array([1, 2, 3]),
        logged_clicks=np.array([0, 1, 0]),
    )
    assert len(obs.history) == 1
    assert obs.logged_action.tolist() == [1, 2, 3]
    assert obs.logged_clicks.tolist() == [0, 1, 0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_recobs.py -v`
Expected: FAIL with `ImportError: cannot import name 'HistoryStep'`.

- [ ] **Step 3: Implement the dataclass changes**

```python
# rl_recsys/environments/base.py — replace top of file
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class HistoryStep:
    """One past step within the current session.

    Used by sequence-aware agents (SASRec, TopK REINFORCE, Decision
    Transformer). Populated by trajectory loaders; outside replay
    contexts the consumer constructs an empty tuple.
    """

    slate: np.ndarray   # (slate_size,) candidate indices into RecObs.candidate_*
    clicks: np.ndarray  # (slate_size,) 0/1


@dataclass
class RecObs:
    """Observation returned by a recommendation environment."""

    user_features: np.ndarray  # (user_dim,)
    candidate_features: np.ndarray  # (num_candidates, item_dim)
    candidate_ids: np.ndarray  # (num_candidates,)
    history: tuple[HistoryStep, ...] = field(default_factory=tuple)
    # Replay-mode-only fields. None outside replay sources.
    logged_action: np.ndarray | None = None
    logged_clicks: np.ndarray | None = None
```

(The `RecStep` / `RecEnv` definitions below stay unchanged.)

- [ ] **Step 4: Run the new tests, then run the full existing suite**

Run: `.venv/bin/pytest tests/test_recobs.py -v`
Expected: PASS.

Run: `.venv/bin/pytest -x -q`
Expected: All 197 existing tests still PASS (plus the 3 new ones).

- [ ] **Step 5: Commit**

```bash
git add rl_recsys/environments/base.py tests/test_recobs.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat: extend RecObs with HistoryStep, optional history and replay fields

Adds optional sequence history and replay-only logged_action/logged_clicks
fields to RecObs so DL/RL agents can consume sequential context and
LoggedReplay/Oracle agents can score against the logged slate. Defaults
preserve the legacy constructor.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Populate `history`/`logged_action`/`logged_clicks` from `RL4RSTrajectoryOPESource`

**Files:**
- Modify: `rl_recsys/data/loaders/rl4rs_trajectory_ope.py:113-163`
- Test: `tests/test_rl4rs_trajectory_ope.py` (extend)

- [ ] **Step 1: Write failing tests for loader population**

Append to `tests/test_rl4rs_trajectory_ope.py`:

```python
def test_loader_history_accumulates_within_session(tmp_path):
    # Use the existing fixture builder in this test file (search for one
    # that writes a small parquet — copy its setup verbatim).
    source = _build_small_source(tmp_path, num_sessions=1, steps_per_session=3)
    [traj] = list(source.iter_trajectories(max_trajectories=1, seed=0))
    assert len(traj[0].obs.history) == 0
    assert len(traj[1].obs.history) == 1
    assert len(traj[2].obs.history) == 2


def test_loader_history_resets_between_sessions(tmp_path):
    source = _build_small_source(tmp_path, num_sessions=2, steps_per_session=2)
    trajs = list(source.iter_trajectories(max_trajectories=2, seed=0))
    # Both sessions start with empty history at step 0.
    assert len(trajs[0][0].obs.history) == 0
    assert len(trajs[1][0].obs.history) == 0


def test_loader_obs_logged_action_matches_step_logged_action(tmp_path):
    source = _build_small_source(tmp_path, num_sessions=1, steps_per_session=2)
    [traj] = list(source.iter_trajectories(max_trajectories=1, seed=0))
    for step in traj:
        assert step.obs.logged_action is not None
        assert step.obs.logged_clicks is not None
        np.testing.assert_array_equal(step.obs.logged_action, step.logged_action)
        np.testing.assert_array_equal(step.obs.logged_clicks, step.logged_clicks)
```

If `_build_small_source` doesn't already exist, hoist the existing per-test fixture setup into a helper. Read the file first; reuse what's there.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_rl4rs_trajectory_ope.py -v -k "history or logged_action_matches"`
Expected: FAIL — `obs.history` doesn't exist on objects yielded by the current loader.

- [ ] **Step 3: Update `iter_trajectories` to populate the new fields**

In `rl_recsys/data/loaders/rl4rs_trajectory_ope.py`, replace the inner per-row loop (lines 132-161) with:

```python
            history: list[HistoryStep] = []
            steps: list[LoggedTrajectoryStep] = []
            for row_pos, row in zip(group.index, group.itertuples(index=False)):
                user_features = np.array(list(row.user_state), dtype=np.float64)
                logged_slate_ids = np.array(list(row.slate), dtype=np.int64)
                logged_reward = float(np.sum(row.user_feedback))
                logged_clicks = np.array(
                    list(row.user_feedback), dtype=np.int64,
                )

                slate_indices = np.array(
                    [self._cand_id_to_idx[int(x)] for x in logged_slate_ids],
                    dtype=np.int64,
                )

                propensity = float(self._propensities[row_pos])
                obs = RecObs(
                    user_features=user_features,
                    candidate_features=self._candidate_features,
                    candidate_ids=self._candidate_ids,
                    history=tuple(history),
                    logged_action=slate_indices,
                    logged_clicks=logged_clicks,
                )
                steps.append(
                    LoggedTrajectoryStep(
                        obs=obs,
                        logged_action=slate_indices,
                        logged_reward=logged_reward,
                        logged_clicks=logged_clicks,
                        propensity=propensity,
                    )
                )
                history.append(
                    HistoryStep(slate=slate_indices, clicks=logged_clicks)
                )
            yield steps
            emitted += 1
```

Add `from rl_recsys.environments.base import HistoryStep, RecObs` at the top (RecObs is already imported; just add HistoryStep).

- [ ] **Step 4: Run the new + all existing loader tests**

Run: `.venv/bin/pytest tests/test_rl4rs_trajectory_ope.py -v`
Expected: All PASS, including the three new tests.

Run: `.venv/bin/pytest -x -q`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add rl_recsys/data/loaders/rl4rs_trajectory_ope.py tests/test_rl4rs_trajectory_ope.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat: populate RecObs history and replay fields in RL4RS loader

iter_trajectories now attaches per-step session history and the upcoming
logged_action/logged_clicks to obs, enabling sequence-aware agents and
the LoggedReplay/Oracle baselines.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Extend `Agent` ABC with `train_offline` and `score_items` defaults

**Files:**
- Modify: `rl_recsys/agents/base.py`
- Test: `tests/test_agents.py` (extend)

- [ ] **Step 1: Write failing tests for the new defaults**

Append to `tests/test_agents.py`:

```python
def test_agent_score_items_default_returns_zeros():
    from rl_recsys.agents.random import RandomAgent
    from rl_recsys.environments.base import RecObs

    agent = RandomAgent(slate_size=3)
    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.zeros((10, 3)),
        candidate_ids=np.arange(10, dtype=np.int64),
    )
    scores = agent.score_items(obs)
    assert scores.shape == (10,)
    assert np.allclose(scores, 0.0)


def test_agent_train_offline_default_calls_pretrain_helper():
    # The default delegates to pretrain_agent_on_logged, which calls
    # agent.update for every step. We assert by counting update calls
    # against a fake source that yields one trajectory of two steps.
    from rl_recsys.agents.linucb import LinUCBAgent
    from rl_recsys.environments.base import RecObs
    from rl_recsys.evaluation.ope_trajectory import LoggedTrajectoryStep

    class _FakeSource:
        def iter_trajectories(self, *, max_trajectories=None, seed=0):
            obs = RecObs(
                user_features=np.zeros(4),
                candidate_features=np.zeros((10, 3)),
                candidate_ids=np.arange(10, dtype=np.int64),
            )
            step = LoggedTrajectoryStep(
                obs=obs,
                logged_action=np.array([0, 1, 2], dtype=np.int64),
                logged_reward=1.0,
                logged_clicks=np.array([1, 0, 0], dtype=np.int64),
                propensity=0.1,
            )
            yield [step, step]

    agent = LinUCBAgent(slate_size=3, user_dim=4, item_dim=3)
    metrics = agent.train_offline(_FakeSource(), seed=0)
    assert metrics["total_steps"] == 2.0
    assert metrics["trajectories"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_agents.py -v -k "score_items_default or train_offline_default"`
Expected: FAIL — the methods don't exist on the ABC.

- [ ] **Step 3: Add defaults to `Agent` ABC**

Replace `rl_recsys/agents/base.py` with:

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from rl_recsys.environments.base import RecObs

if TYPE_CHECKING:
    from rl_recsys.evaluation.ope_trajectory import LoggedTrajectorySource


class Agent(ABC):
    """Abstract base class for recommendation agents."""

    @abstractmethod
    def select_slate(self, obs: RecObs) -> np.ndarray:
        """Return an array of candidate indices forming the slate."""
        ...

    @abstractmethod
    def update(
        self,
        obs: RecObs,
        slate: np.ndarray,
        reward: float,
        clicks: np.ndarray,
        next_obs: RecObs,
    ) -> dict[str, float]:
        """Update the agent and return a dict of logged metrics."""
        ...

    def score_items(self, obs: RecObs) -> np.ndarray:
        """Per-candidate scores. Default: zeros (uniform softmax under the
        Boltzmann shim). Override for any agent that wants to influence the
        target-policy probability used by Sequential DR."""
        return np.zeros(len(obs.candidate_features), dtype=np.float64)

    def train_offline(
        self,
        source: "LoggedTrajectorySource",
        *,
        seed: int = 0,
    ) -> dict[str, float]:
        """Train on a logged trajectory source. Default: per-step update via
        pretrain_agent_on_logged. Heuristic agents override with no-op;
        DL/batch agents override with their own training loop."""
        from rl_recsys.training.offline_pretrain import pretrain_agent_on_logged
        return pretrain_agent_on_logged(self, source, seed=seed)
```

`RandomAgent.score_items` returns `np.zeros(...)` already; that's now redundant with the default but harmless. Leave it. Same for the existing `LinUCBAgent.score_items` override (which adds the UCB bonus) — that override is intentional, keep it.

- [ ] **Step 4: Run new + full suite**

Run: `.venv/bin/pytest tests/test_agents.py -v -k "score_items_default or train_offline_default"`
Expected: PASS.

Run: `.venv/bin/pytest -x -q`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add rl_recsys/agents/base.py tests/test_agents.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat: add train_offline and score_items defaults to Agent ABC

Default train_offline delegates to pretrain_agent_on_logged. Default
score_items returns zeros (uniform softmax). Concrete agents override
where they care.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Refactor `factory.py` to `AGENT_REGISTRY`

**Files:**
- Modify: `rl_recsys/agents/factory.py`
- Test: `tests/test_factory.py` (extend)

- [ ] **Step 1: Write failing tests for the registry**

Append to `tests/test_factory.py`:

```python
def test_agent_registry_contains_existing_agents():
    from rl_recsys.agents.factory import AGENT_REGISTRY

    assert "random" in AGENT_REGISTRY
    assert "linucb" in AGENT_REGISTRY


def test_build_agent_unknown_name_raises_valueerror():
    from rl_recsys.agents.factory import build_agent
    from rl_recsys.config import AgentConfig, EnvConfig

    with pytest.raises(ValueError, match="Unknown agent"):
        build_agent(
            AgentConfig(name="not-real"),
            EnvConfig(slate_size=3, user_dim=4, item_dim=3, num_candidates=10),
        )
```

(If `pytest` isn't already imported at the top of the file, add `import pytest`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_factory.py -v -k "agent_registry or unknown_name"`
Expected: FAIL — `AGENT_REGISTRY` doesn't exist; `build_agent` raises a different message.

- [ ] **Step 3: Replace `factory.py` with the registry**

```python
# rl_recsys/agents/factory.py
from __future__ import annotations

from typing import Callable

from rl_recsys.agents.base import Agent
from rl_recsys.agents.linucb import LinUCBAgent
from rl_recsys.agents.random import RandomAgent
from rl_recsys.config import AgentConfig, EnvConfig

AgentBuilder = Callable[[AgentConfig, EnvConfig], Agent]


def _build_random(agent_cfg: AgentConfig, env_cfg: EnvConfig) -> Agent:
    return RandomAgent(slate_size=env_cfg.slate_size)


def _build_linucb(agent_cfg: AgentConfig, env_cfg: EnvConfig) -> Agent:
    return LinUCBAgent(
        slate_size=env_cfg.slate_size,
        user_dim=env_cfg.user_dim,
        item_dim=env_cfg.item_dim,
        alpha=agent_cfg.alpha,
    )


AGENT_REGISTRY: dict[str, AgentBuilder] = {
    "random": _build_random,
    "linucb": _build_linucb,
}


def build_agent(agent_cfg: AgentConfig, env_cfg: EnvConfig) -> Agent:
    """Build an agent from structured experiment config via AGENT_REGISTRY."""
    name = agent_cfg.name.lower()
    try:
        builder = AGENT_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown agent: {agent_cfg.name}") from exc
    return builder(agent_cfg, env_cfg)
```

- [ ] **Step 4: Run new + full suite**

Run: `.venv/bin/pytest tests/test_factory.py -v`
Expected: All PASS.

Run: `.venv/bin/pytest -x -q`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add rl_recsys/agents/factory.py tests/test_factory.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
refactor: factory uses AGENT_REGISTRY for dispatch

Replaces the if/elif chain with a name → builder dict so subsequent
agents register a single entry. Keeps build_agent's external contract
identical.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Dataset-agnostic lint test

**Files:**
- Test: `tests/test_agents_dataset_agnostic.py` (new)

- [ ] **Step 1: Write the lint test**

```python
# tests/test_agents_dataset_agnostic.py
from __future__ import annotations

import ast
import pkgutil
from pathlib import Path

import rl_recsys.agents

FORBIDDEN_PREFIXES = (
    "rl_recsys.data.loaders.",
    "rl_recsys.environments.rl4rs",
    "rl_recsys.environments.kuairec",
    "rl_recsys.environments.finn_no_slate",
    "rl_recsys.environments.movielens",
    "rl_recsys.environments.open_bandit",
)


def _imported_modules(py_path: Path) -> set[str]:
    tree = ast.parse(py_path.read_text())
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                seen.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                seen.add(node.module)
    return seen


def test_no_agent_module_imports_dataset_specific_code():
    pkg_path = Path(rl_recsys.agents.__file__).parent
    offenders: list[tuple[str, str]] = []
    for entry in pkgutil.iter_modules([str(pkg_path)]):
        py = pkg_path / f"{entry.name}.py"
        if not py.exists():
            continue
        for mod in _imported_modules(py):
            if mod.startswith(FORBIDDEN_PREFIXES):
                offenders.append((entry.name, mod))
    assert not offenders, (
        f"Agent modules must stay dataset-agnostic. Offenders: {offenders}"
    )
```

- [ ] **Step 2: Run the test**

Run: `.venv/bin/pytest tests/test_agents_dataset_agnostic.py -v`
Expected: PASS — current agents (random, linucb) don't import dataset code.

- [ ] **Step 3: Commit**

```bash
git add tests/test_agents_dataset_agnostic.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
test: add dataset-agnostic lint over agent modules

AST-walks every module under rl_recsys.agents and asserts none import
loaders or dataset-specific environments. Guard rail for the multi-
agent grid: agents must stay portable across datasets.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `MostPopularAgent`

**Files:**
- Create: `rl_recsys/agents/most_popular.py`
- Modify: `rl_recsys/agents/factory.py`, `rl_recsys/agents/__init__.py`
- Test: `tests/test_agents_heuristic.py` (new)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agents_heuristic.py
from __future__ import annotations

import numpy as np
import pytest

from rl_recsys.agents.most_popular import MostPopularAgent
from rl_recsys.environments.base import RecObs
from rl_recsys.evaluation.ope_trajectory import LoggedTrajectoryStep


def _make_obs(num_candidates: int = 5):
    return RecObs(
        user_features=np.zeros(4),
        candidate_features=np.zeros((num_candidates, 3)),
        candidate_ids=np.arange(num_candidates, dtype=np.int64),
    )


def _make_step(slate: list[int], clicks: list[int]):
    obs = _make_obs(num_candidates=5)
    return LoggedTrajectoryStep(
        obs=obs,
        logged_action=np.array(slate, dtype=np.int64),
        logged_reward=float(sum(clicks)),
        logged_clicks=np.array(clicks, dtype=np.int64),
        propensity=0.1,
    )


class _StubSource:
    def __init__(self, trajs):
        self._trajs = trajs

    def iter_trajectories(self, *, max_trajectories=None, seed=0):
        yield from self._trajs


def test_most_popular_train_offline_counts_clicks_per_item():
    source = _StubSource([
        [_make_step([0, 1, 2], [1, 0, 1])],
        [_make_step([1, 2, 3], [0, 1, 0])],
    ])
    agent = MostPopularAgent(slate_size=2, num_candidates=5)
    agent.train_offline(source)
    np.testing.assert_array_equal(
        agent._clicks_per_item, np.array([1, 0, 2, 0, 0], dtype=np.float64),
    )


def test_most_popular_select_slate_picks_top_k():
    agent = MostPopularAgent(slate_size=2, num_candidates=5)
    agent._clicks_per_item = np.array([1, 5, 0, 3, 2], dtype=np.float64)
    slate = agent.select_slate(_make_obs())
    assert set(slate.tolist()) == {1, 3}


def test_most_popular_select_slate_raises_when_slate_too_large():
    agent = MostPopularAgent(slate_size=10, num_candidates=5)
    with pytest.raises(ValueError, match="exceeds num_candidates"):
        agent.select_slate(_make_obs(num_candidates=5))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_agents_heuristic.py -v`
Expected: FAIL — `most_popular` module doesn't exist.

- [ ] **Step 3: Implement `MostPopularAgent`**

```python
# rl_recsys/agents/most_popular.py
from __future__ import annotations

import numpy as np

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecObs


class MostPopularAgent(Agent):
    """Ranks candidates by global click counts seen during training."""

    def __init__(self, slate_size: int, num_candidates: int) -> None:
        self._slate_size = int(slate_size)
        self._num_candidates = int(num_candidates)
        self._clicks_per_item = np.zeros(num_candidates, dtype=np.float64)

    def train_offline(self, source, *, seed: int = 0) -> dict[str, float]:
        for traj in source.iter_trajectories(seed=seed):
            for step in traj:
                self._clicks_per_item[step.logged_action] += step.logged_clicks
        return {"items_seen": float(self._clicks_per_item.sum())}

    def select_slate(self, obs: RecObs) -> np.ndarray:
        n = len(obs.candidate_features)
        if self._slate_size > n:
            raise ValueError(
                f"slate_size={self._slate_size} exceeds num_candidates={n}"
            )
        return np.argsort(self._clicks_per_item)[-self._slate_size:][::-1]

    def score_items(self, obs: RecObs) -> np.ndarray:
        return self._clicks_per_item.astype(np.float64)

    def update(self, obs, slate, reward, clicks, next_obs) -> dict[str, float]:
        return {}
```

- [ ] **Step 4: Register in factory + `__init__.py`**

In `rl_recsys/agents/factory.py`, add an import + builder + registry entry:

```python
from rl_recsys.agents.most_popular import MostPopularAgent

def _build_most_popular(agent_cfg: AgentConfig, env_cfg: EnvConfig) -> Agent:
    return MostPopularAgent(
        slate_size=env_cfg.slate_size,
        num_candidates=env_cfg.num_candidates,
    )

AGENT_REGISTRY["most_popular"] = _build_most_popular
```

In `rl_recsys/agents/__init__.py`, add `from rl_recsys.agents.most_popular import MostPopularAgent` and add the name to `__all__` if present.

- [ ] **Step 5: Run all tests**

Run: `.venv/bin/pytest tests/test_agents_heuristic.py tests/test_factory.py tests/test_agents_dataset_agnostic.py -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add rl_recsys/agents/most_popular.py rl_recsys/agents/factory.py rl_recsys/agents/__init__.py tests/test_agents_heuristic.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat: add MostPopularAgent baseline

Counts per-item clicks from the train source and ranks candidates by
that count. Registered in AGENT_REGISTRY.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `LoggedReplayAgent`

**Files:**
- Create: `rl_recsys/agents/logged_replay.py`
- Modify: `rl_recsys/agents/factory.py`, `rl_recsys/agents/__init__.py`
- Test: `tests/test_agents_heuristic.py` (extend)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_agents_heuristic.py`:

```python
def test_logged_replay_returns_logged_slate():
    from rl_recsys.agents.logged_replay import LoggedReplayAgent

    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.zeros((10, 3)),
        candidate_ids=np.arange(10, dtype=np.int64),
        logged_action=np.array([4, 7, 1], dtype=np.int64),
    )
    agent = LoggedReplayAgent(slate_size=3)
    np.testing.assert_array_equal(agent.select_slate(obs), [4, 7, 1])


def test_logged_replay_raises_without_logged_action():
    from rl_recsys.agents.logged_replay import LoggedReplayAgent

    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.zeros((10, 3)),
        candidate_ids=np.arange(10, dtype=np.int64),
    )
    agent = LoggedReplayAgent(slate_size=3)
    with pytest.raises(ValueError, match="replay-mode source"):
        agent.select_slate(obs)


def test_logged_replay_score_items_peaks_on_logged_slate():
    from rl_recsys.agents.logged_replay import LoggedReplayAgent

    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.zeros((6, 3)),
        candidate_ids=np.arange(6, dtype=np.int64),
        logged_action=np.array([2, 4], dtype=np.int64),
    )
    agent = LoggedReplayAgent(slate_size=2)
    scores = agent.score_items(obs)
    assert scores.shape == (6,)
    assert scores[2] > scores[0] and scores[4] > scores[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_agents_heuristic.py -v -k logged_replay`
Expected: FAIL.

- [ ] **Step 3: Implement `LoggedReplayAgent`**

```python
# rl_recsys/agents/logged_replay.py
from __future__ import annotations

import numpy as np

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecObs


class LoggedReplayAgent(Agent):
    """Replays the logged slate. Used as an OPE sanity check — the resulting
    DR value should match avg_logged_discounted_return up to IS noise."""

    def __init__(self, slate_size: int) -> None:
        self._slate_size = int(slate_size)

    def _require_logged(self, obs: RecObs) -> np.ndarray:
        if obs.logged_action is None:
            raise ValueError(
                "LoggedReplayAgent requires a replay-mode source — "
                "obs.logged_action is None"
            )
        return obs.logged_action

    def select_slate(self, obs: RecObs) -> np.ndarray:
        return self._require_logged(obs).astype(np.int64)

    def score_items(self, obs: RecObs) -> np.ndarray:
        scores = np.zeros(len(obs.candidate_features), dtype=np.float64)
        scores[self._require_logged(obs)] = 1.0
        return scores

    def update(self, obs, slate, reward, clicks, next_obs) -> dict[str, float]:
        return {}

    def train_offline(self, source, *, seed: int = 0) -> dict[str, float]:
        return {}
```

- [ ] **Step 4: Register in factory + `__init__.py`**

```python
# in factory.py
from rl_recsys.agents.logged_replay import LoggedReplayAgent

def _build_logged_replay(agent_cfg: AgentConfig, env_cfg: EnvConfig) -> Agent:
    return LoggedReplayAgent(slate_size=env_cfg.slate_size)

AGENT_REGISTRY["logged_replay"] = _build_logged_replay
```

Add re-export in `rl_recsys/agents/__init__.py`.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/test_agents_heuristic.py tests/test_agents_dataset_agnostic.py -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add rl_recsys/agents/logged_replay.py rl_recsys/agents/factory.py rl_recsys/agents/__init__.py tests/test_agents_heuristic.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat: add LoggedReplayAgent (OPE sanity check)

Replays obs.logged_action. score_items peaks on the logged slate so
the Boltzmann shim yields target probability ~ 1, making the DR value
match avg_logged_discounted_return within IS noise.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `OracleClickAgent`

**Files:**
- Create: `rl_recsys/agents/oracle_click.py`
- Modify: `rl_recsys/agents/factory.py`, `rl_recsys/agents/__init__.py`
- Test: `tests/test_agents_heuristic.py` (extend)

- [ ] **Step 1: Write failing tests**

```python
def test_oracle_click_picks_clicked_items_first():
    from rl_recsys.agents.oracle_click import OracleClickAgent

    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.zeros((10, 3)),
        candidate_ids=np.arange(10, dtype=np.int64),
        logged_action=np.array([2, 5, 7, 9], dtype=np.int64),
        logged_clicks=np.array([0, 1, 1, 0], dtype=np.int64),
    )
    agent = OracleClickAgent(slate_size=2)
    slate = agent.select_slate(obs)
    assert set(slate.tolist()) == {5, 7}


def test_oracle_click_pads_when_underclicked():
    from rl_recsys.agents.oracle_click import OracleClickAgent

    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.zeros((10, 3)),
        candidate_ids=np.arange(10, dtype=np.int64),
        logged_action=np.array([2, 5, 7], dtype=np.int64),
        logged_clicks=np.array([0, 1, 0], dtype=np.int64),
    )
    agent = OracleClickAgent(slate_size=3)
    slate = agent.select_slate(obs)
    assert slate.shape == (3,)
    assert 5 in slate.tolist()  # clicked item must be present
    assert len(set(slate.tolist())) == 3  # all unique


def test_oracle_click_raises_without_logged_clicks():
    from rl_recsys.agents.oracle_click import OracleClickAgent

    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.zeros((10, 3)),
        candidate_ids=np.arange(10, dtype=np.int64),
        logged_action=np.array([0, 1], dtype=np.int64),
    )
    agent = OracleClickAgent(slate_size=2)
    with pytest.raises(ValueError, match="logged_clicks"):
        agent.select_slate(obs)
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/pytest tests/test_agents_heuristic.py -v -k oracle_click`
Expected: FAIL.

- [ ] **Step 3: Implement `OracleClickAgent`**

```python
# rl_recsys/agents/oracle_click.py
from __future__ import annotations

import numpy as np

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecObs


class OracleClickAgent(Agent):
    """Cheats by reading the logged clicks to rank logged items first.
    Eval-only upper bound. Opt-in via the harness's --agents flag."""

    def __init__(self, slate_size: int) -> None:
        self._slate_size = int(slate_size)

    def select_slate(self, obs: RecObs) -> np.ndarray:
        if obs.logged_action is None or obs.logged_clicks is None:
            raise ValueError(
                "OracleClickAgent requires obs.logged_action and "
                "obs.logged_clicks (replay-mode source)"
            )
        # Rank logged items by their click value; take top-k.
        order = np.argsort(-obs.logged_clicks, kind="stable")
        ranked_logged = obs.logged_action[order]
        if len(ranked_logged) >= self._slate_size:
            return ranked_logged[: self._slate_size].astype(np.int64)
        # Pad with non-logged candidates in arbitrary order.
        used = set(int(x) for x in ranked_logged)
        n = len(obs.candidate_features)
        padding = [i for i in range(n) if i not in used][
            : self._slate_size - len(ranked_logged)
        ]
        return np.concatenate(
            [ranked_logged, np.asarray(padding, dtype=np.int64)]
        ).astype(np.int64)

    def score_items(self, obs: RecObs) -> np.ndarray:
        if obs.logged_action is None or obs.logged_clicks is None:
            raise ValueError(
                "OracleClickAgent.score_items requires logged_action and "
                "logged_clicks"
            )
        scores = np.zeros(len(obs.candidate_features), dtype=np.float64)
        scores[obs.logged_action] = obs.logged_clicks.astype(np.float64)
        return scores

    def update(self, obs, slate, reward, clicks, next_obs) -> dict[str, float]:
        return {}

    def train_offline(self, source, *, seed: int = 0) -> dict[str, float]:
        return {}
```

- [ ] **Step 4: Register, run tests, commit**

```python
# in factory.py
from rl_recsys.agents.oracle_click import OracleClickAgent

def _build_oracle_click(agent_cfg: AgentConfig, env_cfg: EnvConfig) -> Agent:
    return OracleClickAgent(slate_size=env_cfg.slate_size)

AGENT_REGISTRY["oracle_click"] = _build_oracle_click
```

Run: `.venv/bin/pytest tests/test_agents_heuristic.py tests/test_agents_dataset_agnostic.py -v`
Expected: PASS.

```bash
git add rl_recsys/agents/oracle_click.py rl_recsys/agents/factory.py rl_recsys/agents/__init__.py tests/test_agents_heuristic.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat: add OracleClickAgent (eval-only upper bound)

Reads obs.logged_clicks to put clicked items at the top of the slate.
Strict upper bound; opt-in via --agents flag in the harness.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `LinearBanditBase` mixin

**Files:**
- Create: `rl_recsys/agents/_linear_base.py`
- Test: `tests/test_agents_linear_bandit.py` (new)

- [ ] **Step 1: Write failing tests for the shared feature pipeline + update**

```python
# tests/test_agents_linear_bandit.py
from __future__ import annotations

import numpy as np
import pytest

from rl_recsys.agents._linear_base import LinearBanditBase
from rl_recsys.environments.base import RecObs


class _DummyLinear(LinearBanditBase):
    def select_slate(self, obs):
        return np.array([0, 1, 2], dtype=np.int64)

    def score_items(self, obs):
        features = self._candidate_features(obs)
        theta = np.linalg.solve(self._a_matrix, self._b_vector)
        return features @ theta


def test_linear_base_update_matches_linucb_accumulator():
    from rl_recsys.agents.linucb import LinUCBAgent

    obs = RecObs(
        user_features=np.array([1.0, 0.0, 0.0, 0.0]),
        candidate_features=np.eye(5, 3),
        candidate_ids=np.arange(5, dtype=np.int64),
    )
    slate = np.array([0, 1, 2], dtype=np.int64)
    clicks = np.array([1, 0, 1], dtype=np.float64)

    a = _DummyLinear(slate_size=3, user_dim=4, item_dim=3)
    b = LinUCBAgent(slate_size=3, user_dim=4, item_dim=3)
    a.update(obs, slate, reward=2.0, clicks=clicks, next_obs=obs)
    b.update(obs, slate, reward=2.0, clicks=clicks, next_obs=obs)
    np.testing.assert_allclose(a._a_matrix, b._a_matrix)
    np.testing.assert_allclose(a._b_vector, b._b_vector)


def test_linear_base_features_validate_shape():
    bad_obs = RecObs(
        user_features=np.zeros(99),  # wrong dim
        candidate_features=np.zeros((5, 3)),
        candidate_ids=np.arange(5, dtype=np.int64),
    )
    a = _DummyLinear(slate_size=3, user_dim=4, item_dim=3)
    with pytest.raises(ValueError, match="user_features shape"):
        a.score_items(bad_obs)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_agents_linear_bandit.py -v`
Expected: FAIL — `_linear_base` doesn't exist.

- [ ] **Step 3: Implement the mixin**

```python
# rl_recsys/agents/_linear_base.py
from __future__ import annotations

import numpy as np

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecObs


class LinearBanditBase(Agent):
    """Shared feature pipeline + sufficient-stats update for the linear
    bandit family. Subclasses override score_items (and select_slate when
    they need explicit exploration like ε-greedy)."""

    def __init__(
        self, slate_size: int, user_dim: int, item_dim: int,
    ) -> None:
        self._slate_size = int(slate_size)
        self._user_dim = int(user_dim)
        self._item_dim = int(item_dim)
        self._interaction_dim = min(user_dim, item_dim)
        self._feature_dim = user_dim + item_dim + self._interaction_dim
        self._a_matrix = np.eye(self._feature_dim, dtype=np.float64)
        self._b_vector = np.zeros(self._feature_dim, dtype=np.float64)

    def _candidate_features(self, obs: RecObs) -> np.ndarray:
        user = np.asarray(obs.user_features, dtype=np.float64)
        items = np.asarray(obs.candidate_features, dtype=np.float64)
        if user.shape != (self._user_dim,):
            raise ValueError(
                f"user_features shape {user.shape} does not match "
                f"({self._user_dim},)"
            )
        if items.ndim != 2 or items.shape[1] != self._item_dim:
            raise ValueError(
                "candidate_features shape "
                f"{items.shape} does not match (*, {self._item_dim})"
            )
        user_block = np.broadcast_to(user, (items.shape[0], self._user_dim))
        d = self._interaction_dim
        interaction = user_block[:, :d] * items[:, :d]
        return np.concatenate([user_block, items, interaction], axis=1)

    def select_slate(self, obs: RecObs) -> np.ndarray:
        n = len(obs.candidate_features)
        if self._slate_size > n:
            raise ValueError(
                f"slate_size={self._slate_size} exceeds num_candidates={n}"
            )
        scores = self.score_items(obs)
        return np.argsort(scores)[-self._slate_size:][::-1]

    def update(
        self, obs, slate, reward, clicks, next_obs,
    ) -> dict[str, float]:
        features = self._candidate_features(obs)[np.asarray(slate)]
        clicks_arr = np.asarray(clicks, dtype=np.float64)
        if len(clicks_arr) != len(features):
            raise ValueError(
                f"clicks length {len(clicks_arr)} does not match slate "
                f"length {len(features)}"
            )
        for x, click in zip(features, clicks_arr):
            self._a_matrix += np.outer(x, x)
            self._b_vector += float(click) * x
        return {
            "agent_updates": float(len(features)),
            "agent_click_mean": float(clicks_arr.mean()) if len(clicks_arr) else 0.0,
        }
```

- [ ] **Step 4: Run tests + commit**

Run: `.venv/bin/pytest tests/test_agents_linear_bandit.py tests/test_agents_dataset_agnostic.py -v`
Expected: PASS.

```bash
git add rl_recsys/agents/_linear_base.py tests/test_agents_linear_bandit.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat: LinearBanditBase mixin for shared feature pipeline

Holds (A, b), the user||item||interaction feature build, and the
shared update step. LinTS/EpsGreedy/Boltzmann agents subclass it and
override score_items.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: `LinTSAgent`

**Files:**
- Create: `rl_recsys/agents/lin_ts.py`
- Modify: `rl_recsys/agents/factory.py`, `rl_recsys/agents/__init__.py`
- Test: `tests/test_agents_linear_bandit.py` (extend)

- [ ] **Step 1: Write failing tests**

```python
def test_lints_sample_is_deterministic_with_rng():
    from rl_recsys.agents.lin_ts import LinTSAgent

    obs = RecObs(
        user_features=np.array([1.0, 0.0, 0.0, 0.0]),
        candidate_features=np.eye(5, 3),
        candidate_ids=np.arange(5, dtype=np.int64),
    )
    a = LinTSAgent(
        slate_size=2, user_dim=4, item_dim=3,
        sigma=1.0, rng=np.random.default_rng(0),
    )
    b = LinTSAgent(
        slate_size=2, user_dim=4, item_dim=3,
        sigma=1.0, rng=np.random.default_rng(0),
    )
    np.testing.assert_array_equal(a.select_slate(obs), b.select_slate(obs))


def test_lints_score_shape_matches_num_candidates():
    from rl_recsys.agents.lin_ts import LinTSAgent

    obs = RecObs(
        user_features=np.array([1.0, 0.0, 0.0, 0.0]),
        candidate_features=np.eye(7, 3),
        candidate_ids=np.arange(7, dtype=np.int64),
    )
    agent = LinTSAgent(slate_size=3, user_dim=4, item_dim=3)
    assert agent.score_items(obs).shape == (7,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_agents_linear_bandit.py -v -k lints`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# rl_recsys/agents/lin_ts.py
from __future__ import annotations

import numpy as np

from rl_recsys.agents._linear_base import LinearBanditBase
from rl_recsys.environments.base import RecObs


class LinTSAgent(LinearBanditBase):
    """Linear Thompson sampling: sample θ ~ N(A^{-1} b, σ^2 A^{-1}) and
    score with features @ θ."""

    def __init__(
        self,
        slate_size: int,
        user_dim: int,
        item_dim: int,
        *,
        sigma: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(slate_size=slate_size, user_dim=user_dim, item_dim=item_dim)
        self._sigma = float(sigma)
        self._rng = rng if rng is not None else np.random.default_rng()

    def score_items(self, obs: RecObs) -> np.ndarray:
        features = self._candidate_features(obs)
        a_inv = np.linalg.inv(self._a_matrix)
        mu = a_inv @ self._b_vector
        cov = self._sigma ** 2 * a_inv
        # Symmetrize for numerical stability before Cholesky.
        cov = 0.5 * (cov + cov.T)
        # Add tiny jitter — A starts at I so cov is well-conditioned, but be safe.
        cov += 1e-9 * np.eye(cov.shape[0])
        l = np.linalg.cholesky(cov)
        z = self._rng.standard_normal(size=mu.shape[0])
        theta = mu + l @ z
        return features @ theta
```

- [ ] **Step 4: Register, run, commit**

```python
# in factory.py
from rl_recsys.agents.lin_ts import LinTSAgent

def _build_lin_ts(agent_cfg: AgentConfig, env_cfg: EnvConfig) -> Agent:
    return LinTSAgent(
        slate_size=env_cfg.slate_size,
        user_dim=env_cfg.user_dim,
        item_dim=env_cfg.item_dim,
        sigma=getattr(agent_cfg, "sigma", 1.0),
    )

AGENT_REGISTRY["lin_ts"] = _build_lin_ts
```

(`AgentConfig` may need a new optional `sigma` field — see Task 19's config pass for the bulk addition.)

Run: `.venv/bin/pytest tests/test_agents_linear_bandit.py tests/test_agents_dataset_agnostic.py -v`
Expected: PASS.

```bash
git add rl_recsys/agents/lin_ts.py rl_recsys/agents/factory.py rl_recsys/agents/__init__.py tests/test_agents_linear_bandit.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat: add LinTSAgent (Linear Thompson Sampling)

Samples theta ~ N(A^{-1} b, sigma^2 A^{-1}) per call to select_slate.
Shares feature pipeline + update with LinUCB via LinearBanditBase.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: `EpsGreedyLinearAgent`

**Files:**
- Create: `rl_recsys/agents/eps_greedy_linear.py`
- Modify: factory, `__init__.py`
- Test: `tests/test_agents_linear_bandit.py` (extend)

- [ ] **Step 1: Write failing tests**

```python
def test_eps_greedy_explores_at_eps_one():
    from rl_recsys.agents.eps_greedy_linear import EpsGreedyLinearAgent

    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.eye(8, 3),
        candidate_ids=np.arange(8, dtype=np.int64),
    )
    agent = EpsGreedyLinearAgent(
        slate_size=3, user_dim=4, item_dim=3,
        epsilon=1.0, rng=np.random.default_rng(0),
    )
    slate = agent.select_slate(obs)
    assert slate.shape == (3,)
    assert len(set(slate.tolist())) == 3


def test_eps_greedy_exploits_at_eps_zero():
    from rl_recsys.agents.eps_greedy_linear import EpsGreedyLinearAgent

    obs = RecObs(
        user_features=np.array([1.0, 0, 0, 0]),
        candidate_features=np.eye(8, 3),
        candidate_ids=np.arange(8, dtype=np.int64),
    )
    agent = EpsGreedyLinearAgent(
        slate_size=3, user_dim=4, item_dim=3,
        epsilon=0.0,
    )
    # Inject a hand-set b vector that prefers items 5,6,7.
    agent._b_vector = np.zeros_like(agent._b_vector)
    # Items rows 5,6,7 in candidate_features=np.eye(8,3) have user||item||
    # interaction = (user, eye(8,3)[i], user[:3]*eye(8,3)[i,:3]).
    # Easier: just set b to push features such that items 5,6,7 win — set
    # entry that interacts with item index correlation. We sidestep by
    # directly checking: when epsilon=0, the call must reduce to argsort
    # of score_items.
    scores = agent.score_items(obs)
    expected_top3 = set(np.argsort(scores)[-3:].tolist())
    assert set(agent.select_slate(obs).tolist()) == expected_top3
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/pytest tests/test_agents_linear_bandit.py -v -k eps_greedy`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# rl_recsys/agents/eps_greedy_linear.py
from __future__ import annotations

import numpy as np

from rl_recsys.agents._linear_base import LinearBanditBase
from rl_recsys.environments.base import RecObs


class EpsGreedyLinearAgent(LinearBanditBase):
    """Linear regression scorer with ε-greedy exploration."""

    def __init__(
        self,
        slate_size: int,
        user_dim: int,
        item_dim: int,
        *,
        epsilon: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(slate_size=slate_size, user_dim=user_dim, item_dim=item_dim)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self._epsilon = float(epsilon)
        self._rng = rng if rng is not None else np.random.default_rng()

    def score_items(self, obs: RecObs) -> np.ndarray:
        features = self._candidate_features(obs)
        theta = np.linalg.solve(self._a_matrix, self._b_vector)
        return features @ theta

    def select_slate(self, obs: RecObs) -> np.ndarray:
        n = len(obs.candidate_features)
        if self._slate_size > n:
            raise ValueError(
                f"slate_size={self._slate_size} exceeds num_candidates={n}"
            )
        if self._rng.random() < self._epsilon:
            return self._rng.choice(n, size=self._slate_size, replace=False).astype(
                np.int64
            )
        scores = self.score_items(obs)
        return np.argsort(scores)[-self._slate_size:][::-1]
```

- [ ] **Step 4: Register, run, commit**

```python
# in factory.py
from rl_recsys.agents.eps_greedy_linear import EpsGreedyLinearAgent

def _build_eps_greedy_linear(agent_cfg, env_cfg):
    return EpsGreedyLinearAgent(
        slate_size=env_cfg.slate_size,
        user_dim=env_cfg.user_dim,
        item_dim=env_cfg.item_dim,
        epsilon=getattr(agent_cfg, "epsilon", 0.1),
    )

AGENT_REGISTRY["eps_greedy_linear"] = _build_eps_greedy_linear
```

Run: `.venv/bin/pytest tests/test_agents_linear_bandit.py tests/test_agents_dataset_agnostic.py -v`
Expected: PASS.

```bash
git add rl_recsys/agents/eps_greedy_linear.py rl_recsys/agents/factory.py rl_recsys/agents/__init__.py tests/test_agents_linear_bandit.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat: add EpsGreedyLinearAgent

Linear regression scores with epsilon-rate uniform exploration. Shares
feature pipeline with LinUCB via LinearBanditBase.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: `BoltzmannLinearAgent`

**Files:**
- Create: `rl_recsys/agents/boltzmann_linear.py`
- Modify: factory, `__init__.py`
- Test: `tests/test_agents_linear_bandit.py` (extend)

- [ ] **Step 1: Write failing tests**

```python
def test_boltzmann_collapses_to_argmax_at_low_T():
    from rl_recsys.agents.boltzmann_linear import BoltzmannLinearAgent

    obs = RecObs(
        user_features=np.array([1.0, 0, 0, 0]),
        candidate_features=np.eye(8, 3),
        candidate_ids=np.arange(8, dtype=np.int64),
    )
    agent = BoltzmannLinearAgent(
        slate_size=3, user_dim=4, item_dim=3,
        temperature=1e-4, rng=np.random.default_rng(0),
    )
    scores = agent.score_items(obs)
    top3 = set(np.argsort(scores)[-3:].tolist())
    assert set(agent.select_slate(obs).tolist()) == top3


def test_boltzmann_select_slate_returns_unique_indices():
    from rl_recsys.agents.boltzmann_linear import BoltzmannLinearAgent

    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.eye(20, 3),
        candidate_ids=np.arange(20, dtype=np.int64),
    )
    agent = BoltzmannLinearAgent(
        slate_size=5, user_dim=4, item_dim=3,
        temperature=1.0, rng=np.random.default_rng(0),
    )
    for _ in range(5):
        slate = agent.select_slate(obs)
        assert len(set(slate.tolist())) == 5
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/pytest tests/test_agents_linear_bandit.py -v -k boltzmann`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# rl_recsys/agents/boltzmann_linear.py
from __future__ import annotations

import numpy as np

from rl_recsys.agents._linear_base import LinearBanditBase
from rl_recsys.environments.base import RecObs


class BoltzmannLinearAgent(LinearBanditBase):
    """Plackett-Luce sampling without replacement from softmax(scores/T)
    via the Gumbel-top-K trick."""

    def __init__(
        self,
        slate_size: int,
        user_dim: int,
        item_dim: int,
        *,
        temperature: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(slate_size=slate_size, user_dim=user_dim, item_dim=item_dim)
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self._temperature = float(temperature)
        self._rng = rng if rng is not None else np.random.default_rng()

    def score_items(self, obs: RecObs) -> np.ndarray:
        features = self._candidate_features(obs)
        theta = np.linalg.solve(self._a_matrix, self._b_vector)
        return features @ theta

    def select_slate(self, obs: RecObs) -> np.ndarray:
        n = len(obs.candidate_features)
        if self._slate_size > n:
            raise ValueError(
                f"slate_size={self._slate_size} exceeds num_candidates={n}"
            )
        scores = self.score_items(obs) / self._temperature
        # Gumbel-top-K: argsort(scores + Gumbel(0,1)) for sampling without
        # replacement from softmax(scores).
        gumbel = -np.log(-np.log(self._rng.uniform(size=n) + 1e-20) + 1e-20)
        return np.argsort(scores + gumbel)[-self._slate_size:][::-1].astype(np.int64)
```

- [ ] **Step 4: Register, run, commit**

```python
# factory
from rl_recsys.agents.boltzmann_linear import BoltzmannLinearAgent

def _build_boltzmann_linear(agent_cfg, env_cfg):
    return BoltzmannLinearAgent(
        slate_size=env_cfg.slate_size,
        user_dim=env_cfg.user_dim,
        item_dim=env_cfg.item_dim,
        temperature=getattr(agent_cfg, "temperature", 1.0),
    )

AGENT_REGISTRY["boltzmann_linear"] = _build_boltzmann_linear
```

Run + commit (same pattern as Task 11).

```bash
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat: add BoltzmannLinearAgent

Softmax-T over linear scores; Plackett-Luce sampling without
replacement via the Gumbel-top-K trick.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: `NeuralLinearAgent`

**Files:**
- Create: `rl_recsys/agents/neural_linear.py`
- Modify: factory, `__init__.py`
- Test: `tests/test_agents_neural.py` (new)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agents_neural.py
from __future__ import annotations

import numpy as np
import pytest

from rl_recsys.environments.base import RecObs
from rl_recsys.evaluation.ope_trajectory import LoggedTrajectoryStep


def _make_step(num_candidates=8, slate_size=3):
    obs = RecObs(
        user_features=np.array([1.0, 0.0, 0.0, 0.0]),
        candidate_features=np.eye(num_candidates, 3),
        candidate_ids=np.arange(num_candidates, dtype=np.int64),
    )
    return LoggedTrajectoryStep(
        obs=obs,
        logged_action=np.array([0, 1, 2], dtype=np.int64)[:slate_size],
        logged_reward=1.0,
        logged_clicks=np.array([1, 0, 1], dtype=np.int64)[:slate_size],
        propensity=0.1,
    )


class _StubSource:
    def __init__(self, n: int = 4):
        self._n = n

    def iter_trajectories(self, *, max_trajectories=None, seed=0):
        for _ in range(self._n):
            yield [_make_step(), _make_step()]


def test_neural_linear_train_offline_completes_on_tiny_dataset():
    from rl_recsys.agents.neural_linear import NeuralLinearAgent

    agent = NeuralLinearAgent(
        slate_size=3, user_dim=4, item_dim=3,
        hidden_dim=8, embedding_dim=4, mlp_epochs=1, alpha=1.0,
        device="cpu",
    )
    metrics = agent.train_offline(_StubSource(n=4), seed=0)
    assert "mlp_loss" in metrics


def test_neural_linear_score_shape():
    from rl_recsys.agents.neural_linear import NeuralLinearAgent

    agent = NeuralLinearAgent(
        slate_size=3, user_dim=4, item_dim=3,
        hidden_dim=8, embedding_dim=4, mlp_epochs=1, alpha=1.0,
        device="cpu",
    )
    agent.train_offline(_StubSource(n=2), seed=0)
    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.eye(7, 3),
        candidate_ids=np.arange(7, dtype=np.int64),
    )
    assert agent.score_items(obs).shape == (7,)
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/pytest tests/test_agents_neural.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# rl_recsys/agents/neural_linear.py
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecObs


class _NeuralLinearMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralLinearAgent(Agent):
    """MLP feature extractor + LinUCB head in embedding space (Riquelme 2018)."""

    def __init__(
        self,
        slate_size: int,
        user_dim: int,
        item_dim: int,
        *,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        mlp_epochs: int = 5,
        alpha: float = 1.0,
        device: str = "cuda",
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA required for NeuralLinearAgent; pass device='cpu' to run on CPU"
            )
        self._slate_size = int(slate_size)
        self._user_dim = int(user_dim)
        self._item_dim = int(item_dim)
        self._embedding_dim = int(embedding_dim)
        self._mlp_epochs = int(mlp_epochs)
        self._alpha = float(alpha)
        self._device = torch.device(device)
        self._mlp = _NeuralLinearMLP(
            user_dim + item_dim, hidden_dim, embedding_dim
        ).to(self._device)
        self._a_matrix = np.eye(embedding_dim, dtype=np.float64)
        self._b_vector = np.zeros(embedding_dim, dtype=np.float64)

    def train_offline(self, source, *, seed: int = 0) -> dict[str, float]:
        torch.manual_seed(seed)
        # Materialize the regression dataset (context_features, click) one
        # tuple per (step, slate position).
        rows_x: list[np.ndarray] = []
        rows_y: list[float] = []
        for traj in source.iter_trajectories(seed=seed):
            for step in traj:
                u = step.obs.user_features
                items = step.obs.candidate_features[step.logged_action]
                for item, click in zip(items, step.logged_clicks):
                    rows_x.append(np.concatenate([u, item]))
                    rows_y.append(float(click))
        if not rows_x:
            return {"mlp_loss": float("nan"), "items_seen": 0.0}
        x = torch.tensor(np.stack(rows_x), dtype=torch.float32, device=self._device)
        y = torch.tensor(rows_y, dtype=torch.float32, device=self._device)

        opt = torch.optim.Adam(self._mlp.parameters(), lr=1e-3)
        last_loss = float("nan")
        for _ in range(self._mlp_epochs):
            # One linear layer on top of the MLP for click regression. The
            # head is discarded; we only keep the embedding.
            head = nn.Linear(self._embedding_dim, 1).to(self._device)
            head_opt = torch.optim.Adam(head.parameters(), lr=1e-3)
            for batch_start in range(0, len(x), 4096):
                xb = x[batch_start : batch_start + 4096]
                yb = y[batch_start : batch_start + 4096]
                phi = self._mlp(xb)
                pred = head(phi).squeeze(-1)
                loss = nn.functional.mse_loss(pred, yb)
                opt.zero_grad()
                head_opt.zero_grad()
                loss.backward()
                opt.step()
                head_opt.step()
                last_loss = float(loss.item())

        # Recompute (A, b) in embedding space using the trained MLP.
        self._mlp.eval()
        with torch.no_grad():
            phi = self._mlp(x).cpu().numpy().astype(np.float64)
        clicks = y.cpu().numpy().astype(np.float64)
        self._a_matrix = np.eye(self._embedding_dim, dtype=np.float64)
        self._b_vector = np.zeros(self._embedding_dim, dtype=np.float64)
        for p, c in zip(phi, clicks):
            self._a_matrix += np.outer(p, p)
            self._b_vector += c * p
        return {"mlp_loss": last_loss, "items_seen": float(len(x))}

    def _embed(self, obs: RecObs) -> np.ndarray:
        u = np.broadcast_to(
            obs.user_features, (len(obs.candidate_features), self._user_dim)
        )
        x = np.concatenate([u, obs.candidate_features], axis=1)
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32, device=self._device)
            phi = self._mlp(t).cpu().numpy().astype(np.float64)
        return phi

    def score_items(self, obs: RecObs) -> np.ndarray:
        phi = self._embed(obs)
        theta = np.linalg.solve(self._a_matrix, self._b_vector)
        means = phi @ theta
        solved = np.linalg.solve(self._a_matrix, phi.T).T
        variances = np.einsum("ij,ij->i", phi, solved)
        bonuses = self._alpha * np.sqrt(np.clip(variances, 0.0, None))
        return means + bonuses

    def select_slate(self, obs: RecObs) -> np.ndarray:
        n = len(obs.candidate_features)
        if self._slate_size > n:
            raise ValueError(
                f"slate_size={self._slate_size} exceeds num_candidates={n}"
            )
        return np.argsort(self.score_items(obs))[-self._slate_size:][::-1]

    def update(self, obs, slate, reward, clicks, next_obs) -> dict[str, float]:
        # NeuralLinear is a batch-trained agent; per-step update is a no-op.
        return {}
```

- [ ] **Step 4: Register, run, commit**

```python
# factory
from rl_recsys.agents.neural_linear import NeuralLinearAgent

def _build_neural_linear(agent_cfg, env_cfg):
    return NeuralLinearAgent(
        slate_size=env_cfg.slate_size,
        user_dim=env_cfg.user_dim,
        item_dim=env_cfg.item_dim,
        hidden_dim=getattr(agent_cfg, "hidden_dim", 64),
        embedding_dim=getattr(agent_cfg, "embedding_dim", 32),
        mlp_epochs=getattr(agent_cfg, "epochs", 5),
        alpha=getattr(agent_cfg, "alpha", 1.0),
        device=getattr(agent_cfg, "device", "cuda"),
    )

AGENT_REGISTRY["neural_linear"] = _build_neural_linear
```

Run: `.venv/bin/pytest tests/test_agents_neural.py tests/test_agents_dataset_agnostic.py -v`
Expected: PASS.

```bash
git add rl_recsys/agents/neural_linear.py rl_recsys/agents/factory.py rl_recsys/agents/__init__.py tests/test_agents_neural.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat: add NeuralLinearAgent (MLP features + LinUCB head)

Riquelme et al. NeuralLinear: train an MLP on click regression, then
recompute LinUCB sufficient statistics in the frozen embedding space.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: `BCAgent`

**Files:**
- Create: `rl_recsys/agents/bc.py`
- Modify: factory, `__init__.py`
- Test: `tests/test_agents_imitation.py` (new)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agents_imitation.py
from __future__ import annotations

import numpy as np
import pytest

from rl_recsys.environments.base import RecObs
from rl_recsys.evaluation.ope_trajectory import LoggedTrajectoryStep


def _stub_step(num_candidates=8):
    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.eye(num_candidates, 3),
        candidate_ids=np.arange(num_candidates, dtype=np.int64),
    )
    return LoggedTrajectoryStep(
        obs=obs,
        logged_action=np.array([0, 1, 2], dtype=np.int64),
        logged_reward=1.0,
        logged_clicks=np.array([1, 0, 1], dtype=np.int64),
        propensity=0.1,
    )


class _StubSource:
    def iter_trajectories(self, *, max_trajectories=None, seed=0):
        yield [_stub_step()] * 2


def test_bc_score_items_shape():
    from rl_recsys.agents.bc import BCAgent
    from rl_recsys.evaluation.behavior_policy import BehaviorPolicy

    policy = BehaviorPolicy(
        user_dim=4, item_dim=3, slate_size=3, num_items=8, device="cpu",
    )
    agent = BCAgent(
        slate_size=3, behavior_policy=policy,
        candidate_features=np.eye(8, 3),
    )
    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.eye(8, 3),
        candidate_ids=np.arange(8, dtype=np.int64),
    )
    assert agent.score_items(obs).shape == (8,)


def test_bc_train_offline_with_prefit_policy_is_noop():
    from rl_recsys.agents.bc import BCAgent
    from rl_recsys.evaluation.behavior_policy import BehaviorPolicy

    policy = BehaviorPolicy(
        user_dim=4, item_dim=3, slate_size=3, num_items=8, device="cpu",
    )
    agent = BCAgent(
        slate_size=3, behavior_policy=policy,
        candidate_features=np.eye(8, 3),
    )
    metrics = agent.train_offline(_StubSource(), seed=0)
    assert metrics == {}


def test_bc_raises_when_score_called_without_policy():
    from rl_recsys.agents.bc import BCAgent

    agent = BCAgent(
        slate_size=3, behavior_policy=None,
        candidate_features=np.eye(8, 3),
    )
    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.eye(8, 3),
        candidate_ids=np.arange(8, dtype=np.int64),
    )
    with pytest.raises(RuntimeError, match="behavior_policy"):
        agent.score_items(obs)
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/pytest tests/test_agents_imitation.py -v -k bc`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# rl_recsys/agents/bc.py
from __future__ import annotations

import numpy as np
import torch

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecObs
from rl_recsys.evaluation.behavior_policy import BehaviorPolicy


class BCAgent(Agent):
    """Behavior cloning: wraps a pre-fit BehaviorPolicy. Selects the top-k
    items by sum-of-position log-softmax scores."""

    def __init__(
        self,
        slate_size: int,
        candidate_features: np.ndarray,
        *,
        behavior_policy: BehaviorPolicy | None = None,
    ) -> None:
        self._slate_size = int(slate_size)
        self._candidate_features = np.asarray(candidate_features, dtype=np.float64)
        self._behavior_policy = behavior_policy

    def train_offline(self, source, *, seed: int = 0) -> dict[str, float]:
        # BC reuses an already-fit BehaviorPolicy; no per-agent training.
        # The grid runner injects the fitted policy via inject_behavior_policy().
        return {}

    def inject_behavior_policy(self, policy: BehaviorPolicy) -> None:
        self._behavior_policy = policy

    def _per_position_log_probs(self, obs: RecObs) -> np.ndarray:
        # Returns log-prob matrix of shape (slate_size, num_candidates).
        user = obs.user_features.astype(np.float64)[None, :]
        log_probs = np.zeros(
            (self._slate_size, len(obs.candidate_features)), dtype=np.float64
        )
        for k in range(self._slate_size):
            logits = self._behavior_policy._score_position(
                torch.tensor(user, dtype=torch.float32),
                torch.tensor(obs.candidate_features, dtype=torch.float32),
                k,
            )
            log_probs[k] = (
                torch.nn.functional.log_softmax(logits, dim=-1)
                .cpu().numpy().astype(np.float64).squeeze(0)
            )
        return log_probs

    def score_items(self, obs: RecObs) -> np.ndarray:
        if self._behavior_policy is None:
            raise RuntimeError("BCAgent.score_items called before train_offline")
        return self._per_position_log_probs(obs).sum(axis=0)

    def select_slate(self, obs: RecObs) -> np.ndarray:
        scores = self.score_items(obs)
        return np.argsort(scores)[-self._slate_size:][::-1].astype(np.int64)

    def update(self, obs, slate, reward, clicks, next_obs) -> dict[str, float]:
        return {}
```

> Note: this assumes `BehaviorPolicy._score_position(user_t, cand_t, k) -> logits`. If the actual private API differs, swap to `_score_batch`. Read `rl_recsys/evaluation/behavior_policy.py` and use whichever method gives per-position logits over the full candidate universe; the public `slate_log_propensities_batch` is for full slates and isn't useful here.

- [ ] **Step 4: Register, run, commit**

```python
# factory
from rl_recsys.agents.bc import BCAgent

def _build_bc(agent_cfg, env_cfg):
    # candidate_features and behavior_policy must be injected by the runner
    # (Task 20) before train_offline. The factory constructs a stub.
    return BCAgent(
        slate_size=env_cfg.slate_size,
        candidate_features=np.zeros((env_cfg.num_candidates, env_cfg.item_dim)),
    )

AGENT_REGISTRY["bc"] = _build_bc
```

> Note: BC's BehaviorPolicy is supplied by the runner via `inject_behavior_policy`. The CLI fits the policy once for the whole grid and passes it down. `_candidate_features` is similarly injected from the source. The alternative — making the factory dataset-aware — violates the dataset-agnostic constraint.

Run: `.venv/bin/pytest tests/test_agents_imitation.py tests/test_agents_dataset_agnostic.py -v`
Expected: PASS.

```bash
git add rl_recsys/agents/bc.py rl_recsys/agents/factory.py rl_recsys/agents/__init__.py tests/test_agents_imitation.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat: add BCAgent (behavior cloning over BehaviorPolicy)

Wraps a fit BehaviorPolicy and ranks candidates by sum-of-position
log-softmax. train_offline fits a fresh BehaviorPolicy when none is
supplied.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: `GBDTAgent` + `lightgbm` dependency

**Files:**
- Create: `rl_recsys/agents/gbdt.py`
- Modify: factory, `__init__.py`, `requirements.txt`
- Test: `tests/test_agents_imitation.py` (extend)

- [ ] **Step 1: Add `lightgbm` to `requirements.txt`**

Append `lightgbm>=4.0` to `requirements.txt`. Then:

```bash
.venv/bin/pip install lightgbm
```

- [ ] **Step 2: Write failing tests**

Append to `tests/test_agents_imitation.py`:

```python
def test_gbdt_train_offline_returns_metrics():
    from rl_recsys.agents.gbdt import GBDTAgent

    agent = GBDTAgent(
        slate_size=3, candidate_features=np.eye(8, 3),
        n_estimators=10, max_depth=3,
    )
    metrics = agent.train_offline(_StubSource(), seed=0)
    assert "n_train_rows" in metrics


def test_gbdt_score_shape():
    from rl_recsys.agents.gbdt import GBDTAgent

    agent = GBDTAgent(
        slate_size=3, candidate_features=np.eye(8, 3),
        n_estimators=10, max_depth=3,
    )
    agent.train_offline(_StubSource(), seed=0)
    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.eye(8, 3),
        candidate_ids=np.arange(8, dtype=np.int64),
    )
    assert agent.score_items(obs).shape == (8,)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_agents_imitation.py -v -k gbdt`
Expected: FAIL.

- [ ] **Step 4: Implement**

```python
# rl_recsys/agents/gbdt.py
from __future__ import annotations

import numpy as np

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecObs


class GBDTAgent(Agent):
    """LightGBM regressor on (concat(user, item), click)."""

    def __init__(
        self,
        slate_size: int,
        candidate_features: np.ndarray,
        *,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.05,
    ) -> None:
        self._slate_size = int(slate_size)
        self._candidate_features = np.asarray(candidate_features, dtype=np.float64)
        self._n_estimators = int(n_estimators)
        self._max_depth = int(max_depth)
        self._learning_rate = float(learning_rate)
        self._model = None

    def train_offline(self, source, *, seed: int = 0) -> dict[str, float]:
        try:
            import lightgbm as lgb  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "GBDTAgent requires lightgbm — pip install lightgbm"
            ) from exc

        rows_x: list[np.ndarray] = []
        rows_y: list[float] = []
        for traj in source.iter_trajectories(seed=seed):
            for step in traj:
                u = step.obs.user_features
                items = step.obs.candidate_features[step.logged_action]
                for item, click in zip(items, step.logged_clicks):
                    rows_x.append(np.concatenate([u, item]))
                    rows_y.append(float(click))
        if not rows_x:
            return {"n_train_rows": 0.0}
        x = np.stack(rows_x)
        y = np.asarray(rows_y, dtype=np.float64)
        self._model = lgb.LGBMRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            random_state=seed,
            verbose=-1,
        )
        self._model.fit(x, y)
        return {"n_train_rows": float(len(x))}

    def score_items(self, obs: RecObs) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("GBDTAgent.score_items called before train_offline")
        u = np.broadcast_to(
            obs.user_features,
            (len(obs.candidate_features), len(obs.user_features)),
        )
        x = np.concatenate([u, obs.candidate_features], axis=1)
        return self._model.predict(x).astype(np.float64)

    def select_slate(self, obs: RecObs) -> np.ndarray:
        return np.argsort(self.score_items(obs))[-self._slate_size:][::-1].astype(
            np.int64
        )

    def update(self, obs, slate, reward, clicks, next_obs) -> dict[str, float]:
        return {}
```

- [ ] **Step 5: Register, run, commit**

```python
# factory
from rl_recsys.agents.gbdt import GBDTAgent

def _build_gbdt(agent_cfg, env_cfg):
    return GBDTAgent(
        slate_size=env_cfg.slate_size,
        candidate_features=np.zeros((env_cfg.num_candidates, env_cfg.item_dim)),
        n_estimators=getattr(agent_cfg, "n_estimators", 100),
        max_depth=getattr(agent_cfg, "max_depth", 6),
        learning_rate=getattr(agent_cfg, "learning_rate", 0.05),
    )

AGENT_REGISTRY["gbdt"] = _build_gbdt
```

(Same `candidate_features` injection contract as BCAgent — runner sets it before training.)

Run: `.venv/bin/pytest tests/test_agents_imitation.py tests/test_agents_dataset_agnostic.py -v`
Expected: PASS.

```bash
git add rl_recsys/agents/gbdt.py rl_recsys/agents/factory.py rl_recsys/agents/__init__.py tests/test_agents_imitation.py requirements.txt
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat: add GBDTAgent (LightGBM ranker)

Per-(step, slate position) row regressor. Adds lightgbm>=4.0 to
requirements.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: `SASRecAgent`

**Files:**
- Create: `rl_recsys/agents/sasrec.py`
- Modify: factory, `__init__.py`
- Test: `tests/test_agents_dl.py` (new)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agents_dl.py
from __future__ import annotations

import numpy as np
import pytest

from rl_recsys.environments.base import HistoryStep, RecObs
from rl_recsys.evaluation.ope_trajectory import LoggedTrajectoryStep


def _stub_step(history=()):
    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.eye(8, 3),
        candidate_ids=np.arange(8, dtype=np.int64),
        history=history,
    )
    return LoggedTrajectoryStep(
        obs=obs,
        logged_action=np.array([0, 1, 2], dtype=np.int64),
        logged_reward=1.0,
        logged_clicks=np.array([1, 0, 1], dtype=np.int64),
        propensity=0.1,
    )


class _StubSource:
    def iter_trajectories(self, *, max_trajectories=None, seed=0):
        h: tuple = ()
        traj = []
        for _ in range(3):
            step = _stub_step(history=h)
            traj.append(step)
            h = h + (HistoryStep(step.logged_action, step.logged_clicks),)
        yield traj


def test_sasrec_forward_shape():
    from rl_recsys.agents.sasrec import SASRecAgent

    agent = SASRecAgent(
        slate_size=3, num_candidates=8, item_dim=3,
        hidden_dim=8, n_heads=2, n_blocks=1, max_history_len=5,
        epochs=1, device="cpu",
    )
    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.eye(8, 3),
        candidate_ids=np.arange(8, dtype=np.int64),
    )
    assert agent.score_items(obs).shape == (8,)


def test_sasrec_handles_empty_history_via_sentinel():
    from rl_recsys.agents.sasrec import SASRecAgent

    agent = SASRecAgent(
        slate_size=3, num_candidates=8, item_dim=3,
        hidden_dim=8, n_heads=2, n_blocks=1, max_history_len=5,
        epochs=1, device="cpu",
    )
    obs_empty = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.eye(8, 3),
        candidate_ids=np.arange(8, dtype=np.int64),
        history=(),
    )
    # Must run without raising.
    scores = agent.score_items(obs_empty)
    assert scores.shape == (8,)


def test_sasrec_train_offline_runs():
    from rl_recsys.agents.sasrec import SASRecAgent

    agent = SASRecAgent(
        slate_size=3, num_candidates=8, item_dim=3,
        hidden_dim=8, n_heads=2, n_blocks=1, max_history_len=5,
        epochs=1, device="cpu",
    )
    metrics = agent.train_offline(_StubSource(), seed=0)
    assert "loss" in metrics or "epochs" in metrics
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_agents_dl.py -v -k sasrec`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# rl_recsys/agents/sasrec.py
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecObs


class _SASRecEncoder(nn.Module):
    def __init__(
        self,
        num_candidates: int,
        hidden_dim: int,
        n_heads: int,
        n_blocks: int,
        max_history_len: int,
    ) -> None:
        super().__init__()
        # +1 for pad/sentinel token at index `num_candidates`.
        self._pad_idx = num_candidates
        self.item_emb = nn.Embedding(num_candidates + 1, hidden_dim, padding_idx=self._pad_idx)
        self.pos_emb = nn.Embedding(max_history_len + 1, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=4 * hidden_dim,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)

    def forward(
        self,
        item_ids: torch.Tensor,        # (B, T, slate_size) long
        click_mask: torch.Tensor,      # (B, T, slate_size) float (0/1)
        positions: torch.Tensor,       # (B, T) long
        attn_mask: torch.Tensor | None,  # (B, T+1) bool, True = pad
    ) -> torch.Tensor:
        # Token embedding: mean-pool slate items, scaled by (1 + click).
        emb = self.item_emb(item_ids) * (1.0 + click_mask).unsqueeze(-1)
        token = emb.mean(dim=2)
        token = token + self.pos_emb(positions)
        cls = self.cls_token.expand(token.size(0), -1, -1)
        seq = torch.cat([cls, token], dim=1)
        out = self.encoder(seq, src_key_padding_mask=attn_mask)
        return out[:, -1, :]  # last position pooled state


class SASRecAgent(Agent):
    """Self-attention sequential ranker: encoder over (slate, clicks)
    history; per-candidate score = h_session . W_out @ item_embedding."""

    def __init__(
        self,
        slate_size: int,
        num_candidates: int,
        item_dim: int,
        *,
        hidden_dim: int = 64,
        n_heads: int = 2,
        n_blocks: int = 2,
        max_history_len: int = 20,
        epochs: int = 10,
        device: str = "cuda",
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA required for SASRecAgent; pass device='cpu' to run on CPU"
            )
        self._slate_size = int(slate_size)
        self._num_candidates = int(num_candidates)
        self._item_dim = int(item_dim)
        self._max_history_len = int(max_history_len)
        self._epochs = int(epochs)
        self._hidden_dim = int(hidden_dim)
        self._device = torch.device(device)
        self._encoder = _SASRecEncoder(
            num_candidates=num_candidates,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_blocks=n_blocks,
            max_history_len=max_history_len,
        ).to(self._device)
        self._out = nn.Linear(hidden_dim, hidden_dim, bias=True).to(self._device)

    def _build_history_tensors(self, history) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T = min(len(history), self._max_history_len)
        if T == 0:
            ids = torch.full(
                (1, 1, self._slate_size), self._encoder._pad_idx,
                dtype=torch.long, device=self._device,
            )
            clicks = torch.zeros((1, 1, self._slate_size), device=self._device)
            pos = torch.zeros((1, 1), dtype=torch.long, device=self._device)
            return ids, clicks, pos
        recent = history[-T:]
        ids = torch.tensor(
            np.stack([h.slate for h in recent]), dtype=torch.long, device=self._device
        ).unsqueeze(0)
        clicks = torch.tensor(
            np.stack([h.clicks for h in recent]), dtype=torch.float32, device=self._device,
        ).unsqueeze(0)
        pos = torch.arange(1, T + 1, dtype=torch.long, device=self._device).unsqueeze(0)
        return ids, clicks, pos

    def score_items(self, obs: RecObs) -> np.ndarray:
        self._encoder.eval()
        with torch.no_grad():
            ids, clicks, pos = self._build_history_tensors(obs.history)
            h = self._encoder(ids, clicks, pos, attn_mask=None)
            h = self._out(h)
            all_items = torch.arange(
                self._num_candidates, dtype=torch.long, device=self._device,
            )
            item_emb = self._encoder.item_emb(all_items)
            scores = (item_emb @ h.squeeze(0)).cpu().numpy().astype(np.float64)
        return scores

    def select_slate(self, obs: RecObs) -> np.ndarray:
        return np.argsort(self.score_items(obs))[-self._slate_size:][::-1].astype(
            np.int64
        )

    def train_offline(self, source, *, seed: int = 0) -> dict[str, float]:
        torch.manual_seed(seed)
        self._encoder.train()
        opt = torch.optim.Adam(
            list(self._encoder.parameters()) + list(self._out.parameters()),
            lr=1e-3,
        )
        last_loss = float("nan")
        for epoch in range(self._epochs):
            for traj in source.iter_trajectories(seed=seed + epoch):
                # At each step (except the first, which has no history),
                # predict the items the user actually clicked at this step
                # given the prior session history.
                for step in traj:
                    if len(step.obs.history) == 0:
                        continue
                    ids, clicks, pos = self._build_history_tensors(step.obs.history)
                    pooled = self._out(
                        self._encoder(ids, clicks, pos, attn_mask=None)
                    )
                    all_items = torch.arange(
                        self._num_candidates, dtype=torch.long, device=self._device,
                    )
                    item_emb = self._encoder.item_emb(all_items)
                    logits = item_emb @ pooled.squeeze(0)
                    logp = torch.log_softmax(logits, dim=-1)
                    pos_idx = torch.tensor(
                        step.logged_action[step.logged_clicks > 0],
                        dtype=torch.long, device=self._device,
                    )
                    if pos_idx.numel() == 0:
                        continue
                    loss = -logp[pos_idx].mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    last_loss = float(loss.item())
        return {"loss": last_loss, "epochs": float(self._epochs)}

    def update(self, obs, slate, reward, clicks, next_obs) -> dict[str, float]:
        return {}
```

> Note: the `train_offline` implementation above is the smallest one that exercises the pieces. A future batch can add proper batching and negative sampling — for now we want correctness on small inputs.

- [ ] **Step 4: Register, run, commit**

```python
# factory
from rl_recsys.agents.sasrec import SASRecAgent

def _build_sasrec(agent_cfg, env_cfg):
    return SASRecAgent(
        slate_size=env_cfg.slate_size,
        num_candidates=env_cfg.num_candidates,
        item_dim=env_cfg.item_dim,
        hidden_dim=getattr(agent_cfg, "hidden_dim", 64),
        n_heads=getattr(agent_cfg, "n_heads", 2),
        n_blocks=getattr(agent_cfg, "n_blocks", 2),
        max_history_len=getattr(agent_cfg, "max_history_len", 20),
        epochs=getattr(agent_cfg, "epochs", 10),
        device=getattr(agent_cfg, "device", "cuda"),
    )

AGENT_REGISTRY["sasrec"] = _build_sasrec
```

Run: `.venv/bin/pytest tests/test_agents_dl.py tests/test_agents_dataset_agnostic.py -v`
Expected: PASS.

```bash
git add rl_recsys/agents/sasrec.py rl_recsys/agents/factory.py rl_recsys/agents/__init__.py tests/test_agents_dl.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat: add SASRecAgent (transformer over session history)

Self-attention encoder over (slate, clicks) history; per-candidate
score = item_embedding . pooled_session_state.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 17: `TopKReinforceAgent`

**Files:**
- Create: `rl_recsys/agents/topk_reinforce.py`
- Modify: factory, `__init__.py`
- Test: `tests/test_agents_dl.py` (extend)

- [ ] **Step 1: Write failing tests**

```python
def test_topk_reinforce_score_shape():
    from rl_recsys.agents.topk_reinforce import TopKReinforceAgent

    agent = TopKReinforceAgent(
        slate_size=3, num_candidates=8, item_dim=3,
        hidden_dim=8, n_heads=2, n_blocks=1, max_history_len=5,
        epochs=1, clip_c=10.0, device="cpu",
    )
    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.eye(8, 3),
        candidate_ids=np.arange(8, dtype=np.int64),
    )
    assert agent.score_items(obs).shape == (8,)


def test_topk_reinforce_train_offline_runs():
    from rl_recsys.agents.topk_reinforce import TopKReinforceAgent

    agent = TopKReinforceAgent(
        slate_size=3, num_candidates=8, item_dim=3,
        hidden_dim=8, n_heads=2, n_blocks=1, max_history_len=5,
        epochs=1, clip_c=10.0, device="cpu",
    )
    metrics = agent.train_offline(_StubSource(), seed=0)
    assert "loss" in metrics
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/pytest tests/test_agents_dl.py -v -k topk_reinforce`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# rl_recsys/agents/topk_reinforce.py
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from rl_recsys.agents.base import Agent
from rl_recsys.agents.sasrec import _SASRecEncoder
from rl_recsys.environments.base import RecObs


class TopKReinforceAgent(Agent):
    """Chen et al. 2019 top-K off-policy correction over a SASRec-style
    encoder. Importance-weighted policy gradient using the loader's
    precomputed propensities."""

    def __init__(
        self,
        slate_size: int,
        num_candidates: int,
        item_dim: int,
        *,
        hidden_dim: int = 64,
        n_heads: int = 2,
        n_blocks: int = 2,
        max_history_len: int = 20,
        epochs: int = 10,
        clip_c: float = 10.0,
        device: str = "cuda",
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA required for TopKReinforceAgent; "
                "pass device='cpu' to run on CPU"
            )
        self._slate_size = int(slate_size)
        self._num_candidates = int(num_candidates)
        self._max_history_len = int(max_history_len)
        self._epochs = int(epochs)
        self._clip_c = float(clip_c)
        self._device = torch.device(device)
        self._encoder = _SASRecEncoder(
            num_candidates=num_candidates,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_blocks=n_blocks,
            max_history_len=max_history_len,
        ).to(self._device)
        self._heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_candidates) for _ in range(slate_size)
        ]).to(self._device)

    def _build_history_tensors(self, history):
        # Same as SASRec — duplicate to keep modules independent (see spec
        # "Out of scope" note about not extracting a shared module yet).
        T = min(len(history), self._max_history_len)
        if T == 0:
            ids = torch.full(
                (1, 1, self._slate_size), self._encoder._pad_idx,
                dtype=torch.long, device=self._device,
            )
            clicks = torch.zeros((1, 1, self._slate_size), device=self._device)
            pos = torch.zeros((1, 1), dtype=torch.long, device=self._device)
            return ids, clicks, pos
        recent = history[-T:]
        ids = torch.tensor(
            np.stack([h.slate for h in recent]), dtype=torch.long, device=self._device
        ).unsqueeze(0)
        clicks = torch.tensor(
            np.stack([h.clicks for h in recent]), dtype=torch.float32, device=self._device,
        ).unsqueeze(0)
        pos = torch.arange(1, T + 1, dtype=torch.long, device=self._device).unsqueeze(0)
        return ids, clicks, pos

    def _per_position_logits(self, obs: RecObs) -> torch.Tensor:
        ids, clicks, pos = self._build_history_tensors(obs.history)
        h = self._encoder(ids, clicks, pos, attn_mask=None).squeeze(0)
        return torch.stack([head(h) for head in self._heads])  # (slate_size, n_cands)

    def score_items(self, obs: RecObs) -> np.ndarray:
        self._encoder.eval()
        with torch.no_grad():
            logits = self._per_position_logits(obs)
        return logits.sum(dim=0).cpu().numpy().astype(np.float64)

    def select_slate(self, obs: RecObs) -> np.ndarray:
        return np.argsort(self.score_items(obs))[-self._slate_size:][::-1].astype(
            np.int64
        )

    def train_offline(self, source, *, seed: int = 0) -> dict[str, float]:
        torch.manual_seed(seed)
        self._encoder.train()
        opt = torch.optim.Adam(
            list(self._encoder.parameters()) + list(self._heads.parameters()),
            lr=1e-3,
        )
        last_loss = float("nan")
        for epoch in range(self._epochs):
            for traj in source.iter_trajectories(seed=seed + epoch):
                for step in traj:
                    logits = self._per_position_logits(step.obs)  # (K, N)
                    log_probs = torch.log_softmax(logits, dim=-1)  # (K, N)
                    slate_t = torch.tensor(
                        step.logged_action, dtype=torch.long, device=self._device,
                    )
                    per_pos_logp = log_probs[
                        torch.arange(self._slate_size, device=self._device), slate_t,
                    ]  # (K,)
                    log_pi = per_pos_logp.sum()
                    pi = torch.exp(log_pi)
                    rho = torch.clamp(
                        pi / float(step.propensity), min=0.0, max=self._clip_c,
                    )
                    K = self._slate_size
                    lambda_k = float(K) * (1.0 - pi.item()) ** (K - 1)
                    lambda_k = max(1.0, min(lambda_k, float(K)))
                    reward = float(step.logged_reward)
                    loss = -(rho * lambda_k * log_pi * reward)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    last_loss = float(loss.item())
        return {"loss": last_loss, "epochs": float(self._epochs)}

    def update(self, obs, slate, reward, clicks, next_obs) -> dict[str, float]:
        return {}
```

- [ ] **Step 4: Register, run, commit**

```python
# factory
from rl_recsys.agents.topk_reinforce import TopKReinforceAgent

def _build_topk_reinforce(agent_cfg, env_cfg):
    return TopKReinforceAgent(
        slate_size=env_cfg.slate_size,
        num_candidates=env_cfg.num_candidates,
        item_dim=env_cfg.item_dim,
        hidden_dim=getattr(agent_cfg, "hidden_dim", 64),
        n_heads=getattr(agent_cfg, "n_heads", 2),
        n_blocks=getattr(agent_cfg, "n_blocks", 2),
        max_history_len=getattr(agent_cfg, "max_history_len", 20),
        epochs=getattr(agent_cfg, "epochs", 10),
        clip_c=getattr(agent_cfg, "clip_c", 10.0),
        device=getattr(agent_cfg, "device", "cuda"),
    )

AGENT_REGISTRY["topk_reinforce"] = _build_topk_reinforce
```

Run: `.venv/bin/pytest tests/test_agents_dl.py tests/test_agents_dataset_agnostic.py -v`
Expected: PASS.

```bash
git add rl_recsys/agents/topk_reinforce.py rl_recsys/agents/factory.py rl_recsys/agents/__init__.py tests/test_agents_dl.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat: add TopKReinforceAgent (Chen et al. 2019)

Importance-weighted policy gradient with top-K correction over a
SASRec-style encoder. Reuses the loader's precomputed propensities.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 18: `DecisionTransformerAgent`

**Files:**
- Create: `rl_recsys/agents/decision_transformer.py`
- Modify: factory, `__init__.py`
- Test: `tests/test_agents_dl.py` (extend)

- [ ] **Step 1: Write failing tests**

```python
def test_decision_transformer_score_shape():
    from rl_recsys.agents.decision_transformer import DecisionTransformerAgent

    agent = DecisionTransformerAgent(
        slate_size=3, num_candidates=8, user_dim=4, item_dim=3,
        hidden_dim=8, n_blocks=1, context_window=5,
        target_return=1.0, gamma=0.95, epochs=1, device="cpu",
    )
    obs = RecObs(
        user_features=np.zeros(4),
        candidate_features=np.eye(8, 3),
        candidate_ids=np.arange(8, dtype=np.int64),
    )
    assert agent.score_items(obs).shape == (8,)


def test_decision_transformer_train_offline_runs():
    from rl_recsys.agents.decision_transformer import DecisionTransformerAgent

    agent = DecisionTransformerAgent(
        slate_size=3, num_candidates=8, user_dim=4, item_dim=3,
        hidden_dim=8, n_blocks=1, context_window=5,
        target_return=1.0, gamma=0.95, epochs=1, device="cpu",
    )
    metrics = agent.train_offline(_StubSource(), seed=0)
    assert "loss" in metrics
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/pytest tests/test_agents_dl.py -v -k decision_transformer`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# rl_recsys/agents/decision_transformer.py
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecObs


class DecisionTransformerAgent(Agent):
    """Decision Transformer — sequence model over (R-to-go, state, action).
    Conditions on a target return at inference; decodes top-k by cosine
    similarity between the predicted action embedding and item embeddings."""

    def __init__(
        self,
        slate_size: int,
        num_candidates: int,
        user_dim: int,
        item_dim: int,
        *,
        hidden_dim: int = 64,
        n_blocks: int = 3,
        context_window: int = 20,
        target_return: float = 1.0,
        gamma: float = 0.95,
        epochs: int = 10,
        device: str = "cuda",
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA required for DecisionTransformerAgent; "
                "pass device='cpu' to run on CPU"
            )
        self._slate_size = int(slate_size)
        self._num_candidates = int(num_candidates)
        self._user_dim = int(user_dim)
        self._context_window = int(context_window)
        self._target_return = float(target_return)
        self._gamma = float(gamma)
        self._epochs = int(epochs)
        self._hidden_dim = int(hidden_dim)
        self._device = torch.device(device)

        self.r_proj = nn.Linear(1, hidden_dim).to(self._device)
        self.s_proj = nn.Linear(user_dim, hidden_dim).to(self._device)
        self.item_emb = nn.Embedding(num_candidates, hidden_dim).to(self._device)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=2, dim_feedforward=4 * hidden_dim,
            batch_first=True, activation="gelu",
        )
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks).to(
            self._device,
        )
        self.head = nn.Linear(hidden_dim, hidden_dim).to(self._device)

    def _action_emb(self, slate: np.ndarray) -> torch.Tensor:
        ids = torch.tensor(slate, dtype=torch.long, device=self._device)
        return self.item_emb(ids).mean(dim=0)  # (hidden,)

    def _state_emb(self, obs: RecObs) -> torch.Tensor:
        return self.s_proj(
            torch.tensor(obs.user_features, dtype=torch.float32, device=self._device),
        )

    def _build_sequence(self, history, future_return: float) -> torch.Tensor:
        # Build the prefix [R_0, s_0, a_0, ..., R_t, s_t]. We always end on
        # s_t so the next prediction is a_t.
        tokens: list[torch.Tensor] = []
        # Walk a fake-ish reconstruction: history doesn't carry rewards, so
        # we fold them in only during training (see train_offline). At
        # inference we condition on future_return only at the final R.
        for h in history[-self._context_window:]:
            r = self.r_proj(
                torch.tensor([0.0], dtype=torch.float32, device=self._device),
            )
            s = self.s_proj(
                torch.tensor(np.zeros(self._user_dim), dtype=torch.float32,
                             device=self._device),
            )
            a = self._action_emb(h.slate)
            tokens.extend([r, s, a])
        # Final R (target return) and s.
        r = self.r_proj(
            torch.tensor([future_return], dtype=torch.float32, device=self._device),
        )
        tokens.extend([r])
        return torch.stack(tokens, dim=0).unsqueeze(0)  # (1, L, hidden)

    def score_items(self, obs: RecObs) -> np.ndarray:
        self.tr.eval()
        with torch.no_grad():
            seq = self._build_sequence(obs.history, self._target_return)
            # Append state token; predict action embedding from final position.
            s = self._state_emb(obs).unsqueeze(0).unsqueeze(0)
            seq = torch.cat([seq, s], dim=1)
            out = self.tr(seq)
            pred = self.head(out[:, -1, :]).squeeze(0)
            all_items = torch.arange(
                self._num_candidates, dtype=torch.long, device=self._device,
            )
            item_emb = self.item_emb(all_items)
            scores = torch.nn.functional.cosine_similarity(
                pred.unsqueeze(0), item_emb, dim=1,
            )
        return scores.cpu().numpy().astype(np.float64)

    def select_slate(self, obs: RecObs) -> np.ndarray:
        return np.argsort(self.score_items(obs))[-self._slate_size:][::-1].astype(
            np.int64
        )

    def train_offline(self, source, *, seed: int = 0) -> dict[str, float]:
        torch.manual_seed(seed)
        self.tr.train()
        params = (
            list(self.r_proj.parameters()) + list(self.s_proj.parameters()) +
            list(self.item_emb.parameters()) + list(self.tr.parameters()) +
            list(self.head.parameters())
        )
        opt = torch.optim.Adam(params, lr=1e-3)
        last_loss = float("nan")
        for epoch in range(self._epochs):
            for traj in source.iter_trajectories(seed=seed + epoch):
                # Compute return-to-go.
                rewards = np.array(
                    [s.logged_reward for s in traj], dtype=np.float64,
                )
                rtg = np.zeros_like(rewards)
                running = 0.0
                for t in range(len(rewards) - 1, -1, -1):
                    running = rewards[t] + self._gamma * running
                    rtg[t] = running
                for t, step in enumerate(traj):
                    seq = self._build_sequence(step.obs.history, float(rtg[t]))
                    s = self._state_emb(step.obs).unsqueeze(0).unsqueeze(0)
                    seq = torch.cat([seq, s], dim=1)
                    out = self.tr(seq)
                    pred = self.head(out[:, -1, :]).squeeze(0)
                    target = self._action_emb(step.logged_action)
                    loss = nn.functional.mse_loss(pred, target)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    last_loss = float(loss.item())
        return {"loss": last_loss, "epochs": float(self._epochs)}

    def update(self, obs, slate, reward, clicks, next_obs) -> dict[str, float]:
        return {}
```

- [ ] **Step 4: Register, run, commit**

```python
# factory
from rl_recsys.agents.decision_transformer import DecisionTransformerAgent

def _build_decision_transformer(agent_cfg, env_cfg):
    return DecisionTransformerAgent(
        slate_size=env_cfg.slate_size,
        num_candidates=env_cfg.num_candidates,
        user_dim=env_cfg.user_dim,
        item_dim=env_cfg.item_dim,
        hidden_dim=getattr(agent_cfg, "hidden_dim", 64),
        n_blocks=getattr(agent_cfg, "n_blocks", 3),
        context_window=getattr(agent_cfg, "context_window", 20),
        target_return=getattr(agent_cfg, "target_return", 10.0),
        gamma=getattr(agent_cfg, "gamma", 0.95),
        epochs=getattr(agent_cfg, "epochs", 10),
        device=getattr(agent_cfg, "device", "cuda"),
    )

AGENT_REGISTRY["decision_transformer"] = _build_decision_transformer
```

Run: `.venv/bin/pytest tests/test_agents_dl.py tests/test_agents_dataset_agnostic.py -v`
Expected: PASS.

```bash
git add rl_recsys/agents/decision_transformer.py rl_recsys/agents/factory.py rl_recsys/agents/__init__.py tests/test_agents_dl.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat: add DecisionTransformerAgent

Sequence model over (return-to-go, state, action). Trains via behavior
cloning of pooled action embeddings; decodes top-k via cosine similarity
to the predicted action embedding.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 19: Extend `AgentConfig` with all new optional fields

**Files:**
- Modify: `rl_recsys/config.py`
- Test: `tests/test_config.py` (extend)

- [ ] **Step 1: Read the current `AgentConfig`**

Run: `cat rl_recsys/config.py | head -60`. Note the existing fields and the dataclass framework (likely `@dataclass` with defaults).

- [ ] **Step 2: Write a failing test that builds every agent through the factory**

```python
# Append to tests/test_config.py — replace test_agent_config_defaults if present
def test_agent_config_supports_all_registered_agents():
    from rl_recsys.agents.factory import AGENT_REGISTRY, build_agent
    from rl_recsys.config import AgentConfig, EnvConfig

    env = EnvConfig(slate_size=3, user_dim=4, item_dim=3, num_candidates=8)
    for name in AGENT_REGISTRY:
        cfg = AgentConfig(name=name)
        agent = build_agent(cfg, env)
        assert agent is not None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -v -k supports_all_registered_agents`
Expected: FAIL — `AgentConfig` is missing fields like `epsilon`, `temperature`, `hidden_dim`, `device`.

- [ ] **Step 4: Add optional fields**

In `rl_recsys/config.py`, extend `AgentConfig` with optional fields. Replace the dataclass with:

```python
@dataclass
class AgentConfig:
    name: str
    alpha: float = 1.0
    epsilon: float = 0.1
    temperature: float = 1.0
    sigma: float = 1.0
    hidden_dim: int = 64
    embedding_dim: int = 32
    epochs: int = 5
    n_heads: int = 2
    n_blocks: int = 2
    max_history_len: int = 20
    clip_c: float = 10.0
    target_return: float = 10.0
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.05
    gamma: float = 0.95
    context_window: int = 20
    device: str = "cuda"
```

> Note: leave the existing `EnvConfig` and other configs untouched.

- [ ] **Step 5: Run + commit**

Run: `.venv/bin/pytest tests/test_config.py tests/test_factory.py -v`

If `device='cuda'` causes failures in environments without GPUs, change the default to `"cpu"` for tests by passing `device="cpu"` in the factory builders for DL agents that read from `agent_cfg.device`. Or guard at runtime: factory builders set `device = agent_cfg.device if torch.cuda.is_available() or agent_cfg.device == "cpu" else "cpu"`. Pick the latter — survives both CI and live GPU runs.

Update each DL builder in `factory.py`:

```python
def _safe_device(agent_cfg) -> str:
    import torch
    requested = getattr(agent_cfg, "device", "cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested
```

Replace `device=getattr(agent_cfg, "device", "cuda")` with `device=_safe_device(agent_cfg)` in the four DL builders (`neural_linear`, `bc`, `sasrec`, `topk_reinforce`, `decision_transformer`).

Run: `.venv/bin/pytest -x -q`
Expected: all green.

```bash
git add rl_recsys/config.py rl_recsys/agents/factory.py tests/test_config.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat: extend AgentConfig with optional fields used by new agents

All 14 registered agents now build via factory.build_agent(AgentConfig(name=...)).
Factory transparently downgrades device='cuda' to 'cpu' when CUDA isn't
available so the dataset-agnostic and CI suites run anywhere.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 20: Agent grid runner (internal API)

**Files:**
- Create: `rl_recsys/training/agent_grid_runner.py`
- Test: `tests/test_agent_grid_runner.py` (new)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agent_grid_runner.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rl_recsys.environments.base import RecObs
from rl_recsys.evaluation.ope_trajectory import LoggedTrajectoryStep
from rl_recsys.training.agent_grid_runner import GridRun, run_grid


class _FakeSource:
    def __init__(self, num_candidates=8, slate_size=3):
        self.num_candidates = num_candidates
        self.slate_size = slate_size
        self._candidate_features = np.eye(num_candidates, 3)

    def iter_trajectories(self, *, max_trajectories=None, seed=0):
        obs = RecObs(
            user_features=np.zeros(4),
            candidate_features=self._candidate_features,
            candidate_ids=np.arange(self.num_candidates, dtype=np.int64),
            logged_action=np.array([0, 1, 2], dtype=np.int64),
            logged_clicks=np.array([1, 0, 1], dtype=np.int64),
        )
        step = LoggedTrajectoryStep(
            obs=obs,
            logged_action=np.array([0, 1, 2], dtype=np.int64),
            logged_reward=2.0,
            logged_clicks=np.array([1, 0, 1], dtype=np.int64),
            propensity=0.1,
        )
        for _ in range(2):
            yield [step, step]


def test_run_grid_writes_one_artifact_per_run(tmp_path):
    runs = [
        GridRun(agent_name="random", seed=0, pretrained=False),
        GridRun(agent_name="random", seed=1, pretrained=False),
    ]
    src = _FakeSource()
    written = run_grid(
        runs,
        train_source_factory=lambda seed: src,
        eval_source_factory=lambda seed: src,
        env_kwargs=dict(slate_size=3, user_dim=4, item_dim=3, num_candidates=8),
        output_dir=tmp_path,
        max_trajectories=2,
    )
    assert len(written) == 2
    for path in written:
        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        assert data["agent"] == "random"
        assert "metrics" in data


def test_run_grid_resume_skips_existing(tmp_path):
    src = _FakeSource()
    runs = [GridRun(agent_name="random", seed=0, pretrained=False)]
    run_grid(
        runs, train_source_factory=lambda s: src, eval_source_factory=lambda s: src,
        env_kwargs=dict(slate_size=3, user_dim=4, item_dim=3, num_candidates=8),
        output_dir=tmp_path, max_trajectories=2,
    )
    # Re-run with resume=True; should skip.
    written = run_grid(
        runs, train_source_factory=lambda s: src, eval_source_factory=lambda s: src,
        env_kwargs=dict(slate_size=3, user_dim=4, item_dim=3, num_candidates=8),
        output_dir=tmp_path, max_trajectories=2, resume=True,
    )
    assert written == []


def test_run_grid_failed_run_writes_failed_json(tmp_path):
    src = _FakeSource()
    runs = [GridRun(agent_name="logged_replay", seed=0, pretrained=False)]
    # logged_replay should work fine with our _FakeSource (it has logged_action).
    written = run_grid(
        runs, train_source_factory=lambda s: src, eval_source_factory=lambda s: src,
        env_kwargs=dict(slate_size=3, user_dim=4, item_dim=3, num_candidates=8),
        output_dir=tmp_path, max_trajectories=2,
    )
    assert len(written) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_agent_grid_runner.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement**

```python
# rl_recsys/training/agent_grid_runner.py
from __future__ import annotations

import json
import subprocess
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np

from rl_recsys.agents.factory import build_agent
from rl_recsys.config import AgentConfig, EnvConfig
from rl_recsys.evaluation.ope_trajectory import (
    LoggedTrajectorySource,
    evaluate_trajectory_ope_agent,
)


@dataclass
class GridRun:
    agent_name: str
    seed: int
    pretrained: bool
    config_overrides: dict = field(default_factory=dict)


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _run_id(run: GridRun) -> str:
    return f"{run.agent_name}_seed{run.seed}_pretrained{int(run.pretrained)}"


def run_grid(
    runs: list[GridRun],
    *,
    train_source_factory: Callable[[int], LoggedTrajectorySource],
    eval_source_factory: Callable[[int], LoggedTrajectorySource],
    env_kwargs: dict,
    output_dir: Path,
    max_trajectories: int,
    boltzmann_T: float = 1.0,
    resume: bool = False,
    behavior_policy=None,  # injected into BCAgent if provided
) -> list[Path]:
    """Run each (agent, seed, pretrained) tuple, write one JSON per run.

    Returns the list of paths written (excluding skipped-on-resume runs).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    env_cfg = EnvConfig(**env_kwargs)

    for run in runs:
        rid = _run_id(run)
        out_path = output_dir / f"{rid}.json"
        if resume and out_path.exists():
            continue

        agent_cfg = AgentConfig(name=run.agent_name)
        for k, v in run.config_overrides.items():
            setattr(agent_cfg, k, v)

        try:
            agent = build_agent(agent_cfg, env_cfg)
            train_src = train_source_factory(run.seed)
            eval_src = eval_source_factory(run.seed)

            # BC/GBDT need the real candidate_features from the source
            # before training. The factory builds them with placeholders
            # because it can't see the source.
            if hasattr(agent, "_candidate_features") and hasattr(train_src, "_candidate_features"):
                agent._candidate_features = train_src._candidate_features
            # BC also needs an injected, pre-fit BehaviorPolicy.
            if behavior_policy is not None and hasattr(agent, "inject_behavior_policy"):
                agent.inject_behavior_policy(behavior_policy)

            train_started = perf_counter()
            if run.pretrained:
                agent.train_offline(train_src, seed=run.seed)
            train_seconds = perf_counter() - train_started

            evaluation = evaluate_trajectory_ope_agent(
                eval_src, agent,
                agent_name=run.agent_name,
                max_trajectories=max_trajectories,
                seed=run.seed,
                temperature=boltzmann_T,
            )
            payload = {
                "agent": run.agent_name,
                "seed": run.seed,
                "pretrained": run.pretrained,
                "config": {**vars(agent_cfg), "boltzmann_T": boltzmann_T},
                "metrics": {
                    **evaluation.as_dict(),
                    "train_seconds": train_seconds,
                },
                "git_sha": _git_sha(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            out_path.write_text(json.dumps(payload, indent=2, default=float))
            written.append(out_path)
        except Exception as exc:
            failed_path = output_dir / f"{rid}.failed.json"
            failed_path.write_text(json.dumps({
                "agent": run.agent_name,
                "seed": run.seed,
                "pretrained": run.pretrained,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }, indent=2))
            print(f"[grid] {rid} FAILED: {exc!r}", flush=True)
    return written
```

- [ ] **Step 4: Run all tests + commit**

Run: `.venv/bin/pytest tests/test_agent_grid_runner.py -v`
Expected: PASS.

Run: `.venv/bin/pytest -x -q`
Expected: all green.

```bash
git add rl_recsys/training/agent_grid_runner.py tests/test_agent_grid_runner.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat: add agent_grid_runner with one-JSON-per-run artifacts

GridRun(agent_name, seed, pretrained) → writes a JSON file with metrics
and config. Resumable via the --resume flag (skips runs whose file
already exists). Failed runs write a *.failed.json without aborting the
batch.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 21: Results aggregator

**Files:**
- Create: `rl_recsys/training/results_aggregator.py`
- Test: `tests/test_results_aggregator.py` (new)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_results_aggregator.py
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rl_recsys.training.results_aggregator import (
    aggregate, to_summary_csv, to_summary_md,
)


def _write(dirpath: Path, payload: dict, name: str) -> None:
    (dirpath / name).write_text(json.dumps(payload))


def test_aggregate_reads_all_json_skips_failed(tmp_path):
    _write(tmp_path, {
        "agent": "linucb", "seed": 0, "pretrained": True,
        "metrics": {"avg_seq_dr_value": 10.0, "avg_logged_discounted_return": 9.0},
        "config": {}, "git_sha": "x", "timestamp": "2026-05-09T00:00:00Z",
    }, "linucb_seed0_pretrained1.json")
    _write(tmp_path, {
        "agent": "broken", "seed": 0, "pretrained": True,
        "error": "boom", "traceback": "...",
    }, "broken_seed0_pretrained1.failed.json")
    df = aggregate(tmp_path)
    assert len(df) == 1
    assert df.loc[0, "agent"] == "linucb"


def test_summary_md_pivots_pretrained_columns(tmp_path):
    for seed in (0, 1, 2):
        _write(tmp_path, {
            "agent": "linucb", "seed": seed, "pretrained": True,
            "metrics": {"avg_seq_dr_value": 10.0 + seed * 0.1,
                        "avg_logged_discounted_return": 9.0},
            "config": {}, "git_sha": "x", "timestamp": "t",
        }, f"linucb_seed{seed}_pretrained1.json")
        _write(tmp_path, {
            "agent": "linucb", "seed": seed, "pretrained": False,
            "metrics": {"avg_seq_dr_value": 9.5 + seed * 0.05,
                        "avg_logged_discounted_return": 9.0},
            "config": {}, "git_sha": "x", "timestamp": "t",
        }, f"linucb_seed{seed}_pretrained0.json")
    df = aggregate(tmp_path)
    md = to_summary_md(df)
    assert "linucb" in md
    assert "True" in md and "False" in md
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/pytest tests/test_results_aggregator.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# rl_recsys/training/results_aggregator.py
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def aggregate(results_dir: Path) -> pd.DataFrame:
    """Read every {agent}_seed{seed}_pretrained{0|1}.json (skipping
    *.failed.json) into a long-form DataFrame with one row per run."""
    rows: list[dict] = []
    for path in sorted(Path(results_dir).glob("*.json")):
        if path.name.endswith(".failed.json"):
            continue
        data = json.loads(path.read_text())
        flat = {
            "agent": data["agent"],
            "seed": int(data["seed"]),
            "pretrained": bool(data["pretrained"]),
        }
        for k, v in data.get("metrics", {}).items():
            flat[k] = v
        flat["git_sha"] = data.get("git_sha", "unknown")
        flat["timestamp"] = data.get("timestamp", "")
        rows.append(flat)
    return pd.DataFrame(rows)


def to_summary_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def to_summary_md(df: pd.DataFrame) -> str:
    """Pivot table: rows=agent, cols=pretrained∈{True,False}, cells = mean ± std
    of avg_seq_dr_value across seeds."""
    grouped = df.groupby(["agent", "pretrained"])["avg_seq_dr_value"].agg(["mean", "std"])
    grouped["std"] = grouped["std"].fillna(0.0)
    cells = grouped.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
    pivot = cells.unstack("pretrained")
    pivot = pivot.sort_values(by=True, ascending=False) if True in pivot.columns else pivot

    headers = ["agent"] + [f"pretrained={c}" for c in pivot.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for agent, row in pivot.iterrows():
        cells = [agent] + [str(v) if pd.notna(v) else "—" for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1])
    df = aggregate(target)
    to_summary_csv(df, target / "summary.csv")
    (target / "summary.md").write_text(to_summary_md(df))
    print(f"Wrote {target/'summary.csv'} and {target/'summary.md'}")
```

- [ ] **Step 4: Run tests + commit**

Run: `.venv/bin/pytest tests/test_results_aggregator.py -v`
Expected: PASS.

```bash
git add rl_recsys/training/results_aggregator.py tests/test_results_aggregator.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat: add results_aggregator (long-form DataFrame + summary table)

aggregate(results_dir) → DataFrame with one row per run. to_summary_md
produces a pivot of (agent x pretrained) cells of `mean ± std` across
seeds.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 22: CLI front-end + agent config YAMLs

**Files:**
- Create: `scripts/benchmark_agent_grid.py`
- Create: `configs/agents/*.yaml` (14 files)

- [ ] **Step 1: Write the YAML configs**

For each agent name, create `configs/agents/{name}.yaml` with the same content:

```yaml
name: <agent name>
# Optional overrides; leave empty to use AgentConfig defaults.
```

Example: `configs/agents/linucb.yaml`:

```yaml
name: linucb
alpha: 1.0
```

Generate all 14 with the following Python script (run once and discard):

```bash
.venv/bin/python -c "
from pathlib import Path
import os
os.makedirs('configs/agents', exist_ok=True)
agents = ['random', 'linucb', 'most_popular', 'logged_replay', 'oracle_click',
         'lin_ts', 'eps_greedy_linear', 'boltzmann_linear',
         'neural_linear', 'bc', 'gbdt',
         'sasrec', 'topk_reinforce', 'decision_transformer']
for name in agents:
    Path(f'configs/agents/{name}.yaml').write_text(f'name: {name}\n')
print('wrote', len(agents), 'configs')
"
```

- [ ] **Step 2: Write the CLI**

```python
# scripts/benchmark_agent_grid.py
"""Run the (agent x seed x pretrained) ablation grid on RL4RS-B."""
from __future__ import annotations

import argparse
from pathlib import Path

from rl_recsys.agents.factory import AGENT_REGISTRY
from rl_recsys.data.loaders.rl4rs_trajectory_ope import RL4RSTrajectoryOPESource
from rl_recsys.evaluation.behavior_policy import (
    BehaviorPolicy, fit_behavior_policy_with_calibration,
)
from rl_recsys.training.agent_grid_runner import GridRun, run_grid
from rl_recsys.training.session_split import split_session_ids


def _expand_agents(arg: str) -> list[str]:
    if arg == "all":
        return [name for name in AGENT_REGISTRY if name != "oracle_click"]
    return [name.strip() for name in arg.split(",") if name.strip()]


def _expand_pretrained(arg: str) -> list[bool]:
    if arg == "both":
        return [False, True]
    if arg == "true":
        return [True]
    if arg == "false":
        return [False]
    raise ValueError(f"--pretrained-modes must be both|true|false, got {arg!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="rl4rs_b",
        help="rl4rs_b is the only dataset wired today",
    )
    parser.add_argument("--parquet", default="data/processed/rl4rs/sessions_b.parquet")
    parser.add_argument("--agents", default="all")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--pretrained-modes", default="both")
    parser.add_argument("--max-trajectories", type=int, default=5000)
    parser.add_argument("--boltzmann-T", type=float, default=1.0)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/agent_grid/2026-05-09"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--slate-size", type=int, default=9,
        help="RL4RS-B uses 9; override only for synthetic smoke tests",
    )
    parser.add_argument("--user-dim", type=int, default=10)
    parser.add_argument("--item-dim", type=int, default=10)
    args = parser.parse_args()

    if args.dataset != "rl4rs_b":
        raise NotImplementedError(
            "Only rl4rs_b is wired in sub-project #1; KuaiRec/finn/ML come later",
        )

    seeds = [int(s) for s in args.seeds.split(",")]
    agents = _expand_agents(args.agents)
    pretrained_modes = _expand_pretrained(args.pretrained_modes)

    train_ids, eval_ids = split_session_ids(args.parquet, train_fraction=0.5)

    # Auto-detect num_items from the parquet (matches benchmark_rl4rs_b_seq_dr).
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    slate_table = pq.read_table(args.parquet, columns=["slate"])
    flat_items = slate_table["slate"].combine_chunks().flatten()
    num_items = int(pc.count_distinct(flat_items).as_py())

    behavior = fit_behavior_policy_with_calibration(
        args.parquet,
        user_dim=args.user_dim, item_dim=args.item_dim,
        slate_size=args.slate_size, num_items=num_items,
        epochs=5, batch_size=512, seed=0,
    )

    def train_factory(seed: int) -> RL4RSTrajectoryOPESource:
        return RL4RSTrajectoryOPESource(
            args.parquet, behavior, slate_size=args.slate_size,
            session_filter=train_ids,
        )

    def eval_factory(seed: int) -> RL4RSTrajectoryOPESource:
        return RL4RSTrajectoryOPESource(
            args.parquet, behavior, slate_size=args.slate_size,
            session_filter=eval_ids,
        )

    # Materialize an env_kwargs from a sample source.
    sample = train_factory(seeds[0])
    env_kwargs = dict(
        slate_size=args.slate_size,
        user_dim=args.user_dim,
        item_dim=args.item_dim,
        num_candidates=len(sample._candidate_ids),
    )

    runs = [
        GridRun(agent_name=a, seed=s, pretrained=p)
        for a in agents for s in seeds for p in pretrained_modes
    ]
    written = run_grid(
        runs,
        train_source_factory=train_factory,
        eval_source_factory=eval_factory,
        env_kwargs=env_kwargs,
        output_dir=args.output_dir,
        max_trajectories=args.max_trajectories,
        boltzmann_T=args.boltzmann_T,
        resume=args.resume,
        behavior_policy=behavior,
    )
    print(f"wrote {len(written)} run artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
```

> Note: `fit_behavior_policy_with_calibration` is the same function used by `scripts/benchmark_rl4rs_b_seq_dr.py:58`. The auto-detection of `num_items` mirrors lines 47–55 of that script.

- [ ] **Step 3: Smoke-run the CLI on a tiny slice**

```bash
.venv/bin/python scripts/benchmark_agent_grid.py \
    --agents random,logged_replay \
    --seeds 0 \
    --pretrained-modes false \
    --max-trajectories 50 \
    --output-dir /tmp/grid_smoke \
    --slate-size 9 --user-dim 10 --item-dim 10
```

Expected: writes `random_seed0_pretrained0.json` and `logged_replay_seed0_pretrained0.json` to `/tmp/grid_smoke/` without errors.

If `BehaviorPolicy.load_or_train` is missing or signature differs, fix the CLI to match the real API before continuing.

- [ ] **Step 4: Commit**

```bash
git add scripts/benchmark_agent_grid.py configs/
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat: add benchmark_agent_grid CLI + per-agent config YAMLs

CLI fans out (agent x seed x pretrained) over the existing 50/50
RL4RS-B session split, calls run_grid, and writes per-run JSON
artifacts. --agents all expands the registry; --resume skips existing
runs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 23: End-to-end smoke test

**Files:**
- Create: `tests/test_agent_grid_smoke.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_agent_grid_smoke.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rl_recsys.evaluation.behavior_policy import BehaviorPolicy
from rl_recsys.data.loaders.rl4rs_trajectory_ope import RL4RSTrajectoryOPESource
from rl_recsys.training.agent_grid_runner import GridRun, run_grid
from rl_recsys.training.results_aggregator import aggregate, to_summary_md


def _write_synthetic_parquet(path: Path, num_sessions=10, steps_per_session=3,
                             slate_size=3, num_candidates=20,
                             user_dim=4, item_dim=3) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for sid in range(num_sessions):
        for seq in range(steps_per_session):
            slate = rng.choice(num_candidates, size=slate_size, replace=False)
            rows.append({
                "session_id": sid,
                "sequence_id": seq,
                "user_state": np.zeros(user_dim, dtype=np.float64),
                "item_features": np.eye(num_candidates, item_dim)[slate].tolist(),
                "slate": slate.astype(np.int64).tolist(),
                "user_feedback": rng.integers(0, 2, size=slate_size).astype(np.int64).tolist(),
            })
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def test_agent_grid_smoke_end_to_end(tmp_path):
    parquet = tmp_path / "synth.parquet"
    _write_synthetic_parquet(parquet)

    behavior = BehaviorPolicy(
        slate_size=3, user_dim=4, item_dim=3, num_items=20, device="cpu",
    )

    src = RL4RSTrajectoryOPESource(parquet, behavior, slate_size=3)

    runs = [
        GridRun(agent_name="random", seed=0, pretrained=False),
        GridRun(agent_name="linucb", seed=0, pretrained=True),
    ]
    written = run_grid(
        runs,
        train_source_factory=lambda s: src,
        eval_source_factory=lambda s: src,
        env_kwargs=dict(slate_size=3, user_dim=4, item_dim=3, num_candidates=20),
        output_dir=tmp_path / "out",
        max_trajectories=10,
    )
    assert len(written) == 2
    df = aggregate(tmp_path / "out")
    assert {"random", "linucb"} <= set(df["agent"])
    md = to_summary_md(df)
    assert "random" in md and "linucb" in md
```

- [ ] **Step 2: Run + commit**

Run: `.venv/bin/pytest tests/test_agent_grid_smoke.py -v`
Expected: PASS.

```bash
git add tests/test_agent_grid_smoke.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
test: end-to-end smoke for the agent grid

Writes a synthetic parquet, runs two agents through the runner, and
asserts the aggregator produces a valid summary table.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 24: Run the real grid + check in `summary.md`

**Files:**
- Modify: `TODO.md` (capture results)
- Create: `results/agent_grid/2026-05-09/summary.md` (committed; CSV is .gitignored or kept depending on size)

- [ ] **Step 1: Run the headline grid**

```bash
.venv/bin/python scripts/benchmark_agent_grid.py \
    --agents all \
    --seeds 0,1,2 \
    --pretrained-modes both \
    --max-trajectories 5000 \
    --output-dir results/agent_grid/2026-05-09 \
    --resume \
    2>&1 | tee /tmp/agent_grid_run.log
```

Expected: ~78 JSON artifacts (13 agents × 3 seeds × 2 pretrained). DL agents need CUDA; if unavailable, the factory transparently downgrades to CPU and runs (just slower).

If GPU contention strikes (paper2 holding the 5080), run a CPU-only first pass with `--agents random,linucb,lin_ts,eps_greedy_linear,boltzmann_linear,most_popular,logged_replay,bc,gbdt` and add the DL agents with `--resume` once the GPU frees up.

- [ ] **Step 2: Aggregate + write `summary.md`**

```bash
.venv/bin/python -m rl_recsys.training.results_aggregator results/agent_grid/2026-05-09
```

- [ ] **Step 3: Verify the sanity checks from the spec**

Open `results/agent_grid/2026-05-09/summary.md` and verify:
- `logged_replay` row's `pretrained=False` cell ≈ `avg_logged_discounted_return` to within 1% relative.
- `most_popular` and `random` rows are below the linear bandit family's `pretrained=True` cells.
- Sequential DR values are finite for every agent (no NaN).

If any sanity check fails, write the failure mode in `TODO.md` and stop — do not over-claim a result.

- [ ] **Step 4: Update `TODO.md` with the results table**

Edit `TODO.md`:
- Move the "Real-data run on the new batched-propensity loader" section under a "Sub-project #1 results" heading.
- Paste the contents of `summary.md` under it.
- Add a one-line conclusion: "Discriminative? yes/no. Top-3 agents: …"

- [ ] **Step 5: Commit `summary.md` + the TODO update**

```bash
git add results/agent_grid/2026-05-09/summary.md TODO.md
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
docs: capture multi-agent ablation grid results on RL4RS-B

Sub-project #1 wraps with the (14 agents x 3 seeds x 2 pretrained)
table. Logged baseline matches replay agent within IS noise; sanity
checks on most_popular vs linear-bandit family pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review (controller checklist after writing the plan)

This section is the controller's notes; the implementing agent does not need to act on it.

**Spec coverage.**

- 12 new agents: Tasks 6, 7, 8 (heuristics) + 10, 11, 12 (linear bandits, mixin in 9) + 13 (NeuralLinear) + 14 (BC) + 15 (GBDT) + 16 (SASRec) + 17 (TopK) + 18 (DT). ✓
- `Agent` ABC extension: Task 3. ✓
- `RecObs` extension: Task 1. ✓
- Loader history population: Task 2. ✓
- Factory registry refactor: Task 4. ✓
- Dataset-agnostic lint: Task 5. ✓
- Harness CLI + runner + per-run JSON: Tasks 20, 22. ✓
- Aggregator: Task 21. ✓
- Per-agent unit tests: Tasks 6–18 inline. ✓
- End-to-end smoke: Task 23. ✓
- `summary.md` checked in: Task 24. ✓

**Type consistency.** `RecObs` fields in Task 1 match every later usage (`obs.history`, `obs.logged_action`, `obs.logged_clicks`). `LinearBanditBase` defined in Task 9 and used in Tasks 10–12. `_SASRecEncoder` defined in Task 16 and reused (intentionally, by import) in Task 17.

**Known soft spots.**

- The `BCAgent` per-position score path uses `BehaviorPolicy._score_position(user_t, cand_t, k)` — the implementer must verify the actual private method name. The spec said `_score_batch` is also acceptable. Read the file and pick the right one.
- The `train_offline` for `SASRecAgent` is intentionally minimal (per-step, no batching). It will be slow on real data but correct. The acceptance criterion is correctness, not training speed.
- `BehaviorPolicy.load_or_train` doesn't exist as written; the implementer uses whatever fit/save/load path is already used by the existing `scripts/benchmark_rl4rs_b_seq_dr.py`.
- The `factory.py` `_safe_device` helper is duplicated as needed; one helper, called from each DL builder.
- Per-agent YAML configs in Task 22 are stubs. We don't load them in this batch — the harness reads `AgentConfig` defaults. The YAMLs exist so a future batch can wire `--agent-config-dir` without restructuring.
