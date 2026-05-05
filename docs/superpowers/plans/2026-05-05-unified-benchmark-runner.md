# Unified Benchmark Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `run_synthetic.py` and `run_dataset_bandit.py` with a single Hydra-based `experiments/run.py` that benchmarks Random and LinUCB agents across all five env types (synthetic, KuaiRec, FinnNoSlate, RL4RS, logged) with MLflow tracking and Hydra multirun for ablation sweeps.

**Architecture:** Extend `trainer.py`'s single-step loop to `while not done` (backward-compatible with bandit envs). Add `rl_recsys/environments/factory.py` that builds the right `RecEnv` subclass from a Hydra `DictConfig`. Add Hydra YAML configs for each new env type. Wire it all together in `experiments/run.py`.

**Tech stack:** Python 3.12, Hydra 1.3, OmegaConf, MLflow (SQLite backend), pandas/parquet.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `rl_recsys/training/trainer.py` | Modify | Replace single-step loop with `while not done` |
| `tests/test_trainer.py` | Create | Verify single-step and multi-step episode accumulation |
| `rl_recsys/environments/factory.py` | Create | `build_env(cfg)` → correct `RecEnv` subclass |
| `tests/test_factory.py` | Create | Factory builds each env type; raises on unknown/missing data |
| `conf/env/synthetic.yaml` | Modify | Add `type: synthetic` |
| `conf/env/kuairec.yaml` | Create | KuaiRec env config |
| `conf/env/finn_no_slate.yaml` | Create | FinnNoSlate env config |
| `conf/env/rl4rs.yaml` | Create | RL4RS env config |
| `conf/env/logged.yaml` | Create | LoggedInteractionEnv config (MovieLens etc.) |
| `conf/mlflow/local.yaml` | Create | MLflow enabled, SQLite backend |
| `experiments/run.py` | Create | Unified Hydra entry point |
| `experiments/run_synthetic.py` | Modify | Add deprecation comment |
| `experiments/run_dataset_bandit.py` | Modify | Add deprecation comment |

---

## Task 1: Extend trainer to multi-step loop

**Files:**
- Modify: `rl_recsys/training/trainer.py`
- Create: `tests/test_trainer.py`

The trainer's inner loop currently does exactly one `select_slate` + `step` per episode. Session envs (`KuaiRecEnv`, `RL4RSEnv`) need multiple steps per episode. We wrap in `while not done`. Single-step bandit envs return `done=True` on the first step, so they are unaffected.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trainer.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from rl_recsys.agents.random import RandomAgent
from rl_recsys.config import (
    AgentConfig, EnvConfig, ExperimentConfig,
    MlflowConfig, RuntimeConfig, TrainConfig, WandbConfig,
)
from rl_recsys.environments.base import RecEnv, RecObs, RecStep
from rl_recsys.training.trainer import train


def _minimal_cfg(tmp_path) -> ExperimentConfig:
    base = str(tmp_path)
    return ExperimentConfig(
        env=EnvConfig(),
        agent=AgentConfig(),
        train=TrainConfig(num_episodes=1, log_every=1, seed=0),
        wandb=WandbConfig(enabled=False),
        mlflow=MlflowConfig(enabled=False),
        runtime=RuntimeConfig(
            repo_root=base,
            workspace_root=base,
            results_root=base,
            project_results_dir=base,
            run_dir=base,
            hydra_dir=base,
            wandb_dir=base,
            tb_dir=base,
            mlflow_dir=base,
            logs_dir=base,
            checkpoints_dir=base,
            exports_dir=base,
            mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
            project_manifest_path=str(tmp_path / "project.yaml"),
            run_manifest_path=str(tmp_path / "run.yaml"),
        ),
    )


def _obs() -> RecObs:
    return RecObs(
        user_features=np.zeros(4, dtype=np.float32),
        candidate_features=np.zeros((3, 4), dtype=np.float32),
        candidate_ids=np.arange(3),
    )


class _ThreeStepEnv(RecEnv):
    """Session env that requires 3 steps per episode, reward=1.0 each step."""

    def __init__(self):
        self._steps = 0

    @property
    def slate_size(self) -> int: return 2
    @property
    def num_candidates(self) -> int: return 3
    @property
    def user_dim(self) -> int: return 4
    @property
    def item_dim(self) -> int: return 4

    def reset(self, seed: int | None = None) -> RecObs:
        self._steps = 0
        return _obs()

    def step(self, slate: np.ndarray) -> RecStep:
        self._steps += 1
        return RecStep(obs=_obs(), reward=1.0, clicks=np.array([1, 0]), done=self._steps >= 3)


def test_trainer_single_step_env_still_works(tmp_path):
    """Bandit env (done=True on first step) works after loop change."""
    class _BanditEnv(_ThreeStepEnv):
        def step(self, slate):
            return RecStep(obs=_obs(), reward=2.0, clicks=np.array([1, 0]), done=True)

    history = train(_BanditEnv(), RandomAgent(slate_size=2, seed=0), _minimal_cfg(tmp_path))
    assert history[0]["reward"] == pytest.approx(2.0)


def test_trainer_multi_step_accumulates_all_rewards(tmp_path):
    """Session env: reward must sum across 3 steps."""
    history = train(_ThreeStepEnv(), RandomAgent(slate_size=2, seed=0), _minimal_cfg(tmp_path))
    assert history[0]["reward"] == pytest.approx(3.0)


def test_trainer_returns_one_entry_per_episode(tmp_path):
    """history length equals num_episodes regardless of steps per episode."""
    cfg = _minimal_cfg(tmp_path)
    cfg.train.num_episodes = 4
    history = train(_ThreeStepEnv(), RandomAgent(slate_size=2, seed=0), cfg)
    assert len(history) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/home/vishsangale/workspace/rl-recsys/.venv/bin/python -m pytest tests/test_trainer.py -v
```

Expected: `test_trainer_multi_step_accumulates_all_rewards` and `test_trainer_returns_one_entry_per_episode` FAIL because the current loop only runs one step, collecting reward=1.0 instead of 3.0.

- [ ] **Step 3: Replace the single-step loop in `rl_recsys/training/trainer.py`**

Find this block (around line 30 inside the episode loop):

```python
        # single-step episodes (re-rank a fresh candidate set each episode)
        slate = agent.select_slate(obs)
        step = env.step(slate)

        agent_metrics = agent.update(obs, slate, step.reward, step.clicks, step.obs)

        episode_rewards.append(step.reward)
        episode_clicks.append(step.clicks)
```

Replace with:

```python
        agent_metrics: dict[str, float] = {}
        done = False
        while not done:
            slate = agent.select_slate(obs)
            step = env.step(slate)
            agent_metrics = agent.update(obs, slate, step.reward, step.clicks, step.obs)
            episode_rewards.append(step.reward)
            episode_clicks.append(step.clicks)
            obs = step.obs
            done = step.done
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/home/vishsangale/workspace/rl-recsys/.venv/bin/python -m pytest tests/test_trainer.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Run full suite to confirm no regressions**

```bash
/home/vishsangale/workspace/rl-recsys/.venv/bin/python -m pytest tests/ -q --tb=short
```

Expected: all 124 tests + 3 new = 127 pass.

- [ ] **Step 6: Commit**

```bash
git add rl_recsys/training/trainer.py tests/test_trainer.py
git commit -m "feat: extend trainer to multi-step while-not-done episode loop"
```

---

## Task 2: Env factory

**Files:**
- Create: `rl_recsys/environments/factory.py`
- Create: `tests/test_factory.py`

A single `build_env(cfg: DictConfig) -> RecEnv` function that reads `cfg.type` and instantiates the correct env class. Dataset envs (`kuairec`, `finn_no_slate`, `rl4rs`) take `processed_dir` and load parquet files internally. The `logged` type needs an extra `filename` param to locate the parquet. `synthetic` reconstructs `EnvConfig` from DictConfig fields.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_factory.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from rl_recsys.environments.factory import build_env
from rl_recsys.environments.finn_no_slate import FinnNoSlateEnv
from rl_recsys.environments.kuairec import KuaiRecEnv
from rl_recsys.environments.logged import LoggedInteractionEnv
from rl_recsys.environments.rl4rs import RL4RSEnv
from rl_recsys.environments.synthetic import SyntheticEnv


# ── fixtures ──────────────────────────────────────────────────────────────────

def _make_kuairec(tmp_path, n_rows: int = 30, n_items: int = 60, n_feats: int = 4) -> None:
    rng = np.random.default_rng(0)
    pd.DataFrame({
        "user_id": rng.integers(0, 5, n_rows),
        "item_id": rng.integers(0, n_items, n_rows),
        "rating": rng.uniform(0, 1, n_rows).astype(np.float32),
        "timestamp": np.arange(n_rows),
    }).to_parquet(tmp_path / "interactions.parquet", index=False)
    feat_cols = {f"feat_{i}": rng.standard_normal(n_items).astype(np.float32) for i in range(n_feats)}
    pd.DataFrame({"item_id": np.arange(n_items), **feat_cols}).to_parquet(
        tmp_path / "item_features.parquet", index=False
    )


def _make_finn(tmp_path, n_rows: int = 30) -> None:
    rng = np.random.default_rng(0)
    rows = [
        {
            "request_id": i,
            "user_id": i % 5,
            "slate": list(range(i * 25, i * 25 + 25)),
            "clicks": int(rng.integers(0, 25)),
            "timestamp": 1_600_000_000 + i * 1000,
        }
        for i in range(n_rows)
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "slates.parquet", index=False)


def _make_rl4rs(tmp_path, n_sessions: int = 3, n_steps: int = 3,
                slate_size: int = 3, n_feats: int = 4) -> None:
    rng = np.random.default_rng(0)
    rows = [
        {
            "session_id": sid,
            "step": step,
            "user_state": rng.standard_normal(n_feats).tolist(),
            "slate": rng.integers(0, 100, size=slate_size).tolist(),
            "item_features": rng.standard_normal((slate_size, n_feats)).tolist(),
            "clicks": rng.integers(0, 2, size=slate_size).tolist(),
        }
        for sid in range(n_sessions)
        for step in range(n_steps)
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "sessions.parquet", index=False)


def _make_logged(tmp_path, n_rows: int = 100) -> None:
    rng = np.random.default_rng(0)
    pd.DataFrame({
        "user_id": rng.integers(0, 20, n_rows),
        "item_id": rng.integers(0, 200, n_rows),
        "rating": rng.uniform(1, 5, n_rows).astype(np.float32),
        "timestamp": np.arange(n_rows),
    }).to_parquet(tmp_path / "interactions.parquet", index=False)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_build_synthetic():
    cfg = OmegaConf.create({
        "type": "synthetic", "num_items": 100, "num_candidates": 10,
        "slate_size": 3, "user_dim": 8, "item_dim": 8, "position_bias_decay": 0.5,
    })
    env = build_env(cfg)
    assert isinstance(env, SyntheticEnv)
    assert env.slate_size == 3
    assert env.num_candidates == 10


def test_build_kuairec(tmp_path):
    _make_kuairec(tmp_path)
    cfg = OmegaConf.create({
        "type": "kuairec", "processed_dir": str(tmp_path),
        "slate_size": 3, "num_candidates": 10, "feature_dim": 4,
        "feature_source": "native", "seed": 0,
    })
    env = build_env(cfg)
    assert isinstance(env, KuaiRecEnv)
    assert env.slate_size == 3


def test_build_finn_no_slate(tmp_path):
    _make_finn(tmp_path)
    cfg = OmegaConf.create({
        "type": "finn_no_slate", "processed_dir": str(tmp_path),
        "slate_size": 5, "feature_dim": 8, "feature_source": "hashed", "seed": 0,
    })
    env = build_env(cfg)
    assert isinstance(env, FinnNoSlateEnv)
    assert env.slate_size == 5


def test_build_rl4rs(tmp_path):
    _make_rl4rs(tmp_path)
    cfg = OmegaConf.create({
        "type": "rl4rs", "processed_dir": str(tmp_path),
        "slate_size": 3, "feature_dim": 4, "feature_source": "native", "seed": 0,
    })
    env = build_env(cfg)
    assert isinstance(env, RL4RSEnv)
    assert env.slate_size == 3


def test_build_logged(tmp_path):
    _make_logged(tmp_path)
    cfg = OmegaConf.create({
        "type": "logged", "processed_dir": str(tmp_path),
        "filename": "interactions.parquet",
        "slate_size": 5, "num_candidates": 20, "feature_dim": 8,
        "rating_threshold": 3.0, "seed": 0,
    })
    env = build_env(cfg)
    assert isinstance(env, LoggedInteractionEnv)
    assert env.slate_size == 5


def test_unknown_type_raises():
    cfg = OmegaConf.create({"type": "not_a_real_env"})
    with pytest.raises(ValueError, match="Unknown env type"):
        build_env(cfg)


def test_logged_missing_file_raises(tmp_path):
    cfg = OmegaConf.create({
        "type": "logged", "processed_dir": str(tmp_path),
        "filename": "missing.parquet",
        "slate_size": 5, "num_candidates": 20, "feature_dim": 8,
        "rating_threshold": 3.0, "seed": 0,
    })
    with pytest.raises(FileNotFoundError):
        build_env(cfg)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/home/vishsangale/workspace/rl-recsys/.venv/bin/python -m pytest tests/test_factory.py -v
```

Expected: ImportError — `rl_recsys.environments.factory` does not exist yet.

- [ ] **Step 3: Create `rl_recsys/environments/factory.py`**

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from rl_recsys.config import EnvConfig
from rl_recsys.environments.base import RecEnv
from rl_recsys.environments.finn_no_slate import FinnNoSlateEnv
from rl_recsys.environments.kuairec import KuaiRecEnv
from rl_recsys.environments.logged import LoggedInteractionEnv
from rl_recsys.environments.rl4rs import RL4RSEnv
from rl_recsys.environments.synthetic import SyntheticEnv


def build_env(cfg: DictConfig) -> RecEnv:
    env_type = str(cfg.get("type", "synthetic"))

    if env_type == "synthetic":
        return SyntheticEnv(EnvConfig(
            num_items=int(cfg.get("num_items", 1000)),
            num_candidates=int(cfg.get("num_candidates", 50)),
            slate_size=int(cfg.get("slate_size", 10)),
            user_dim=int(cfg.get("user_dim", 32)),
            item_dim=int(cfg.get("item_dim", 32)),
            position_bias_decay=float(cfg.get("position_bias_decay", 0.5)),
        ))

    if env_type == "kuairec":
        return KuaiRecEnv(
            processed_dir=cfg.processed_dir,
            slate_size=int(cfg.get("slate_size", 6)),
            num_candidates=int(cfg.get("num_candidates", 50)),
            feature_dim=int(cfg.get("feature_dim", 32)),
            feature_source=str(cfg.get("feature_source", "native")),
            seed=int(cfg.get("seed", 0)),
        )

    if env_type == "finn_no_slate":
        return FinnNoSlateEnv(
            processed_dir=cfg.processed_dir,
            slate_size=int(cfg.get("slate_size", 5)),
            feature_dim=int(cfg.get("feature_dim", 16)),
            feature_source=str(cfg.get("feature_source", "hashed")),
            seed=int(cfg.get("seed", 0)),
        )

    if env_type == "rl4rs":
        return RL4RSEnv(
            processed_dir=cfg.processed_dir,
            slate_size=int(cfg.get("slate_size", 6)),
            feature_dim=int(cfg.get("feature_dim", 32)),
            feature_source=str(cfg.get("feature_source", "native")),
            seed=int(cfg.get("seed", 0)),
        )

    if env_type == "logged":
        parquet_path = Path(str(cfg.processed_dir)) / str(cfg.filename)
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Processed data not found: {parquet_path}. "
                "Run scripts/prepare_data.py first."
            )
        df = pd.read_parquet(parquet_path)
        return LoggedInteractionEnv(
            df,
            slate_size=int(cfg.get("slate_size", 10)),
            num_candidates=int(cfg.get("num_candidates", 50)),
            feature_dim=int(cfg.get("feature_dim", 16)),
            rating_threshold=float(cfg.get("rating_threshold", 4.0)),
            seed=int(cfg.get("seed", 0)),
        )

    raise ValueError(
        f"Unknown env type {env_type!r}. "
        "Known: synthetic, kuairec, finn_no_slate, rl4rs, logged"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/home/vishsangale/workspace/rl-recsys/.venv/bin/python -m pytest tests/test_factory.py -v
```

Expected: 7 tests PASS.

- [ ] **Step 5: Run full suite**

```bash
/home/vishsangale/workspace/rl-recsys/.venv/bin/python -m pytest tests/ -q --tb=short
```

Expected: all 127 + 7 new = 134 pass.

- [ ] **Step 6: Commit**

```bash
git add rl_recsys/environments/factory.py tests/test_factory.py
git commit -m "feat: add env factory (build_env) for all five env types"
```

---

## Task 3: Hydra env configs and MLflow local config

**Files:**
- Modify: `conf/env/synthetic.yaml`
- Create: `conf/env/kuairec.yaml`
- Create: `conf/env/finn_no_slate.yaml`
- Create: `conf/env/rl4rs.yaml`
- Create: `conf/env/logged.yaml`
- Create: `conf/mlflow/local.yaml`

No tests needed — YAML correctness is validated by the factory tests (Task 2) and the smoke run (Task 4).

- [ ] **Step 1: Add `type: synthetic` to `conf/env/synthetic.yaml`**

Open `conf/env/synthetic.yaml`. It currently contains:
```yaml
num_items: 1000
num_candidates: 50
slate_size: 10
user_dim: 32
item_dim: 32
position_bias_decay: 0.5
```

Add `type: synthetic` as the first line:
```yaml
type: synthetic
num_items: 1000
num_candidates: 50
slate_size: 10
user_dim: 32
item_dim: 32
position_bias_decay: 0.5
```

- [ ] **Step 2: Create `conf/env/kuairec.yaml`**

```yaml
type: kuairec
processed_dir: data/processed/kuairec
slate_size: 6
num_candidates: 50
feature_dim: 32
feature_source: native
seed: 42
```

- [ ] **Step 3: Create `conf/env/finn_no_slate.yaml`**

```yaml
type: finn_no_slate
processed_dir: data/processed/finn-no-slate
slate_size: 5
feature_dim: 16
feature_source: hashed
seed: 42
```

- [ ] **Step 4: Create `conf/env/rl4rs.yaml`**

```yaml
type: rl4rs
processed_dir: data/processed/rl4rs
slate_size: 6
feature_dim: 32
feature_source: native
seed: 42
```

- [ ] **Step 5: Create `conf/env/logged.yaml`**

```yaml
type: logged
processed_dir: data/processed/movielens
filename: ratings_100k.parquet
slate_size: 10
num_candidates: 50
feature_dim: 16
rating_threshold: 4.0
seed: 42
```

- [ ] **Step 6: Create `conf/mlflow/local.yaml`**

```yaml
enabled: true
tracking_uri: sqlite:///${runtime.project_results_dir}/mlflow/mlflow.db
experiment_name: rl-recsys-benchmark
run_name: ${runtime.workspace_run_id}
artifact_path: ${runtime.workspace_run_id}
tags: {}
```

- [ ] **Step 7: Commit**

```bash
git add conf/env/synthetic.yaml conf/env/kuairec.yaml conf/env/finn_no_slate.yaml \
        conf/env/rl4rs.yaml conf/env/logged.yaml conf/mlflow/local.yaml
git commit -m "feat: add Hydra env configs and MLflow local config"
```

---

## Task 4: Unified run.py and deprecation notices

**Files:**
- Create: `experiments/run.py`
- Modify: `experiments/run_synthetic.py`
- Modify: `experiments/run_dataset_bandit.py`

`run.py` mirrors `run_synthetic.py`'s Hydra boilerplate but uses `build_env` instead of hardcoding `SyntheticEnv`, and derives agent dimensions from the live env's properties instead of `EnvConfig`.

- [ ] **Step 1: Create `experiments/run.py`**

```python
"""Unified benchmark runner — works with all env types and agents.

Replaces experiments/run_synthetic.py and experiments/run_dataset_bandit.py.

Single run (MLflow disabled by default):
    python experiments/run.py env=synthetic agent=linucb

Single run with MLflow tracking:
    python experiments/run.py mlflow=local env=kuairec agent=linucb

Full benchmark matrix — 8 runs:
    python experiments/run.py --multirun \\
        env=synthetic,kuairec,finn_no_slate,rl4rs \\
        agent=random,linucb

View results:
    mlflow ui --backend-store-uri sqlite:///results/rl-recsys/mlflow/mlflow.db
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from rl_recsys.agents import build_agent
from rl_recsys.config import AgentConfig, EnvConfig, to_experiment_config
from rl_recsys.environments.factory import build_env
from rl_recsys.training.trainer import train


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
OmegaConf.register_new_resolver("workspace_root", lambda: str(WORKSPACE_ROOT), replace=True)
OmegaConf.register_new_resolver("workspace_run_id", lambda: WORKSPACE_RUN_ID, replace=True)


@hydra.main(version_base="1.3", config_path="../conf", config_name="train")
def main(cfg: DictConfig) -> None:
    env = build_env(cfg.env)

    exp_cfg = to_experiment_config(cfg)
    env_cfg = EnvConfig(
        slate_size=env.slate_size,
        num_candidates=env.num_candidates,
        user_dim=env.user_dim,
        item_dim=env.item_dim,
    )
    agent = build_agent(exp_cfg.agent, env_cfg)

    env_type = str(cfg.env.get("type", "synthetic"))
    print(f"\nenv={env_type}  agent={exp_cfg.agent.name}  episodes={exp_cfg.train.num_episodes}")

    history = train(env, agent, exp_cfg)
    avg_reward = float(np.mean([h["reward"] for h in history]))
    print(f"Done. avg_reward={avg_reward:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test run.py with synthetic env (no MLflow)**

```bash
cd /home/vishsangale/workspace/rl-recsys && \
  .venv/bin/python experiments/run.py \
    env=synthetic agent=random \
    train.num_episodes=5 train.log_every=1
```

Expected output: lines like `[ep    0] reward=...  ndcg=...  ctr=...` and a final `Done. avg_reward=...`. No errors.

- [ ] **Step 3: Smoke-test multirun with synthetic env**

```bash
cd /home/vishsangale/workspace/rl-recsys && \
  .venv/bin/python experiments/run.py --multirun \
    env=synthetic agent=random,linucb \
    train.num_episodes=3 train.log_every=1
```

Expected: 2 runs complete without error. Hydra prints a multirun summary.

- [ ] **Step 4: Add deprecation comment to `experiments/run_synthetic.py`**

Add these two lines at the very top of the file, before the docstring:

```python
# DEPRECATED: Use experiments/run.py instead.
# Equivalent: python experiments/run.py env=synthetic agent=<agent>
```

- [ ] **Step 5: Add deprecation comment to `experiments/run_dataset_bandit.py`**

Add these two lines at the very top of the file, before the docstring:

```python
# DEPRECATED: Use experiments/run.py instead.
# Equivalent: python experiments/run.py env=kuairec agent=<agent>
```

- [ ] **Step 6: Run full test suite one final time**

```bash
/home/vishsangale/workspace/rl-recsys/.venv/bin/python -m pytest tests/ -q --tb=short
```

Expected: 134 tests pass, 0 failures.

- [ ] **Step 7: Commit**

```bash
git add experiments/run.py experiments/run_synthetic.py experiments/run_dataset_bandit.py
git commit -m "feat: add unified run.py Hydra entry point; deprecate old runners"
```
