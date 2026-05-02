# Dataset-Backed RL Environments Design

**Date:** 2026-05-02
**Status:** Draft

## Goal

Add three `RecEnv` subclasses backed by real logged datasets — KuaiRec, FINN.no Slate, and RL4RS — so that RL agents can train and evaluate on realistic interaction data rather than the synthetic latent-factor simulator alone.

## Architecture

Two thin abstract base classes are introduced under `rl_recsys/environments/`, both extending the existing `RecEnv`:

```
RecEnv (existing)
├── BanditDatasetEnv      ← single-step: sample interaction, build candidate pool, one step, done
│   ├── KuaiRecEnv        ← reward = watch_ratio (0–1)
│   └── FinnNoSlateEnv    ← reward = per-position binary clicks; pool = actual logged slate
└── SessionDatasetEnv     ← multi-step: sample session, step through, done at session end
    └── RL4RSEnv          ← reward = click per step; state from pre-computed RL4RS vectors
```

**New files:**

| File | Contents |
|---|---|
| `rl_recsys/environments/dataset_base.py` | `BanditDatasetEnv`, `SessionDatasetEnv` |
| `rl_recsys/environments/kuairec.py` | `KuaiRecEnv` |
| `rl_recsys/environments/finn_no_slate.py` | `FinnNoSlateEnv` |
| `rl_recsys/environments/rl4rs.py` | `RL4RSEnv` |

**Modified files:**

| File | Change |
|---|---|
| `rl_recsys/data/pipelines/kuairec.py` | Emit `item_features.parquet` alongside `interactions.parquet` |
| `rl_recsys/data/pipelines/rl4rs.py` | Replace placeholder with proper `sessions.parquet` output |

The existing `trainer.py` single-step loop works as-is for `BanditDatasetEnv` subclasses. For `RL4RSEnv` the trainer will need a `while not done` inner loop — that is a trainer concern deferred to sub-project 2 (agents).

## Tech Stack

Python 3.10+, NumPy, Pandas, PyArrow. No new dependencies.

---

## Base Classes

### `BanditDatasetEnv(RecEnv)` — `dataset_base.py`

**Constructor:**
```python
BanditDatasetEnv(
    interactions: pd.DataFrame,
    *,
    slate_size: int = 1,
    num_candidates: int = 50,
    feature_dim: int = 16,
    feature_source: str = "native",  # "native" | "hashed"
    seed: int = 0,
)
```

`interactions` must contain `user_id` and `item_id` columns. Subclasses may require additional columns (e.g. `rating`).

**Episode structure:**
- `reset()`: sample one row uniformly, treat `row["item_id"]` as the positive item, sample `num_candidates - 1` negatives from the global item pool (excluding known positives for that user), shuffle into `candidate_ids`, return `RecObs`
- `step(slate)`: check which slate positions contain the positive item, call `_compute_reward(row, clicks) -> float`, return `RecStep(done=True)`

**Feature source:**
- `"hashed"`: uses `_hashed_vector(prefix, entity_id, dim)` — same approach as `LoggedInteractionEnv`
- `"native"`: expects subclass to populate `self._native_user_features: dict[int, np.ndarray]` and `self._native_item_features: dict[int, np.ndarray]` before calling `super().__init__`; raises `ValueError` if these are not populated

**Abstract method:**
```python
@abstractmethod
def _compute_reward(self, row: pd.Series, clicks: np.ndarray) -> float: ...
```

**Validation at init:**
- `num_candidates > len(unique items)` → `ValueError`
- `slate_size > num_candidates` → `ValueError`
- `feature_dim < 3` → `ValueError`
- empty DataFrame → `ValueError`

---

### `SessionDatasetEnv(RecEnv)` — `dataset_base.py`

**Constructor:**
```python
SessionDatasetEnv(
    sessions: dict[int, pd.DataFrame],
    *,
    slate_size: int = 1,
    num_candidates: int = 50,
    feature_dim: int = 16,
    feature_source: str = "native",
    seed: int = 0,
)
```

`sessions` is a mapping from session ID to a DataFrame of that session's steps, ordered by step index. Each row must contain `slate` (item ID list), `clicks` (binary list), and optionally `user_state` (pre-computed vector) for native features.

**Episode structure:**
- `reset()`: pick a session uniformly, set cursor to step 0, return `RecObs` for step 0
- `step(slate)`: compute reward for current step via `_compute_reward`, advance cursor, return `RecStep(done=(cursor == len(session)))`

**Feature source:** same contract as `BanditDatasetEnv`.

**Abstract method:**
```python
@abstractmethod
def _compute_reward(self, row: pd.Series, clicks: np.ndarray) -> float: ...
```

---

## Concrete Environments

### `KuaiRecEnv(BanditDatasetEnv)` — `kuairec.py`

**Constructor:**
```python
KuaiRecEnv(
    processed_dir: str | Path,
    *,
    slate_size: int = 1,
    num_candidates: int = 50,
    feature_dim: int = 16,
    feature_source: str = "native",
    seed: int = 0,
)
```

Loads `{processed_dir}/interactions.parquet` (columns: `user_id`, `item_id`, `rating`, `timestamp`). For `feature_source="native"`, also loads `{processed_dir}/item_features.parquet` (item category/tag columns produced by the updated KuaiRec pipeline) and populates `self._native_item_features`.

**Reward:**
```python
def _compute_reward(self, row, clicks) -> float:
    return float(row["rating"]) * float(clicks[0])
```

Watch-ratio (0–1) is returned only when the agent clicks the positive item.

**KuaiRec pipeline update (`kuairec.py`):**

The existing pipeline emits only `interactions.parquet`. It must also load `KuaiRec 2.0/data/item_categories.csv` and write `item_features.parquet` with columns `item_id` plus one column per category/tag. `KuaiRecEnv` loads this file for native features. If the file is absent and `feature_source="native"` is requested, `KuaiRecEnv` raises `FileNotFoundError` with a message instructing the user to rerun `--process`.

---

### `FinnNoSlateEnv(BanditDatasetEnv)` — `finn_no_slate.py`

**Constructor:**
```python
FinnNoSlateEnv(
    processed_dir: str | Path,
    *,
    slate_size: int = 25,
    feature_dim: int = 16,
    feature_source: str = "hashed",
    seed: int = 0,
)
```

Loads `{processed_dir}/slates.parquet` (columns: `request_id`, `user_id`, `slate` list, `clicks` index, `timestamp`). `num_candidates` is fixed at 25 (the logged slate length) and cannot be overridden.

**Candidate pool:** On `reset()` the candidate pool is always the 25 items from the logged `slate` column, not sampled negatives. This preserves the real impression distribution. The positive item is derived as `row["slate"][row["clicks"]]` — the item at the logged click index — which is then passed to `BanditDatasetEnv`'s standard click-detection logic.

**Reward:**
```python
def _compute_reward(self, row, clicks) -> float:
    return float(clicks.sum())
```

**Feature source:** FINN.no has no item metadata. If `feature_source="native"` is requested, `FinnNoSlateEnv` logs a warning and falls back to `"hashed"` automatically.

---

### `RL4RSEnv(SessionDatasetEnv)` — `rl4rs.py`

**Constructor:**
```python
RL4RSEnv(
    processed_dir: str | Path,
    *,
    slate_size: int = 6,
    feature_dim: int = 32,
    feature_source: str = "native",
    seed: int = 0,
)
```

Loads `{processed_dir}/sessions.parquet` and groups by `session_id` to build the `sessions` dict passed to `SessionDatasetEnv`.

**Expected `sessions.parquet` schema** (produced by the updated RL4RS pipeline):

| Column | Type | Description |
|---|---|---|
| `session_id` | int64 | Groups rows into episodes |
| `step` | int64 | 0-indexed position within session |
| `user_state` | list[float] | Pre-computed user context vector |
| `slate` | list[int] | Item IDs shown at this step |
| `item_features` | list[list[float]] | Feature matrix for slate items |
| `clicks` | list[int] | Binary click per slate position |

**RL4RS pipeline update (`rl4rs.py`):**

Replace the placeholder `process()` with proper parsing of `rl4rs_dataset_a_rl.csv`, grouping rows by session, serialising `user_state`, `slate`, `item_features`, and `clicks` as list columns in `sessions.parquet`.

**Feature source:**
- `"native"`: uses `row["user_state"]` as `user_features` and `row["item_features"]` as the candidate feature matrix directly — no hashing. `feature_dim` must match the vector length in the data; raises `ValueError` on mismatch.
- `"hashed"`: ignores pre-computed vectors, uses hashed entity features instead.

**Reward:**
```python
def _compute_reward(self, row, clicks) -> float:
    return float(clicks.sum())
```

---

## Testing

| File | What it tests |
|---|---|
| `tests/test_dataset_base.py` | `BanditDatasetEnv` and `SessionDatasetEnv`: feature_source switching, invalid `num_candidates`, empty DataFrame rejection, hashed feature shape |
| `tests/test_env_kuairec.py` | `KuaiRecEnv` with 5 users × 20 items in-memory; `reset()` shape, `step()` returns `done=True`, reward ∈ [0,1], both `feature_source` values produce correct dims |
| `tests/test_env_finn_no_slate.py` | `FinnNoSlateEnv` with tiny slates DataFrame; `candidate_ids == logged slate`, reward is binary, `done=True` after one step, native→hashed fallback warning |
| `tests/test_env_rl4rs.py` | `RL4RSEnv` with 2 sessions × 3 steps each; `done=False` mid-session, `done=True` at end, reward accumulates, native feature_dim mismatch raises `ValueError` |

All tests use in-memory DataFrames — no disk I/O, no downloads.

---

## Out of Scope

- Trainer changes for multi-step episodes (sub-project 2: agents)
- New agent implementations (sub-project 2)
- Benchmark runner harness (sub-project 3)
- Environments for other datasets (MovieLens, Open Bandit as full `RecEnv`, etc.)
