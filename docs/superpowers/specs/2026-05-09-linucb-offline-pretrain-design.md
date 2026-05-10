# Offline LinUCB Pretraining for RL4RS-B Sequential DR — Design

Date: 2026-05-09

## Motivation

The first end-to-end Sequential DR run on RL4RS-B (commit `f9fe2d2`,
5K trajectories × 3 seeds) produced `avg_seq_dr_value` virtually
identical to the logged baseline for both LinUCB and Random:

| Agent  | avg_seq_dr_value | avg_logged_discounted_return |
|--------|------------------|------------------------------|
| LinUCB | 10.565 ± 0.096   | 10.572 ± 0.096               |
| Random | 10.433 ± 0.079   | 10.572 ± 0.096               |

The pipeline is provably wired (finite, low-variance, logged baseline
matches across agents). All agents look like the behavior policy. Of
the three stacked causes recorded in `TODO.md`, **#2 — LinUCB has zero
history at evaluation time** — masks the other two: a LinUCB constructed
fresh per seed has identity `_a_matrix` and zero `_b_vector`, so every
`score_items` output is identical and the Boltzmann shim collapses to
uniform. Without an offline training pass, we cannot tell whether the
behavior model is too flat or the temperature is too high.

This batch gives LinUCB an offline training pass over logged
trajectories *before* evaluation. Once LinUCB has actual signal, the
remaining two causes (#1 behavior model, #3 temperature) become
measurable.

## Goals

- LinUCB receives a single offline pass over logged trajectories before
  Sequential DR evaluation.
- Train and eval data are disjoint at the session level (50/50 split,
  deterministic), eliminating policy-data circularity.
- The pretraining helper is generic over `Agent`, not LinUCB-specific —
  Random falls through unchanged; future learners plug in for free.
- The benchmark script reports a discriminative `avg_seq_dr_value`
  spread between LinUCB-with-pretrain and Random.

## Non-goals

- Hyperparameter search for LinUCB `alpha` (separate batch).
- Behavior-model improvements (cause #1).
- Boltzmann temperature sweep (cause #3).
- Counterfactual rollout / state evolution.
- Trained DM reward model for DR's variance reduction (separate
  TODO entry).

## Architecture

```
sessions_b.parquet
        │
        ├──► (universe build, full parquet)
        │
        ▼
   RL4RSTrajectoryOPESource(session_filter=train_ids)   ──┐
        │                                                 │
        │   iter_trajectories                             │
        ▼                                                 │
   pretrain_agent_on_logged(LinUCBAgent, train_source) ───┘
        │   walks every step, calls agent.update(...)
        │
        ▼   (LinUCB has populated A, b)
   RL4RSTrajectoryOPESource(session_filter=eval_ids)
        │
        ▼
   evaluate_trajectory_ope_with_variance(make_source=…, …)
        │
        ▼
   TrajectoryOPEEvaluation
```

**Invariants:**
- Behavior_policy instance is shared across train/eval sources →
  propensities under one fixed μ.
- Candidate universe is identical in both sources → `score_items`
  shape and indexing are stable.
- LinUCB updates only on `train_ids`, evaluates only on `eval_ids`
  → no circularity.
- Pretraining is a single pass: LinUCB's A-matrix and b-vector are
  sufficient statistics; multiple passes would over-count.

## Components

### 1. `LoggedTrajectoryStep` — extended

`rl_recsys/evaluation/ope_trajectory.py`:

```python
@dataclass(frozen=True)
class LoggedTrajectoryStep:
    obs: RecObs
    logged_action: np.ndarray   # (slate_size,) candidate indices
    logged_reward: float        # sum(user_feedback) — unchanged
    logged_clicks: np.ndarray   # (slate_size,) per-position binary feedback — NEW
    propensity: float
```

`logged_clicks` carries the per-position feedback that LinUCB's
`update()` expects. Existing OPE consumers (the evaluator,
`seq_dr_value`) ignore the new field.

### 2. `RL4RSTrajectoryOPESource` — gains a session filter

`rl_recsys/data/loaders/rl4rs_trajectory_ope.py`:

```python
def __init__(
    self,
    parquet_path: str | Path,
    behavior_policy: BehaviorPolicy,
    *,
    slate_size: int,
    session_filter: set[int] | None = None,  # NEW
) -> None: ...
```

- When `session_filter is None` (default), behavior is unchanged.
- When set, `iter_trajectories` skips sessions whose `session_id` is
  not in the filter.
- The candidate universe is built from the **full** parquet
  regardless of the filter, so propensities and candidate features
  are comparable between train and eval halves.
- Each emitted step populates
  `logged_clicks = np.array(row["user_feedback"], dtype=np.int64)`.

### 3. `pretrain_agent_on_logged` — new helper

`rl_recsys/training/offline_pretrain.py`:

```python
def pretrain_agent_on_logged(
    agent: Agent,
    source: LoggedTrajectorySource,
    *,
    max_trajectories: int | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Single pass over source trajectories, calling agent.update() per step.

    For each LoggedTrajectoryStep, calls
        agent.update(
            obs=step.obs,
            slate=step.logged_action,
            reward=step.logged_reward,
            clicks=step.logged_clicks,
            next_obs=step.obs,
        )

    next_obs == obs because LinUCB ignores it (contextual bandit) and
    Random ignores everything; we don't fabricate state evolution.

    Returns aggregate metrics: trajectories, total_steps,
    mean_click_rate, seconds.
    """
```

- Generic over `Agent`. Random's no-op `update` works unchanged.
- Does NOT swallow exceptions from `agent.update()` — shape errors
  propagate.
- Raises `ValueError` if `source` yields zero trajectories (mirrors
  the evaluator).

### 4. `split_session_ids` — small helper

Co-located with the benchmark or in a tiny util module. Signature:

```python
def split_session_ids(
    parquet_path: Path,
    *,
    train_fraction: float = 0.5,
    seed: int = 42,
) -> tuple[set[int], set[int]]:
    """Deterministic session-level partition.

    For each session_id, hash((seed, sid)) % 1000 < int(1000 * train_fraction)
    decides train vs eval. Stable, no extra dependency.
    """
```

- Raises `ValueError` if `train_fraction` not in `(0, 1)`.
- Raises `ValueError` if either side is empty (e.g., trivially small
  parquet).

### 5. Benchmark script — wired up

`scripts/benchmark_rl4rs_b_seq_dr.py`:

After fitting `BehaviorPolicy`:

```python
train_ids, eval_ids = split_session_ids(parquet, train_fraction=0.5, seed=42)

def make_train_source(): return RL4RSTrajectoryOPESource(
    parquet_path=parquet, behavior_policy=model,
    slate_size=SLATE_SIZE, session_filter=train_ids,
)
def make_eval_source(): return RL4RSTrajectoryOPESource(
    parquet_path=parquet, behavior_policy=model,
    slate_size=SLATE_SIZE, session_filter=eval_ids,
)

def make_linucb():
    agent = LinUCBAgent(
        slate_size=SLATE_SIZE, user_dim=USER_DIM,
        item_dim=ITEM_DIM, alpha=1.0,
    )
    pretrain_agent_on_logged(agent, make_train_source())
    return agent

linucb_result = evaluate_trajectory_ope_with_variance(
    make_source=make_eval_source,
    make_agent=make_linucb,
    agent_name="linucb",
    max_trajectories=5000, n_seeds=3, base_seed=42,
    gamma=0.95, temperature=1.0,
)

random_result = evaluate_trajectory_ope_with_variance(
    make_source=make_eval_source,
    make_agent=lambda: RandomAgent(slate_size=SLATE_SIZE, seed=0),
    agent_name="random",
    max_trajectories=5000, n_seeds=3, base_seed=42,
    gamma=0.95, temperature=1.0,
)
```

Random is unchanged (no pretrain). Each LinUCB seed gets its own
pretrained instance — pretraining is fast on the train half (~0.5 ×
781K rows ≈ 0.4M slate updates).

## Error handling

- `pretrain_agent_on_logged` raises `ValueError` on zero
  trajectories from `source` and propagates `agent.update()`
  exceptions verbatim.
- `RL4RSTrajectoryOPESource(session_filter=…)` with a filter that
  excludes every session in the parquet is allowed at construction
  (universe still builds), but `iter_trajectories` raises
  `ValueError` on first iteration so pretrain fails loud rather
  than silently no-oping.
- `split_session_ids` raises `ValueError` if `train_fraction` is
  outside `(0, 1)` or if either side ends up empty.

## Testing

New tests under `.venv/bin/python -m pytest`:

### `tests/test_offline_pretrain.py`
- `test_pretrain_calls_update_per_step`: spy agent records every
  `update()` call; assert (a) call count = total steps, (b) each
  call's `clicks` matches the parquet row, (c) `next_obs is obs`.
- `test_pretrain_changes_linucb_state`: pretrain a LinUCB on a
  synthetic 2-session fixture, assert `_a_matrix` ≠ identity and
  `_b_vector` ≠ zero afterwards.
- `test_pretrain_random_is_noop_safe`: Random-style agent runs
  without error and returns finite metrics.
- `test_pretrain_raises_on_empty_source`: filter excludes all
  sessions, expect `ValueError` from `iter_trajectories`.

### `tests/test_rl4rs_trajectory_ope.py` — extended
- `test_loader_emits_logged_clicks`: assert `step.logged_clicks`
  matches `row["user_feedback"]` element-wise.
- `test_loader_session_filter`: with `session_filter={1}`, only
  session-1 trajectories are yielded; universe still includes all
  items from the full parquet.

### `tests/test_split_session_ids.py`
- `test_split_is_deterministic`: same seed → same partition.
- `test_split_is_disjoint_and_complete`: train ∩ eval = ∅,
  train ∪ eval = all session ids.
- `test_split_fraction_approximate`: at 0.5 over 1000 ids, train
  size within ±5%.
- `test_split_rejects_invalid_fractions`: 0.0, 1.0, -0.1, 1.5 all
  raise.
- `test_split_rejects_empty_split`: tiny parquet that hashes all
  to one side raises.

### End-to-end smoke
Extend `test_end_to_end_seq_dr_on_synthetic_b_fixture` (or add a
sibling): pretrain LinUCB on train half, evaluate on eval half,
assert resulting `avg_seq_dr_value` differs from a fresh-LinUCB
run on the same eval half. Proves pretrain has effect.

### Existing tests
All existing tests must keep passing. `LoggedTrajectoryStep` is
constructed only in the loader and a small number of unit-test
fixtures; those fixtures gain a `logged_clicks=np.zeros(slate_size,
dtype=np.int64)` filler.

## Out of scope (intentionally)

- **Hyperparameter search.** Single `alpha=1.0`; tuning is a
  separate batch.
- **Behavior-model improvements.** Cause #1 in TODO.md.
- **Temperature sweep.** Cause #3.
- **State evolution / next_obs semantics.** Pretrain passes
  `next_obs = obs`. A future Q-learner would need real next-step
  obs.
- **Reward model for DR.** `dr_value(reward_model=None)` continues
  to fall back to `mean(rewards)`.

## Acceptance

- All new + existing tests pass.
- Benchmark runs end-to-end on `data/processed/rl4rs/sessions_b.parquet`.
- Reported `avg_seq_dr_value` for LinUCB diverges from
  `avg_logged_discounted_return` by more than the ±std band — i.e.,
  the estimator becomes discriminative. (If it still doesn't, the
  remaining masked causes #1 / #3 are confirmed and we capture them
  in TODO.md as the next batch.)
