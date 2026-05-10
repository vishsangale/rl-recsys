# Batched Propensity Precompute in `RL4RSTrajectoryOPESource` — Design

Date: 2026-05-09

## Motivation

The pretrain batch (commits `3f8afae..7f7a6ff`) wired offline LinUCB
training and a session-level train/eval split. The real-data run on
the RTX 5080 stalled in the LinUCB pretrain phase: after one hour, no
`pretrain:` line had printed.

Root cause: `RL4RSTrajectoryOPESource.iter_trajectories` calls
`self._policy.slate_propensity(user_features, candidate_features,
slate)` per step. Each call runs `slate_size` (=9) sequential
single-sample forward passes through `BehaviorPolicy._mlp`. With
~219K train sessions × ~1.78 mean steps × 3 seeds ≈ 1.2M sequential
calls × 9 positions = 10.5M tiny CUDA forwards — launch-overhead
dominated even though the GPU sat at 94% util.

`BehaviorPolicy._score_batch` already exists and supports batched
forward over `(B, num_candidates)`. The fit loop and held-out NLL
calculation use it. The per-call `slate_propensity` does not. We
batch the propensity computation.

## Goals

- Precompute every logged step's propensity in one batched GPU pass
  per loader instance.
- Reduce wall-clock for the LinUCB pretrain phase from hours to
  minutes on the RTX 5080.
- Keep the `RL4RSTrajectoryOPESource` constructor signature
  unchanged so the benchmark script and prior tests keep working.
- Preserve numerical equivalence with the per-call path
  (`np.allclose` to 1e-12).

## Non-goals

- Caching the propensity table across loader instances. The
  benchmark constructs train + eval sources × 3 seeds = 6 instances;
  each computes its filtered subset independently. Roughly equal
  total work; cleaner separation than introducing a shared cache.
- Auto-tuning the batch size. Reuse the fit loop's `batch_size=512`
  default, with no constructor knob.
- Replacing the existing `BehaviorPolicy.slate_propensity()` method.
  It stays for ad-hoc / per-row use.

## Architecture

```
__init__:
    parquet → df → universe + features (unchanged)
              ↓
              sort by (session_id, sequence_id) → ordered
              ↓
              apply session_filter → filtered_ordered (reset_index)
              ↓
              build users (B, U), slates (B, K) tensors
              ↓
              behavior_policy.slate_log_propensities_batch(...)
                    │
                    └─ chunked GPU forwards, returns (B,) log-probs
              ↓
              self._propensities = exp(log_probs)

iter_trajectories(seed):
    walk ordered rows, look up self._propensities[row_idx]
    yield LoggedTrajectoryStep(..., propensity=self._propensities[i])
```

**Invariants:**
- `self._candidate_features` is built from the full parquet
  (unchanged from prior batch). The precompute uses this same
  array, so candidate indexing is identical between train and eval
  sources sharing one `BehaviorPolicy` instance.
- `_propensities` is a 1-D array, length = `len(self._ordered)`.
  `iter_trajectories` looks up by row position in `self._ordered`,
  not by `(session_id, sequence_id)`.
- `slate_size` for slates passed to the batched method must match
  `self._policy._slate_size`; mismatched shapes raise.

## Components

### 1. `BehaviorPolicy.slate_log_propensities_batch` — new method

`rl_recsys/evaluation/behavior_policy.py`:

```python
def slate_log_propensities_batch(
    self,
    users: np.ndarray,             # (B, user_dim)
    slates: np.ndarray,            # (B, slate_size) — candidate indices
    candidate_features: np.ndarray,  # (num_candidates, item_dim)
    *,
    batch_size: int = 512,
) -> np.ndarray:                   # (B,) log-probs
    """log Π_k softmax(score(·, k))[slate[k]] for each (user, slate)."""
```

Implementation:
- Validate `users.shape[0] == slates.shape[0]`,
  `slates.shape[1] == self._slate_size`.
- Move `candidate_features` to device once.
- For each chunk of `batch_size` rows:
  - Build `users_chunk` tensor `(b, user_dim)`.
  - For each `k in range(slate_size)`:
    - Build a `(b,)` `position` tensor with all values `= k`.
    - Expand `cand_t_shared` to `(b, num_candidates, item_dim)`.
    - `logits = self._score_batch(users_chunk, cands, positions)`
      → `(b, num_candidates)`.
    - `log_probs = log_softmax(logits, dim=-1)`.
    - `gathered = log_probs.gather(1, slates_chunk[:, k:k+1]).squeeze(1)`.
    - Accumulate into `chunk_log_total`.
  - Append `chunk_log_total.cpu().numpy()` to results.
- Return concatenated numpy array.

Wrapped in `torch.no_grad()`. Returns log-probs (numerically stable
for the loader's `exp` to recover propensity).

### 2. `RL4RSTrajectoryOPESource` — precompute at __init__

`rl_recsys/data/loaders/rl4rs_trajectory_ope.py`:

After universe build and `session_filter` storage:

```python
ordered = self._df.sort_values(
    ["session_id", "sequence_id"], kind="stable"
)
if self._session_filter is not None:
    ordered = ordered[ordered["session_id"].isin(self._session_filter)]
self._ordered = ordered.reset_index(drop=True)

if len(self._ordered) > 0:
    users = np.stack([
        np.array(list(u), dtype=np.float64)
        for u in self._ordered["user_state"]
    ])
    slates = np.stack([
        np.array(
            [self._cand_id_to_idx[int(x)] for x in s], dtype=np.int64
        )
        for s in self._ordered["slate"]
    ])
    started = perf_counter()
    log_props = self._policy.slate_log_propensities_batch(
        users, slates, self._candidate_features,
    )
    self._propensities = np.exp(log_props).astype(np.float64)
    elapsed = perf_counter() - started
    print(
        f"propensity precompute: {len(self._propensities)} rows "
        f"in {elapsed:.1f}s",
        flush=True,
    )
    if (self._propensities <= 0).any():
        raise ValueError("zero propensity in logged slate")
else:
    self._propensities = np.zeros(0, dtype=np.float64)
```

`iter_trajectories` is rewritten to walk `self._ordered` (already
filtered + sorted), group by `session_id`. After `reset_index(drop=True)`,
each group's `.index` values are positions in `self._ordered` and
therefore valid lookups into `self._propensities`. Per row at
position `i`:

- `user_features = np.array(list(self._ordered.at[i, "user_state"]), …)`
- `logged_slate_ids = np.array(list(self._ordered.at[i, "slate"]), …)`
- `slate_indices = …` (existing translation)
- `logged_clicks = np.array(list(self._ordered.at[i, "user_feedback"]), …)`
- `logged_reward = float(np.sum(self._ordered.at[i, "user_feedback"]))`
- `propensity = float(self._propensities[i])`
- `obs = RecObs(...)` (unchanged)
- emit `LoggedTrajectoryStep(...)`.

The empty-filter raise stays in `iter_trajectories` and triggers when
`len(self._ordered) == 0` after filtering.

The `seed` argument to `iter_trajectories` continues to control
session-order shuffling. Iteration order over groupby groups is
keyed off the (potentially shuffled) `session_ids` list, exactly as
today.

## Error handling

- `slate_log_propensities_batch` raises `ValueError` for
  `users.shape[0] != slates.shape[0]` and
  `slates.shape[1] != self._slate_size`. Returns finite log-probs
  for any finite logits (no zero-prob early-raise inside this
  method).
- Loader raises `ValueError("zero propensity in logged slate")`
  immediately after the precompute, with the same message as the
  previous per-call check. Surfaced at construction rather than
  mid-iteration.
- Loader raises `ValueError("session_filter excludes every session
  in the parquet — no trajectories to emit")` from
  `iter_trajectories` on first call when `len(self._ordered) == 0`.

## Testing

### `tests/test_behavior_policy.py` — extend
- `test_slate_log_propensities_batch_matches_per_call`: 4-row
  synthetic fixture, compute via batched method, compare
  element-wise to `[model.slate_propensity(...) for each row]`.
  `np.allclose(np.exp(batched), per_call, atol=1e-12)`.
- `test_slate_log_propensities_batch_chunking`: same fixture, run
  with `batch_size=2` (forces 2 chunks); assert identical to the
  single-chunk result.
- `test_slate_log_propensities_batch_validates_shapes`: mismatched
  `users.shape[0]` vs `slates.shape[0]` raises; `slates.shape[1] !=
  slate_size` raises.

### `tests/test_rl4rs_trajectory_ope.py` — extend
- New `test_loader_propensity_matches_legacy_per_call`: small
  parquet, two loaders sharing one BehaviorPolicy. One uses the new
  batched precompute path (current loader). The other (a small
  test-only helper) iterates rows and calls
  `model.slate_propensity(user, cand, slate)`. Assert step
  propensities match within 1e-12.
- Existing `test_loader_emits_trajectories_grouped_by_session` and
  `test_loader_emits_logged_clicks` regression — must keep passing
  with the new precompute path.

### End-to-end smoke
Existing `test_pretrained_linucb_diverges_from_fresh_linucb` keeps
passing. The precompute is numerically equivalent so LinUCB scores
won't shift; the > 1e-6 divergence threshold has substantial
headroom.

### Performance verification
After implementation, run the benchmark on the real parquet:
```
.venv/bin/python scripts/benchmark_rl4rs_b_seq_dr.py
```
Expect each loader's `propensity precompute:` line to appear within
~1–3 minutes (vs. unbounded under the current path). Capture the
numbers in `TODO.md`.

## Acceptance

- All tests pass (existing 193 + 4 new = 197).
- `propensity precompute:` lines visible in benchmark output, each
  under ~5 minutes on the RTX 5080.
- LinUCB-with-pretrain `avg_seq_dr_value` reportable in `TODO.md`
  for the discriminative-benchmark question.

## Out of scope

- Cross-source propensity caching (each loader recomputes
  independently for now).
- Universe-build caching across loaders (already noted as a
  separate TODO entry from the prior batch's final review).
- Eliminating the dead `import numpy as np` in
  `scripts/benchmark_rl4rs_b_seq_dr.py` (pre-existing).
