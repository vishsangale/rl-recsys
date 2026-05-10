# Batched Propensity Precompute Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `RL4RSTrajectoryOPESource`'s per-step `slate_propensity` calls with a single batched precompute at __init__, so the LinUCB pretrain pass on the real RL4RS-B parquet finishes in minutes instead of hours.

**Architecture:** Add `BehaviorPolicy.slate_log_propensities_batch(users, slates, candidate_features, batch_size=512)` that runs `slate_size` chunked GPU forwards instead of per-row sequential calls. Loader builds `(B, user_dim)` and `(B, slate_size)` tensors at construction, calls the new method, caches a row-position-indexed `np.ndarray` of propensities, and looks them up in `iter_trajectories`. No public API changes to the loader.

**Tech Stack:** PyTorch, numpy, pandas, pyarrow. All commands use `.venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-05-09-batched-propensity-design.md`

---

## File map

- Modify: `rl_recsys/evaluation/behavior_policy.py` — add `slate_log_propensities_batch`
- Modify: `rl_recsys/data/loaders/rl4rs_trajectory_ope.py` — precompute propensities at __init__, look up in `iter_trajectories`
- Modify: `tests/test_behavior_policy.py` — three new tests for the batched method
- Modify: `tests/test_rl4rs_trajectory_ope.py` — one new parity test
- Modify: `TODO.md` — record real-data run outcome (Task 3)

---

## Task 1: Add `slate_log_propensities_batch` to `BehaviorPolicy`

**Files:**
- Modify: `rl_recsys/evaluation/behavior_policy.py` — add new method on the `BehaviorPolicy` class
- Modify: `tests/test_behavior_policy.py` — append three new tests

The new method computes, for each `(user, slate)` pair in a batch, `log Π_k softmax(score(·, k))[slate[k]]`. Uses the existing `_score_batch` to do `slate_size` chunked GPU forwards, gathers at each slate position, accumulates log-probs.

- [ ] **Step 1: Append three failing tests to `tests/test_behavior_policy.py`**

```python
def test_slate_log_propensities_batch_matches_per_call() -> None:
    """Batched log-propensities match per-row slate_propensity calls."""
    rng = np.random.default_rng(0)
    user_dim, item_dim, slate_size, num_items = 3, 4, 2, 5
    model = BehaviorPolicy(
        user_dim=user_dim, item_dim=item_dim, slate_size=slate_size,
        num_items=num_items, hidden_dim=8, seed=0, device="cpu",
    )

    n = 6
    users = rng.standard_normal((n, user_dim)).astype(np.float64)
    candidate_features = rng.standard_normal((num_items, item_dim)).astype(np.float64)
    # Each row picks 2 distinct candidates.
    slates = np.stack([rng.choice(num_items, size=slate_size, replace=False)
                       for _ in range(n)]).astype(np.int64)

    log_props = model.slate_log_propensities_batch(
        users, slates, candidate_features,
    )
    per_call = np.array([
        model.slate_propensity(users[i], candidate_features, slates[i])
        for i in range(n)
    ])

    assert log_props.shape == (n,)
    np.testing.assert_allclose(np.exp(log_props), per_call, atol=1e-12)


def test_slate_log_propensities_batch_chunking_is_consistent() -> None:
    """Same inputs, different batch_size → identical output."""
    rng = np.random.default_rng(1)
    model = BehaviorPolicy(
        user_dim=2, item_dim=2, slate_size=2, num_items=4,
        hidden_dim=4, seed=0, device="cpu",
    )
    n = 7
    users = rng.standard_normal((n, 2)).astype(np.float64)
    cand = rng.standard_normal((4, 2)).astype(np.float64)
    slates = np.stack([rng.choice(4, size=2, replace=False)
                       for _ in range(n)]).astype(np.int64)

    a = model.slate_log_propensities_batch(users, slates, cand, batch_size=2)
    b = model.slate_log_propensities_batch(users, slates, cand, batch_size=64)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_slate_log_propensities_batch_validates_shapes() -> None:
    """Mismatched B and bad slate width raise ValueError."""
    model = BehaviorPolicy(
        user_dim=2, item_dim=2, slate_size=2, num_items=3,
        hidden_dim=4, seed=0, device="cpu",
    )
    cand = np.zeros((3, 2), dtype=np.float64)

    # users.shape[0] != slates.shape[0]
    with pytest.raises(ValueError, match="batch size mismatch"):
        model.slate_log_propensities_batch(
            np.zeros((4, 2), dtype=np.float64),
            np.zeros((5, 2), dtype=np.int64),
            cand,
        )

    # slates.shape[1] != self._slate_size
    with pytest.raises(ValueError, match="slate width"):
        model.slate_log_propensities_batch(
            np.zeros((3, 2), dtype=np.float64),
            np.zeros((3, 5), dtype=np.int64),
            cand,
        )
```

- [ ] **Step 2: Run the new tests to confirm they fail**

```
.venv/bin/python -m pytest tests/test_behavior_policy.py::test_slate_log_propensities_batch_matches_per_call tests/test_behavior_policy.py::test_slate_log_propensities_batch_chunking_is_consistent tests/test_behavior_policy.py::test_slate_log_propensities_batch_validates_shapes -v
```

Expected: all 3 FAIL with `AttributeError: 'BehaviorPolicy' object has no attribute 'slate_log_propensities_batch'`.

- [ ] **Step 3: Add the method to `BehaviorPolicy`**

Edit `rl_recsys/evaluation/behavior_policy.py`. Insert this method on the `BehaviorPolicy` class, after `slate_propensity` (around the existing line that ends with `return result`):

```python
def slate_log_propensities_batch(
    self,
    users: np.ndarray,
    slates: np.ndarray,
    candidate_features: np.ndarray,
    *,
    batch_size: int = 512,
) -> np.ndarray:
    """log Π_k softmax(score(·, k))[slate[k]] for each (user, slate) pair.

    Returns numpy array of shape (B,). Iterates in chunks of `batch_size`;
    each chunk runs `slate_size` calls to `_score_batch` (one per position),
    log-softmax, and gathers at the logged candidate index.

    Raises ValueError if `users.shape[0] != slates.shape[0]` or
    `slates.shape[1] != self._slate_size`.
    """
    if users.shape[0] != slates.shape[0]:
        raise ValueError(
            f"batch size mismatch: users={users.shape[0]} slates={slates.shape[0]}"
        )
    if slates.shape[1] != self._slate_size:
        raise ValueError(
            f"slate width {slates.shape[1]} != self._slate_size {self._slate_size}"
        )

    n = users.shape[0]
    cand_t = torch.as_tensor(
        candidate_features, dtype=torch.float64, device=self._device,
    )
    out = np.empty(n, dtype=np.float64)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            b = end - start

            users_chunk = torch.as_tensor(
                users[start:end], dtype=torch.float64, device=self._device,
            )
            slates_chunk = torch.as_tensor(
                slates[start:end], dtype=torch.long, device=self._device,
            )
            cands_b = cand_t.unsqueeze(0).expand(b, -1, -1)

            log_total = torch.zeros(b, dtype=torch.float64, device=self._device)
            for k in range(self._slate_size):
                positions = torch.full(
                    (b,), k, dtype=torch.long, device=self._device,
                )
                logits = self._score_batch(users_chunk, cands_b, positions)
                log_probs = torch.log_softmax(logits, dim=-1)
                gathered = log_probs.gather(
                    1, slates_chunk[:, k:k + 1],
                ).squeeze(1)
                log_total = log_total + gathered

            out[start:end] = log_total.cpu().numpy()

    return out
```

- [ ] **Step 4: Run the three new tests; expect PASS**

```
.venv/bin/python -m pytest tests/test_behavior_policy.py::test_slate_log_propensities_batch_matches_per_call tests/test_behavior_policy.py::test_slate_log_propensities_batch_chunking_is_consistent tests/test_behavior_policy.py::test_slate_log_propensities_batch_validates_shapes -v
```

Expected: 3/3 PASS.

- [ ] **Step 5: Run the full suite; expect 196 passed (was 193 + 3 new)**

```
.venv/bin/python -m pytest -q
```

Expected: 196 passed.

- [ ] **Step 6: Commit via HEREDOC**

```bash
git add rl_recsys/evaluation/behavior_policy.py tests/test_behavior_policy.py
git commit -m "$(cat <<'EOF'
feat: add BehaviorPolicy.slate_log_propensities_batch

Batched (users, slates) -> log-propensity helper that runs slate_size
chunked GPU forwards via _score_batch instead of per-row sequential
calls. Returns log-probs (caller exps to recover propensity).

Matches per-call slate_propensity within 1e-12 across batch sizes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Precompute propensities in `RL4RSTrajectoryOPESource.__init__`

**Files:**
- Modify: `rl_recsys/data/loaders/rl4rs_trajectory_ope.py`
- Modify: `tests/test_rl4rs_trajectory_ope.py` — add one parity test

Replace per-step `slate_propensity` calls with a single batched precompute at construction. Cache as `self._propensities: np.ndarray` indexed by row position in the filtered, sorted DataFrame. `iter_trajectories` looks up by row position.

- [ ] **Step 1: Append a failing parity test to `tests/test_rl4rs_trajectory_ope.py`**

```python
def test_loader_propensity_matches_per_call(tmp_path: Path) -> None:
    """Loader's batched propensities match per-row slate_propensity calls."""
    from rl_recsys.data.loaders.rl4rs_trajectory_ope import (
        RL4RSTrajectoryOPESource,
    )
    parquet = _fixture_b_parquet(tmp_path)

    model = BehaviorPolicy(
        user_dim=2, item_dim=2, slate_size=2, num_items=3,
        hidden_dim=4, seed=0, device="cpu",
    )
    source = RL4RSTrajectoryOPESource(
        parquet_path=parquet, behavior_policy=model, slate_size=2,
    )
    trajectories = list(source.iter_trajectories(max_trajectories=10, seed=0))

    # Recompute each step's propensity via the per-call path on the same
    # candidate universe and assert match.
    for traj in trajectories:
        for step in traj:
            expected = model.slate_propensity(
                step.obs.user_features,
                source._candidate_features,
                step.logged_action,
            )
            assert step.propensity == pytest.approx(expected, rel=1e-12, abs=1e-12)
```

The test imports `pytest` — make sure the file already imports it (it does for prior tests; if not, add `import pytest` at the top).

- [ ] **Step 2: Run the new test to confirm it currently passes (sanity baseline)**

```
.venv/bin/python -m pytest tests/test_rl4rs_trajectory_ope.py::test_loader_propensity_matches_per_call -v
```

Expected: PASS, because the legacy per-call path produces these exact propensities. After Step 3 the test continues to pass against the batched precompute (it's a parity guard on the refactor).

- [ ] **Step 3: Refactor `RL4RSTrajectoryOPESource.__init__`**

Edit `rl_recsys/data/loaders/rl4rs_trajectory_ope.py`. Add the `perf_counter` import at the top:

```python
from time import perf_counter
```

Replace the existing `__init__` body, keeping the universe build and `session_filter` logic unchanged. After they execute, materialize the filtered + sorted DataFrame and run the precompute. The new `__init__` body should look like:

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

    # Build the candidate universe from the slate column. Using
    # pyarrow.compute.unique on the flat list-array is ~7x faster than a
    # Python set for large parquets, and mirrors FinnNoSlateTrajectoryLoader.
    slate_table = pq.read_table(parquet_path, columns=["slate"])
    flat_items = slate_table["slate"].combine_chunks().flatten()
    universe = np.sort(pc.unique(flat_items).to_numpy()).astype(np.int64)

    # Build (item_id → features) by scanning slate+item_features once.
    feature_for: dict[int, list[float]] = {}
    for slate, item_feats in zip(self._df["slate"], self._df["item_features"]):
        for item_id, feat in zip(slate, item_feats):
            if int(item_id) not in feature_for:
                feature_for[int(item_id)] = list(feat)
        if len(feature_for) == len(universe):
            break

    self._candidate_ids: np.ndarray = universe
    self._candidate_features: np.ndarray = np.array(
        [feature_for[int(i)] for i in universe], dtype=np.float64
    )
    self._cand_id_to_idx: dict[int, int] = {
        int(cid): k for k, cid in enumerate(universe)
    }

    # Sort + filter once. Row-positions in self._ordered are the indices
    # into self._propensities.
    ordered = self._df.sort_values(
        ["session_id", "sequence_id"], kind="stable",
    )
    if self._session_filter is not None:
        ordered = ordered[ordered["session_id"].isin(self._session_filter)]
    self._ordered = ordered.reset_index(drop=True)

    # Precompute propensities for every row in self._ordered in one batched
    # GPU pass. Empty filter → zero-length _propensities; iter_trajectories
    # raises on iteration as before.
    if len(self._ordered) > 0:
        users = np.stack([
            np.array(list(u), dtype=np.float64)
            for u in self._ordered["user_state"]
        ])
        slate_indices = np.stack([
            np.array(
                [self._cand_id_to_idx[int(x)] for x in s], dtype=np.int64,
            )
            for s in self._ordered["slate"]
        ])
        started = perf_counter()
        log_props = self._policy.slate_log_propensities_batch(
            users, slate_indices, self._candidate_features,
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

- [ ] **Step 4: Refactor `iter_trajectories` to look up cached propensities**

In the same file, replace the body of `iter_trajectories`. The session shuffle and empty-filter raise stay the same; the per-row loop now uses `group.index` to look up `self._propensities`:

```python
def iter_trajectories(
    self, *, max_trajectories: int | None = None, seed: int | None = None,
) -> Iterator[list[LoggedTrajectoryStep]]:
    groups = self._ordered.groupby("session_id", sort=False)
    session_ids = list(groups.groups.keys())
    rng = np.random.default_rng(0 if seed is None else seed)
    if seed is not None:
        rng.shuffle(session_ids)

    if not session_ids:
        raise ValueError(
            "session_filter excludes every session in the parquet — "
            "no trajectories to emit"
        )

    emitted = 0
    for sid in session_ids:
        if max_trajectories is not None and emitted >= max_trajectories:
            break
        group = groups.get_group(sid)
        steps: list[LoggedTrajectoryStep] = []
        for row_pos, row in zip(group.index, group.itertuples(index=False)):
            user_features = np.array(list(row.user_state), dtype=np.float64)
            logged_slate_ids = np.array(list(row.slate), dtype=np.int64)
            logged_reward = float(np.sum(row.user_feedback))
            logged_clicks = np.array(
                list(row.user_feedback), dtype=np.int64,
            )

            try:
                slate_indices = np.array(
                    [self._cand_id_to_idx[int(x)] for x in logged_slate_ids],
                    dtype=np.int64,
                )
            except KeyError as exc:
                raise ValueError(
                    f"session {sid}: logged slate item not found in "
                    f"candidate universe — {exc}"
                ) from exc

            propensity = float(self._propensities[row_pos])
            obs = RecObs(
                user_features=user_features,
                candidate_features=self._candidate_features,
                candidate_ids=self._candidate_ids,
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
        yield steps
        emitted += 1
```

The previous `slate_propensity` call inside the loop is removed; the previous reliance on `self._df` for grouping is replaced with `self._ordered`.

- [ ] **Step 5: Run the parity test plus the full file's tests**

```
.venv/bin/python -m pytest tests/test_rl4rs_trajectory_ope.py -v
```

Expected: all PASS, including the new `test_loader_propensity_matches_per_call`.

- [ ] **Step 6: Run the full suite; expect 197 passed (196 + 1 new)**

```
.venv/bin/python -m pytest -q
```

Expected: 197 passed. The end-to-end smoke
(`test_pretrained_linucb_diverges_from_fresh_linucb`) keeps passing — propensity is numerically equivalent to the legacy path.

- [ ] **Step 7: Commit via HEREDOC**

```bash
git add rl_recsys/data/loaders/rl4rs_trajectory_ope.py tests/test_rl4rs_trajectory_ope.py
git commit -m "$(cat <<'EOF'
perf: batched propensity precompute in RL4RSTrajectoryOPESource

Replaces per-step slate_propensity calls (slate_size sequential
single-sample CUDA forwards) with a single batched precompute at
__init__ via slate_log_propensities_batch. Two orders of magnitude
fewer GPU launches; no API change. Numerical equivalence guarded by
the new parity test.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Run benchmark on real data, capture results

**Files:**
- Modify: `TODO.md`

This is the observational task: confirm pretrain finishes in minutes, then either record the discriminative-benchmark result or document why the agents still collapse.

- [ ] **Step 1: Confirm processed data is present**

```
ls -lah data/processed/rl4rs/sessions_b.parquet
```

Expected: `~171M` file. If missing, run `rl_recsys/data/cli.py` to regenerate.

- [ ] **Step 2: Run the benchmark, tee output to a log**

```
.venv/bin/python scripts/benchmark_rl4rs_b_seq_dr.py 2>&1 | tee /tmp/seq_dr_pretrain_run_v2.log
```

Expect to see, in order:
- `detected dims: ...`
- 5 lines `epoch N/5 mean_loss=...`
- `Behavior policy held-out NLL = ... (threshold ...)`
- `split: train=N sessions, eval=M sessions`
- A series of `propensity precompute: N rows in S.Ts` lines (one per RL4RSTrajectoryOPESource construction). On the RTX 5080 with ~219K rows per filter half, each should be under 5 minutes.
- `--- LinUCB (with offline pretrain) ---` plus `pretrain: ...` lines per seed.
- `VarianceEvaluation(...)` for LinUCB.
- `--- Random ---`.
- `VarianceEvaluation(...)` for Random.

If any `propensity precompute` line exceeds 10 minutes, STOP and investigate before proceeding. That would mean the batched method isn't actually batching — likely a tensor-shape regression.

- [ ] **Step 3: Update TODO.md**

Open `TODO.md`. Replace the existing "Batch the loader's `slate_propensity` calls" section under "Next up" with a new "RL4RS-B Sequential DR — current state" entry containing:

- The `propensity precompute` wall-clock numbers from the run.
- A markdown table of the new LinUCB-with-pretrain vs Random `avg_seq_dr_value ± std` numbers from the VarianceEvaluation output.
- A short paragraph describing whether the result is now discriminative. If yes, mark causes #1 (behavior model) and #3 (Boltzmann temperature) as the next investigations. If no, document which signal is still flat and which of the two remaining causes is now suspect.

Keep the rest of TODO.md ("Loader / data", "Evaluation", "Hygiene / polish", "Out of scope") unchanged. The pre-existing "Reference: pre-pretrain baseline" subsection from the prior batch can stay as historical context.

- [ ] **Step 4: Commit the doc update**

```bash
git add TODO.md
git commit -m "$(cat <<'EOF'
docs: capture batched-propensity benchmark run

[Replace this body with a one-paragraph summary of the actual result:
discriminative? what changed? what's the next investigation per the
remaining causes?]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(The bracketed body placeholder is intentional — fill from observed run output.)

---

## Self-review

**Spec coverage:**
- §Architecture, §Components.1 (slate_log_propensities_batch) → Task 1.
- §Components.2 (loader precompute + iter refactor) → Task 2.
- §Error handling: shape validation in Task 1; zero-propensity raise in Task 2; empty-filter raise preserved in Task 2 Step 4.
- §Testing: 3 new BehaviorPolicy tests in Task 1; 1 parity test in Task 2; existing end-to-end smoke unchanged (verified by Task 2 Step 6).
- §Performance verification → Task 3.

No gaps.

**Placeholder scan:** Task 3 Step 4's commit-message body is intentionally bracketed (the engineer fills it from the observed result). All other steps contain concrete code or commands.

**Type consistency:**
- `slate_log_propensities_batch` returns `np.ndarray` of dtype `float64`, shape `(B,)`. Caller (Task 2) does `np.exp(log_props).astype(np.float64)` — consistent.
- `users.shape == (B, user_dim)`, `slates.shape == (B, slate_size)`, `candidate_features.shape == (num_candidates, item_dim)` — consistent across method signature, test fixtures, and the loader call site.
- `self._propensities: np.ndarray` with `dtype=np.float64`, length `len(self._ordered)`. `iter_trajectories` indexes via `int` (`row_pos` from `group.index`); `propensity = float(self._propensities[row_pos])` — consistent.
