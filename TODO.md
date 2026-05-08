# TODO / Backlog

Items deferred from prior batches and observations that surfaced during runs. Ordered roughly by expected priority.

## Next up

### Make the RL4RS-B Sequential DR benchmark discriminative
First end-to-end run on dataset B (commit `457c0fa`, 5K trajectories × 3 seeds, RTX 5080) produced:

| Agent | avg_seq_dr_value | avg_logged_discounted_return |
|---|---|---|
| LinUCB | 10.565 ± 0.096 | 10.572 ± 0.096 |
| Random | 10.433 ± 0.079 | 10.572 ± 0.096 |

Pipeline is provably wired (finite, low-variance, logged baseline matches across agents). But all agents look like the behavior policy. Three stacked reasons:

1. **Behavior model barely beats uniform.** Trained NLL = 5.168 vs uniform NLL `log(283) = 5.645`. Either RL4RS-B's logging policy really is near-flat or the MLP is undertrained at 5 epochs / batch_size=512. Try (a) more epochs, (b) wider MLP, (c) add positional / sequential context features.
2. **LinUCB has zero history at evaluation time.** Benchmark constructs fresh `LinUCBAgent` per seed with identity `_a_matrix` and zero `_b_vector` → all `score_items` outputs equal → Boltzmann uniform → ratios ≈ 1. Need an offline-training pass over logged trajectories before evaluation, OR a stochastic-policy warmup that's actually learned.
3. **Boltzmann shim at T=1.0 smooths heavily.** Even with a trained agent, T=1 with similar score magnitudes drives target probs toward uniform. Sweep T ∈ {0.1, 0.3, 1.0, 3.0} and report the curve.

Probably attack (2) first — without LinUCB training the other two effects are masked.

### Vectorize / GPU notes captured
Training loop is now batched and runs on CUDA when available (commits `b3be8e4`, `457c0fa`). RTX 5080 epochs took ~2.7 min on 781K × 9 tuples / batch_size=512. The MLP is small (64-hidden, 1-output); GPU util was likely low. Consider larger batch_size (4096+) on real runs to amortize launch overhead.

## Loader / data

### Loader init perf at scale
Reduced from 60s → 16s on the 28M-row finn-no-slate parquet by switching the universe build to `pyarrow.compute.unique`. The remaining 16s is dominated by `pd.read_parquet` of the full DataFrame. Further wins available: (a) cache the loaded DataFrame across loader instances keyed by `(path, mtime)`, (b) accept a pre-loaded DataFrame in the constructor for power users, (c) only read non-slate columns lazily via pyarrow filtering at `iter_sessions` time.

### KuaiRec trajectory loader
KuaiRec has timestamped per-user interactions. A `KuaiRecTrajectoryLoader` would group by `user_id` ordered by `timestamp`, mirroring the finn-no-slate loader's interface. Slate semantics differ — KuaiRec is single-item per impression — so the replay rule needs adapting (logged_clicked_id = item_id when watch_ratio > threshold).

### User-state evolution within sessions
Current loader emits a per-session-invariant `user_features` (hash of user_id). Real RL needs state to evolve as a function of past clicks. Required for counterfactual rollout (Mode B) — until then, `evaluate_trajectory_agent` is locked to replay mode.

## Evaluation

### Trained DM reward model for DR
`dr_value` with `reward_model=None` falls back to `mean(rewards)`, which collapses DR to `avg_logged_reward` when weights are near 1. Train a regressor on logged (context, reward) pairs and pass it as `reward_model`. Unlocks DR's actual variance-reduction benefit.

### Counterfactual rollout mode for trajectory eval
Mode B from the trajectory design: agent's slate drives the next user state via a click model / simulator. Trajectories diverge from logged. Blocked on a click model.

### MDP checker / data understanding tool
RL4RS Sec 4: detect degenerate trajectories (all-zero rewards, missing transitions, trivial action distributions). Lives in `rl_recsys/data/` as a CLI that reports red flags before training.

### Dataset statistics validator
Schema/range/distribution sanity checks for processed parquets. Catches pipeline regressions before they propagate.

## Hygiene / polish

### `_aggregate_runs` doc + tests
Add a test for the empty-input path (`_aggregate_runs([])`) and for the metadata-skip behavior directly (rather than indirectly via `VarianceEvaluation` keys).

### Loader error-path tests
Add tests for `FinnNoSlateTrajectoryLoader`'s missing-columns error and the synthetic-padding fallback (item universe smaller than `n_pad`).

### Empty-input tests for metric helpers
`discounted_return([])` and `per_session_reward([])` both return 0.0 by guard. Add explicit tests.

## Out of scope (intentionally skipped)

- **Simulator quality metrics** (RL4RS Sec 5.1: slate/item AUC, F1, reward MSE) — interesting only once we have a click model worth measuring.
- **Business metrics** (RL4RS Sec 6: IPV, CVR, GMV) — stakeholder-facing numbers; not needed for research-level evaluation.
