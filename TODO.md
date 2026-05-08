# TODO / Backlog

Items deferred from prior batches and observations that surfaced during runs. Ordered roughly by expected priority.

## Next up

### Real-data Sequential DR
The pure `seq_dr_value` estimator and `evaluate_trajectory_ope_agent` orchestrator are shipped (verified end-to-end on a synthetic source). What's left for real-dataset use, after inspecting the raw RL4RS CSVs:

**Findings from raw-CSV inspection (2026-05-07):**
- `rl4rs_dataset_a_rl.csv` (current pipeline target) is **inherently single-step** — every `session_id` has exactly one row. T=1 always, so Seq DR collapses to bandit DR there.
- `rl4rs_dataset_b_rl.csv` is multi-step but trajectories are **short** (mean 1.78 steps/session, max 4). Real-data Seq DR will run on B but won't dramatically differ from bandit DR at γ=0.95.
- Neither CSV logs per-step propensities. `behavior_policy_id` is constant (=1) — one logging policy, no action probabilities. Need an **estimated** behavior model.

**Work:**
- **Design note first** (use brainstorming skill): pick (a) action space — clicked-item-id vs slate-as-action vs per-position, (b) behavior-model class for propensity estimation, (c) target-probability definition for non-Random agents (current top-1 indicator collapses W to 0 most of the time on a 9-item slate).
- **Extend RL4RS pipeline** to also process dataset B and write per-step propensities to `sessions_b.parquet`.
- **Fit behavior-policy model** on logged `(context → clicked-item)` pairs; evaluate calibration before using its outputs as propensities.
- **Build a `LoggedTrajectorySource` impl** over the new parquet — yields `LoggedTrajectoryStep`s with estimated propensity.
- **Add `evaluate_trajectory_ope_with_variance`** sibling once a real source exists to drive it.

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
