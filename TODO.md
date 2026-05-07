# TODO / Backlog

Items deferred from prior batches and observations that surfaced during runs. Ordered roughly by expected priority.

## Next up

### Sequential DR estimator
Needs per-step propensity in trajectory data. The flagship CPE estimator from the RL4RS paper. Two sub-tasks:
- **Extend RL4RS pipeline** to extract behavior-policy propensities from raw CSVs and write multi-step `sessions.parquet` (currently collapsed to single-step). Required before we can implement true sequential DR on the paper's reference dataset.
- **Implement `seq_dr_value`** that walks per-step IS along a trajectory: `sum_t (prod_{u≤t} w_u) * (r_t − r̂_t) + V̂_t` with appropriate clipping. Lives in `rl_recsys/evaluation/ope.py` next to `dr_value`.

## Loader / data

### Loader init perf at scale
60s init on the 28M-row finn-no-slate parquet (set-based item universe build). Acceptable for offline batch use; not great for iterative experimentation. Options: (a) stream the `slate` column via pyarrow row-groups instead of materializing the full DataFrame, (b) cache `_item_universe` to disk keyed by `(path, mtime)`, (c) accept a pre-built universe in the constructor for power users.

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

### Aggregate-skip metadata for `_aggregate_runs`
`sessions`, `total_steps`, `episodes`, `seconds` show up in `VarianceEvaluation.mean`/`std` despite not being metrics. Tag those fields with `metadata={"aggregate": False}` on the source dataclass and have `_aggregate_runs` honor it.

### `_aggregate_runs` doc + tests
Docstring should note `np.std` uses ddof=0 (population std). Add a test for the empty-input path (`_aggregate_runs([])`).

### Loader error-path tests
Add tests for `FinnNoSlateTrajectoryLoader`'s missing-columns error and the synthetic-padding fallback (item universe smaller than `n_pad`).

### Empty-input tests for metric helpers
`discounted_return([])` and `per_session_reward([])` both return 0.0 by guard. Add explicit tests.

## Out of scope (intentionally skipped)

- **Simulator quality metrics** (RL4RS Sec 5.1: slate/item AUC, F1, reward MSE) — interesting only once we have a click model worth measuring.
- **Business metrics** (RL4RS Sec 6: IPV, CVR, GMV) — stakeholder-facing numbers; not needed for research-level evaluation.
