# TODO / Backlog

Items deferred from prior batches and observations that surfaced during runs. Ordered roughly by expected priority.

## Next up

### Sub-project #1 results — multi-agent ablation grid on RL4RS-B (2026-05-10)

First real-data pass shipped. Scope: 9 CPU-friendly agents × 1 seed × pretrained ∈ {True, False}, 500 trajectories. Full table at `results/agent_grid/2026-05-10/summary.md`. Per-run JSON artifacts in the same directory.

| agent | pretrained=False | pretrained=True |
|---|---|---|
| gbdt | — | 9.919 |
| logged_replay | 49.912 | 49.912 |
| bc | 10.494 | 10.494 |
| lin_ts | 10.494 | 10.494 |
| most_popular | 10.134 | 10.494 |
| random | 10.134 | 10.134 |
| linucb | 10.494 | 10.010 |
| boltzmann_linear | 10.134 | 10.000 |
| eps_greedy_linear | 10.134 | 10.000 |

`avg_logged_discounted_return = 10.499` constant across all agents (property of the data).

**Observations.**
- Most pretrained agents land at ~10.5, identical to the logged baseline to 3 decimals. Same "all agents look like the behavior policy" pattern from the prior batch — the Boltzmann T=1 shim still smooths target probs toward μ.
- `logged_replay` reads **49.9**, well above the 10.5 baseline. This is not a bug — its `score_items` peaks 1.0 on the logged slate, so the temperature-1 softmax assigns much more mass to the logged action than μ does, IS weights blow up, and DR variance explodes. ESS for this agent is 30.7 / 500 logged steps (~6 %). The OPE estimator is unreliable here, not the agent — the takeaway is that the sanity-check we wanted (DR ≈ logged baseline) requires either a sharper score (e.g. 100/0 instead of 1/0) or a colder temperature.
- `gbdt` pretrained=False FAILED as expected (model is None until `train_offline` runs). The runner correctly wrote `gbdt_seed0_pretrained0.failed.json` and the aggregator skipped it. Future: either give batch-trained agents a sensible no-model fallback (uniform scores) or skip pretrained=False for them in the runner.
- Std = 0.0 throughout because only seed 0 ran. Expand to seeds {0, 1, 2} for variance bars.

**Caveats / deferred.**
- DL agents (NeuralLinear, SASRec, TopK REINFORCE, Decision Transformer) skipped in this run. They need either a CUDA window or hours of CPU time. Add them when GPU frees up.
- Boltzmann T sweep (cause #3 from the prior batch) still open.

**Next decisions** — pick one before re-running:
1. **Re-run the full grid with 3 seeds + DL agents** once GPU is free for ~3-4 hours. Capture proper mean ± std table.
2. **Fix the LoggedReplay sanity check first** — pick a score magnitude that yields target ≈ μ, document it in the agent docstring, re-run.
3. **Move to sub-project #5 (DM reward model)** — none of the bandit family is discriminative under the current shim, so a trained DM is the natural next leverage point. T sweep + DM reward model together would actually move the needle.

### Then: causes #1 and #3 from the prior batch
Once a discriminative signal exists:

1. **Behavior model barely beats uniform.** Trained NLL = 5.168 vs uniform NLL `log(283) = 5.645`. Either RL4RS-B's logging policy really is near-flat or the MLP is undertrained at 5 epochs / batch_size=512. Try (a) more epochs, (b) wider MLP, (c) add positional / sequential context features.
2. **Boltzmann shim at T=1.0 smooths heavily.** Even with a trained agent, T=1 with similar score magnitudes drives target probs toward uniform. Sweep T ∈ {0.1, 0.3, 1.0, 3.0} and report the curve.

### Reference: pre-pretrain baseline (commit `457c0fa`)
Last fully-completed real-data run, 5K trajectories × 3 seeds:

| Agent | avg_seq_dr_value | avg_logged_discounted_return |
|---|---|---|
| LinUCB | 10.565 ± 0.096 | 10.572 ± 0.096 |
| Random | 10.433 ± 0.079 | 10.572 ± 0.096 |

Pipeline is provably wired (finite, low-variance, logged baseline matches across agents). All agents look like the behavior policy — the discriminative-benchmark work continues with the propensity-batching fix above.

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
