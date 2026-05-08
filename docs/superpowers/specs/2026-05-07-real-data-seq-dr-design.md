# Real-Data Sequential DR on RL4RS — Design

## Goal

Wire `seq_dr_value` and `evaluate_trajectory_ope_agent` (already shipped) up to a real multi-step dataset so we can report Sequential DR numbers for our agents. RL4RS dataset B is the only suitable source on disk: 439K sessions, mean 1.78 / max 4 steps, slate_size = 9.

## Background

- `seq_dr_value` is a pure formula. `evaluate_trajectory_ope_agent` is verified end-to-end on a synthetic source. Both currently assume `LoggedTrajectoryStep.logged_action: int` and a top-1 indicator target probability.
- Raw-CSV inspection (2026-05-07) found:
  - `rl4rs_dataset_a_rl.csv` is single-step per `session_id` — Seq DR collapses to bandit DR there. Skip.
  - `rl4rs_dataset_b_rl.csv` is multi-step (mean 1.78, max 4) and is the target dataset.
  - Neither CSV logs propensity. `behavior_policy_id` is constant. We must fit a behavior model.

## Decisions

1. **Action space: slate-as-action.** `logged_action` becomes the full slate (`np.ndarray` of `slate_size` ints). Propensity is a single scalar per step, computed internally as `Π_k μ(slate[k] | context, k)`. `seq_dr_value` is unchanged — it consumes scalar per-step propensity / target_prob arrays.
2. **Target probability via Boltzmann shim.** Deterministic indicator over a 9-slot slate is ~0 in practice. Wrap agent scores through `softmax(scores / T)` and define `π_target(slate | obs) = Π_k softmax(...)[slate[k]]`. We're evaluating a *smoothed* version of the agent — documented prominently. `T` defaults to 1.0.
3. **Behavior model: per-position softmax classifier.** Small MLP trained on logged `(user_features, item_features, position_index) → item_id` tuples, softmax over the candidate vocabulary. Calibration (held-out NLL) gates use of the fitted model.
4. **Reward shaping: `r_t = sum(user_feedback)`** at step `t`. Total clicks. Standard RL4RS.
5. **Agents must expose `score_items(obs) -> ndarray`.** Required contract for the Boltzmann shim. We add this to `LinUCBAgent` (UCB scores it already computes internally) and `RandomAgent` (uniform). Other agents added later as needed.
6. **Dataset A pipeline path is left untouched.** A is single-step; not useful here.

## Architecture

```
data/raw/rl4rs/.../rl4rs_dataset_b_rl.csv
    ↓ RL4RSPipeline.process_b()
data/processed/rl4rs/sessions_b.parquet
    columns: session_id, sequence_id, user_state, slate, item_features,
             user_feedback, candidate_ids, candidate_features
    ↓ fit_behavior_policy()
artifacts/rl4rs_b/behavior_policy.pt
    ↓ RL4RSTrajectoryOPESource.iter_trajectories()
LoggedTrajectoryStep(obs, logged_action: ndarray[slate_size],
                     logged_reward, propensity)
    ↓ evaluate_trajectory_ope_agent()
TrajectoryOPEEvaluation
    ↓ evaluate_trajectory_ope_with_variance()
{mean, std, ...}
```

## File map

| File | Action | Responsibility |
|---|---|---|
| `rl_recsys/data/pipelines/rl4rs.py` | extend | Add `process_b()`. Reads dataset B CSV, groups by `(session_id, sequence_id)`, emits `sessions_b.parquet`. Updates registry to expose `rl4rs_b` target. |
| `rl_recsys/data/schema.py` | extend | New `rl_sessions_b` schema with `candidate_ids`, `candidate_features` columns alongside the existing slate fields. |
| `rl_recsys/evaluation/behavior_policy.py` | create | `BehaviorPolicy` (PyTorch module + `slate_propensity` API), `fit_behavior_policy`, calibration helpers (`held_out_nll`, `expected_calibration_error`). |
| `rl_recsys/data/loaders/rl4rs_trajectory_ope.py` | create | `RL4RSTrajectoryOPESource` implementing `LoggedTrajectorySource`. Loads parquet, attaches a `BehaviorPolicy`, yields `LoggedTrajectoryStep`s with computed propensity. |
| `rl_recsys/evaluation/ope_trajectory.py` | modify | Change `LoggedTrajectoryStep.logged_action` type to `np.ndarray`. Rewrite `_target_probability` for Boltzmann + factorized slate. Add a `temperature: float = 1.0` parameter to `evaluate_trajectory_ope_agent` and forward it to `_target_probability`. Add ESS computation + reporting. Add `score_items` contract check at function entry. |
| `rl_recsys/agents/linucb.py` | extend | Add `score_items(obs) -> np.ndarray` returning per-candidate UCB scores. |
| `rl_recsys/agents/random.py` | extend | Add `score_items(obs) -> np.ndarray` returning uniform scores (zeros). |
| `rl_recsys/evaluation/variance.py` | extend | Add `evaluate_trajectory_ope_with_variance(make_source, make_agent, *, agent_name, max_trajectories, n_seeds, base_seed, gamma, reward_model, clip, temperature)`. Mirrors existing two siblings. |
| `tests/test_behavior_policy.py` | create | Unit tests for fit/calibration/slate_propensity. |
| `tests/test_rl4rs_trajectory_ope.py` | create | Loader unit tests + small-fixture integration smoke. |
| `tests/test_ope_trajectory.py` | modify | Update existing 10 tests to the new `logged_action: ndarray` shape and Boltzmann target prob. Add tests for `score_items` contract enforcement and slate-product propensity flow. |
| `tests/test_variance.py` | extend | Smoke test for `evaluate_trajectory_ope_with_variance`. |
| `scripts/benchmark_rl4rs_b_seq_dr.py` | create | Runnable benchmark: fits behavior policy, runs LinUCB and Random through Seq DR with variance, prints comparison. |

## Key formulas

**Behavior model output.** For a logged step with context `c = (user_features, item_features)`, position `k ∈ {0..slate_size-1}`, item `i`:

```
score_b(c, k, i) = MLP(concat(user_features, item_features[i], onehot(k)))
μ(i | c, k)      = softmax_{i' in candidates}(score_b(c, k, i'))[i]
μ(slate | c)     = Π_k μ(slate[k] | c, k)
```

**Target probability under Boltzmann shim.** For agent `a` exposing `score_items(obs) -> s ∈ R^|candidates|`:

```
π(i | obs, k) = softmax(s / T)[i]    # position-agnostic for now
π(slate | obs) = Π_k π(slate[k] | obs, k)
```

Note: the score function is position-agnostic for current agents (LinUCB scores items independent of slate position). Per-position score support can be added later by widening the agent contract.

**Per-step ratio.** `w_t = clip(π(slate_t | obs_t) / μ(slate_t | obs_t), clip_lo, clip_hi)`. Then `seq_dr_value` consumes `(rewards, target_probs=π(slate_t|obs_t), propensities=μ(slate_t|obs_t))` arrays unchanged.

## Effective sample size

`evaluate_trajectory_ope_agent` will compute and report `ess = (Σ_t W_t)^2 / Σ_t W_t^2` summed across all evaluated steps. If `ess / total_steps < 0.01`, log a warning. ESS is added to `TrajectoryOPEEvaluation` as a non-aggregating field so it surfaces in single runs but doesn't pollute variance output.

## Error paths (raise, don't silently degrade)

- `fit_behavior_policy`: held-out NLL above threshold → `ValueError("behavior policy NLL exceeds threshold: <actual> > <threshold>")`. Threshold default picked empirically once a baseline exists; first cut: `2 * log(num_candidates)` (twice the uniform NLL).
- `evaluate_trajectory_ope_agent`: agent missing `score_items` → `AttributeError("agent <name> must implement score_items(obs) -> np.ndarray for slate OPE")` raised at the start of the function, not in the inner loop.
- `BehaviorPolicy.slate_propensity`: returns 0 → `ValueError("zero propensity in logged slate")`. Defensive — softmax over finite logits never produces 0, but a numerical edge case shouldn't silently NaN the estimator.
- `_target_probability`: `temperature <= 0` → `ValueError`.

## Testing

| Layer | Test |
|---|---|
| `BehaviorPolicy` | Unit: 2-item, 2-context synthetic dataset where the conditional is hand-computable; assert fitted model recovers it within tolerance. Test calibration helper returns expected NLL on held-out. |
| Schema | `validate_parquet_schema` accepts a fixture with `rl_sessions_b` columns; rejects when `candidate_ids` is missing. |
| `RL4RSTrajectoryOPESource` | Synthetic mini-parquet (~10 sessions, 2 steps each). Assert: `iter_trajectories` shape, candidate-id stability across steps within a session, propensity in (0, 1], reward = sum(user_feedback). |
| `_target_probability` (rewritten) | T=1, slate_size=2, hand-computed Boltzmann probability. RandomAgent path returns uniform-product. Agent without `score_items` raises. `temperature <= 0` raises. |
| `evaluate_trajectory_ope_agent` (updated) | All 10 existing tests updated for `logged_action: ndarray`. Add one test asserting slate-product propensity is fed through unchanged to `seq_dr_value`. |
| `evaluate_trajectory_ope_with_variance` | 3 seeds, synthetic source → mean/std fields present and finite, std > 0. |
| Integration smoke | `tests/test_rl4rs_trajectory_ope.py::test_end_to_end_smoke` — small fixture (~50 sessions from real dataset B if cached, else generated synthetic), full pipeline including `BehaviorPolicy`, assert `avg_seq_dr_value` finite and ESS reported. |

## Out of scope

- **Dataset A**: single-step, not useful. Pipeline path untouched.
- **Other agents** beyond LinUCB/Random getting `score_items`. Add them when those agents land.
- **Hyperparameter sweep** over `T`, clip bounds. Defaults: `T=1.0`, `clip=(0.1, 10.0)`. Document and move on.
- **Behavior-policy artifact registry / re-use across runs**. First cut: benchmark script fits and discards (or saves to a known path). Caching infrastructure later.
- **Recursive DR, MAGIC, WIS variants**.
- **Per-position score function** in agents. Current agents score items independent of slate position; widening the contract waits for a model that needs it.

## Risks

1. **Trajectories are short.** Mean 1.78 steps, max 4. Sequential DR's per-step discount won't dramatically differentiate from bandit DR at γ=0.95. Real value here is having a *correctly-wired* OPE pipeline that we can apply to longer-trajectory datasets later. Set expectations accordingly in the benchmark report.
2. **Behavior model calibration is the long pole.** If the fitted MLP is poorly calibrated, propensities are wrong, and the whole estimator is meaningless. The NLL gate catches catastrophic failures; we should plot calibration curves on the first run and decide whether to invest in temperature scaling / Platt scaling before declaring this useful.
3. **Boltzmann shim changes the policy under evaluation.** What we report is the value of `softmax(LinUCB.score / T)`, not LinUCB itself. As `T → 0` it approaches LinUCB but propensity ratios blow up. Document this clearly in the benchmark output.
