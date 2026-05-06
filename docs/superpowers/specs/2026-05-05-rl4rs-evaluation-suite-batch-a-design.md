# RL4RS Evaluation Suite — Batch A Design

## Goal

Extend the verification suite with the missing evaluation metrics and estimators identified from the RL4RS paper (Wang et al. 2021): SWIS and DR OPE estimators, discounted return, per-session reward, and a multi-seed variance wrapper.

## Background

The RL4RS paper proposes a systematic evaluation framework with four CPE estimators (IS, SWIS, DR, Sequential DR), reward discounting, and results reported as mean ± std across five seeds. Our current suite has IS (as IPS) and SNIPS but lacks the rest. Sequential DR is deferred to Batch B pending propensity score extraction from raw RL4RS data.

## Architecture

**Option B selected:** new file per concern, extending existing modules only where the additions naturally belong.

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `rl_recsys/evaluation/ope.py` | Extend | Add `swis_value`, `dr_value`; add `swis_value` + `dr_value` fields to `OPEEvaluation`; populate in `evaluate_ope_agent` |
| `rl_recsys/training/metrics.py` | Extend | Add `discounted_return`, `per_session_reward` |
| `rl_recsys/evaluation/bandit.py` | Extend | Add `discounted_return` field to `BanditEvaluation`; compute it in `evaluate_bandit_agent` |
| `rl_recsys/evaluation/variance.py` | Create | `VarianceEvaluation` dataclass + `evaluate_with_variance` function |
| `tests/test_ope.py` | Extend | Four new tests for SWIS and DR |
| `tests/test_metrics.py` | Create | Three tests for `discounted_return` and `per_session_reward` |
| `tests/test_variance.py` | Create | Three tests for `evaluate_with_variance` |

## Component Design

### Section 1 — OPE Extensions (`rl_recsys/evaluation/ope.py`)

**`swis_value(rewards, target_probabilities, propensities, clip=(0.1, 10.0)) -> float`**

Step-Wise Importance Sampling with propensity ratio clipping. Identical to `ips_value` except each ratio `target/propensity` is clipped to `[clip_lo, clip_hi]` before averaging. Default clip range `(0.1, 10.0)` matches the paper. Validated via the existing `_validate_ope_arrays` helper.

**`dr_value(rewards, target_probabilities, propensities, reward_model=None, clip=(0.1, 10.0)) -> float`**

Doubly Robust estimator. Computes `mean(w * (r - r_hat) + r_hat)` where:
- `w = clip(target / propensity, clip_lo, clip_hi)`
- `r_hat[i] = reward_model(i)` when `reward_model` is provided
- `r_hat[i] = mean(rewards)` for all `i` when `reward_model=None`

`reward_model` signature: `Callable[[int], float]` — takes an episode index, returns a scalar estimate. The SWIS clipping is also applied to DR for consistency.

**`OPEEvaluation` dataclass additions:**
```python
swis_value: float
dr_value: float
```

`evaluate_ope_agent` computes and populates both. No breaking API change — callers that ignore the new fields are unaffected.

### Section 2 — New Metrics (`rl_recsys/training/metrics.py`)

**`discounted_return(rewards: np.ndarray, gamma: float = 0.95) -> float`**

Computes `sum(gamma^i * r_i)` over a 1D rewards array. Default γ=0.95 matches the paper. Pure function, no side effects.

**`per_session_reward(session_rewards: list[np.ndarray]) -> float`**

Takes a list of per-step reward arrays (one per session), returns `mean(sum(r) for r in session_rewards)`. Used when evaluating against trajectory data from `sessions.parquet` rather than single-step bandit episodes.

**`BanditEvaluation` addition:**
```python
discounted_return: float
```

`evaluate_bandit_agent` computes it from the per-episode reward sequence it already collects internally.

### Section 3 — Variance Wrapper (`rl_recsys/evaluation/variance.py`)

```python
@dataclass
class VarianceEvaluation:
    mean: dict[str, float]
    std: dict[str, float]
    n_seeds: int

def evaluate_with_variance(
    make_env: Callable[[], RecEnv],
    make_agent: Callable[[], Agent],
    *,
    agent_name: str,
    episodes: int,
    n_seeds: int = 5,
    base_seed: int = 42,
) -> VarianceEvaluation:
```

Runs `evaluate_bandit_agent` `n_seeds` times, seeding each run with `base_seed + i`. `make_env` and `make_agent` are factory callables — fresh instances per seed to prevent state leakage between runs. Returns `mean ± std` across all scalar fields of `BanditEvaluation`: `avg_reward`, `hit_rate`, `ctr`, `ndcg`, `mrr`, `discounted_return`. Default `n_seeds=5` matches the paper's reporting convention.

### Section 4 — Tests

**Additions to `tests/test_ope.py`:**
- `test_swis_clips_extreme_ratios` — verify a ratio outside `[0.1, 10.0]` is clipped, producing a different result than raw IPS
- `test_dr_uses_mean_reward_when_no_model` — with `reward_model=None`, DR correction uses `mean(rewards)` as the direct model
- `test_dr_uses_provided_reward_model` — with a constant `reward_model`, verify the DM correction is applied correctly
- `test_evaluate_ope_agent_populates_swis_and_dr` — end-to-end: `evaluate_ope_agent` returns finite `swis_value` and `dr_value`

**New `tests/test_metrics.py`:**
- `test_discounted_return_geometric_decay` — `[1.0, 1.0, 1.0]` with γ=0.5 equals `1 + 0.5 + 0.25 = 1.75`
- `test_discounted_return_single_step` — single-element array returns that element regardless of γ
- `test_per_session_reward_averages_sessions` — two sessions with totals 3 and 5 returns 4.0

**New `tests/test_variance.py`:**
- `test_evaluate_with_variance_returns_finite_mean_and_std` — smoke test over a synthetic env; all means and stds are finite
- `test_evaluate_with_variance_uses_fresh_instances` — `make_env` and `make_agent` are called exactly `n_seeds` times each
- `test_variance_std_is_zero_for_deterministic_env` — a fully deterministic env + agent returns std=0 across all metrics

## Deferred

**Sequential DR** — requires propensity scores per trajectory step. The `sessions.parquet` produced by our RL4RS pipeline does not currently include behavior policy propensities. Deferred to Batch B, which will also extend the RL4RS pipeline to extract propensity from the raw CSVs.

## Out of Scope (Batch B)

- Simulator quality metrics (slate/item accuracy, AUC, F1, reward prediction error)
- MDP checker / data understanding tool
- Business metrics (IPV, CVR, GMV)
- Dataset statistics validator
