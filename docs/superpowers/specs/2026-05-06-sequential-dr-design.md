# Sequential DR — Design

## Goal

Add the Sequential Doubly Robust (Seq DR) off-policy evaluator to the verification suite. Seq DR is the flagship CPE estimator from the RL4RS paper (Wang et al. 2021) for multi-step trajectory data.

## Background

Our existing OPE machinery (`rl_recsys/evaluation/ope.py`: IPS / SNIPS / SWIS / DR) is single-step bandit-only. Seq DR requires per-step propensity *and* multi-step trajectories. No current dataset in the project provides both:

- **finn-no-slate** has multi-step trajectories but no logged behavior-policy propensities.
- **Open Bandit** has propensities but is single-step.
- **RL4RS sessions.parquet** is collapsed to single-step due to a pipeline limitation (separate batch).

This batch ships the algorithmic primitive and a verifiable end-to-end path on **synthetic data we control**. Real-dataset wiring is deferred to a future batch that fixes the RL4RS pipeline.

## Decisions

1. **Scope: pure function + synthetic eval.** Implement `seq_dr_value` (per-trajectory pure function) and `evaluate_trajectory_ope_agent` (orchestrator). Test against a synthetic in-test trajectory source with hand-computable expected values. No real-dataset loader.
2. **Variant: per-decision DR with cumulative IS.** `V_DR(τ) = Σ_t γ^t [W_t (r_t − b_t) + b_t]`, where `W_t = Π_{u≤t} clip(π/μ)` and `b_t` is `reward_model(t)` or `mean(rewards)`. Mirrors the bandit `dr_value` structure; no fitted Q/V required (those would just collapse without a real model).
3. **Skip `agent.update()`.** Pure evaluation, agent state frozen — same rationale as `evaluate_trajectory_agent`.
4. **No variance wrapper sibling now.** YAGNI: users can wrap `evaluate_trajectory_ope_agent` via `_aggregate_runs` directly. Add a dedicated `evaluate_trajectory_ope_with_variance` once we have a real dataset to drive it.

## Architecture

### File map

| File | Action | Responsibility |
|---|---|---|
| `rl_recsys/evaluation/ope_trajectory.py` | Create | `seq_dr_value`, `LoggedTrajectoryStep`, `LoggedTrajectorySource` Protocol, `TrajectoryOPEEvaluation`, `evaluate_trajectory_ope_agent` |
| `rl_recsys/evaluation/__init__.py` | Modify | Export new symbols |
| `tests/test_ope_trajectory.py` | Create | 6 tests + a `_SyntheticTrajectorySource` fixture |

`rl_recsys/evaluation/ope.py` is **not** modified. Sequential pieces are isolated in the new module — mirrors the `bandit.py` (single-step) → `trajectory.py` (multi-step) split.

## Pure Function

```python
def seq_dr_value(
    rewards: np.ndarray,                 # (T,)
    target_probabilities: np.ndarray,    # (T,)
    propensities: np.ndarray,            # (T,)
    *,
    gamma: float = 0.95,
    reward_model: Callable[[int], float] | None = None,
    clip: tuple[float, float] = (0.1, 10.0),
) -> float:
```

**Formula:**

```
w_t = clip(π(a_t|s_t) / μ(a_t|s_t),  clip_lo,  clip_hi)
W_t = Π_{u=0..t} w_u
b_t = reward_model(t)  if reward_model  else  mean(rewards)
V̂_DR(τ) = Σ_{t=0..T-1}  γ^t · [ W_t · (r_t − b_t)  +  b_t ]
```

Returns the scalar `V̂_DR(τ)` — sequential DR estimate of the discounted return for trajectory τ under the target policy.

**Collapse cases:**
- `target = behavior` (W_t = 1) → `Σ γ^t r_t` (logged discounted return)
- `b = 0` → `Σ γ^t W_t r_t` (per-decision IS)
- `T = 1` → matches the bandit `dr_value` formula on a single observation

**Validation:** reuses `_validate_ope_arrays` from `ope.py` (length / finiteness / probability range checks). Empty trajectory raises `ValueError`.

## Trajectory Source and Evaluator

### `LoggedTrajectoryStep` (frozen dataclass)

```python
@dataclass(frozen=True)
class LoggedTrajectoryStep:
    obs: RecObs
    logged_action: int       # index into obs.candidate_ids
    logged_reward: float
    propensity: float        # μ(logged_action | obs)
```

### `LoggedTrajectorySource` (Protocol)

```python
class LoggedTrajectorySource(Protocol):
    def iter_trajectories(
        self, *, max_trajectories: int | None = None, seed: int | None = None
    ) -> Iterator[list[LoggedTrajectoryStep]]:
        ...
```

A trajectory is a `list[LoggedTrajectoryStep]`. No session-id wrapper.

### `evaluate_trajectory_ope_agent`

```python
def evaluate_trajectory_ope_agent(
    source: LoggedTrajectorySource,
    agent: Agent,
    *,
    agent_name: str,
    max_trajectories: int,
    seed: int,
    gamma: float = 0.95,
    reward_model: Callable[[int], float] | None = None,
    clip: tuple[float, float] = (0.1, 10.0),
) -> TrajectoryOPEEvaluation:
```

**Loop:**

```
for traj in source.iter_trajectories(max_trajectories, seed):
    rewards, target_probs, propensities = [], [], []
    for step in traj:
        slate = agent.select_slate(step.obs)
        top_action = int(slate[0])
        target_prob = _target_probability(agent, step.obs, top_action, step.logged_action)
        rewards.append(step.logged_reward)
        target_probs.append(target_prob)
        propensities.append(step.propensity)
    seq_dr = seq_dr_value(rewards_arr, target_probs_arr, propensities_arr,
                          gamma=gamma, reward_model=reward_model, clip=clip)
    logged_return = discounted_return(rewards_arr, gamma=gamma)
    # accumulate seq_dr, logged_return, len(traj) into per-trajectory lists
```

`_target_probability` returns `1/num_candidates` for `RandomAgent`, else `1.0 if top_action == logged_action else 0.0`. Copied (not imported) into `ope_trajectory.py` — two small copies are clearer than a private cross-module import.

`agent.update()` is **not** called.

### `TrajectoryOPEEvaluation` dataclass

```python
@dataclass
class TrajectoryOPEEvaluation:
    agent: str
    trajectories: int = field(metadata={"aggregate": False})
    total_steps: int = field(metadata={"aggregate": False})
    avg_seq_dr_value: float
    avg_logged_discounted_return: float   # baseline for comparison
    seconds: float = field(metadata={"aggregate": False})

    def as_dict(self) -> dict[str, float | int | str]: ...
```

`avg_logged_discounted_return` gives the reader the no-policy-change baseline to compare `avg_seq_dr_value` against.

## Testing

`tests/test_ope_trajectory.py` (new):

1. **`test_seq_dr_value_collapses_to_logged_return_when_target_equals_behavior`** — `target = propensity` everywhere → `W_t = 1` → result equals `Σ γ^t r_t`.

2. **`test_seq_dr_value_collapses_to_per_decision_is_when_baseline_zero`** — `reward_model = lambda i: 0.0` → result equals `Σ γ^t W_t r_t`. Verifies formula structure.

3. **`test_seq_dr_value_clips_extreme_ratios`** — construct a trajectory with one ratio outside `[0.1, 10]`; verify clipping changes the result vs. unclipped manual computation.

4. **`test_seq_dr_value_uses_provided_reward_model`** — pass a non-trivial `reward_model`; verify the per-step baseline correction is applied. Hand-computable.

5. **`test_evaluate_trajectory_ope_agent_aggregates_per_trajectory`** — synthetic source with 2 trajectories of known `seq_dr_value`s; assert `avg_seq_dr_value` is their mean, `trajectories == 2`, `total_steps` sums correctly.

6. **`test_evaluate_trajectory_ope_agent_does_not_mutate_agent_state`** — `LinUCBAgent`; snapshot `_a_matrix`/`_b_vector`; run; assert unchanged.

**Synthetic fixture** (in-test): `_SyntheticTrajectorySource` accepts a pre-built `list[list[LoggedTrajectoryStep]]` and replays deterministically. Each test constructs trajectories by hand with known propensity / target_probability / reward values so the expected `seq_dr_value` can be computed analytically.

## Out of Scope (Future Work)

- Real-dataset `LoggedTrajectorySource` (requires propensity + multi-step data — the RL4RS pipeline fix unlocks this).
- `evaluate_trajectory_ope_with_variance` sibling — defer until a real dataset drives the need.
- Recursive DR (Jiang & Li 2016) — needs a fitted V̂; would collapse without one.
- MAGIC blending / weighted IS / SNIPS-trajectory variants — none demanded by current use cases.
