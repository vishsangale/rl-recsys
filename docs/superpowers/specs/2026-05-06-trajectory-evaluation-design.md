# Trajectory Evaluation — Design

## Goal

Add multi-step trajectory evaluation to the verification suite so that `discounted_return` and `per_session_reward` (added in Batch A) are no longer inert and we have an evaluation path that consumes session-shaped logged data. Closes the bandit-only gap identified after running Batch A.

## Background

Batch A added `discounted_return` and `per_session_reward` metrics, but in bandit mode every episode is a single step — `gamma^0=1`, so `discounted_return` collapses to `avg_reward` and `per_session_reward` is unused. Our `data/processed/finn-no-slate/slates.parquet` already contains natural multi-step trajectories (avg 12 steps/user, max 20). RL4RS `sessions.parquet` is currently single-step due to a pipeline limitation; fixing that pipeline is a separate batch.

This batch builds the evaluator and one concrete loader. Other loaders (RL4RS-fixed, KuaiRec) plug into the same `TrajectoryDataset` Protocol later.

## Decisions

1. **Replay-only semantics.** Agent's slate at step *t* is scored against the logged user state and clicks at that step. Reward = `logged_reward` if the agent's slate covers `logged_clicked_id`, else 0. No counterfactual rollout (no click model required).
2. **Generic loader, finn-no-slate first.** A `TrajectoryDataset` Protocol decouples evaluator from data source. One concrete loader (`FinnNoSlateTrajectoryLoader`) ships now; future loaders plug in without touching the evaluator.
3. **Own `TrajectoryEvaluation` dataclass.** Session-aware fields (`avg_session_length`, `avg_session_reward`, `avg_discounted_return`) — not forced into bandit terminology. The variance wrapper is generalized to introspect any dataclass's numeric fields.
4. **Skip `agent.update()` during eval.** Pure evaluation: agent parameters are frozen for the run. Matches `evaluate_ope_agent`, avoids drifting a learning agent on biased replay rewards.

## Architecture

### File map

| File | Action | Responsibility |
|---|---|---|
| `rl_recsys/evaluation/trajectory.py` | Create | `TrajectoryStep`, `Session`, `TrajectoryDataset` Protocol, `TrajectoryEvaluation` dataclass, `evaluate_trajectory_agent` |
| `rl_recsys/data/loaders/finn_no_slate_trajectory.py` | Create | `FinnNoSlateTrajectoryLoader` reading `slates.parquet`, grouping by user_id |
| `rl_recsys/evaluation/variance.py` | Modify | Generalize `evaluate_with_variance` via dataclass introspection; add `evaluate_trajectory_with_variance` |
| `rl_recsys/evaluation/__init__.py` | Modify | Export new symbols |
| `tests/test_trajectory.py` | Create | 6 tests for evaluator and loader |
| `tests/test_variance.py` | Modify | 2 new tests for introspection + trajectory variance |

No changes to `RecEnv`, agents, `BanditEvaluation`, or `OPEEvaluation`.

## Data Types

```python
@dataclass(frozen=True)
class TrajectoryStep:
    obs: RecObs                  # candidates + user features at this step
    logged_slate: np.ndarray     # (S,) item IDs the user actually saw
    logged_clicked_id: int       # item ID the user clicked, or -1 if no click
    logged_reward: float         # 1.0 if click, 0.0 otherwise

@dataclass(frozen=True)
class Session:
    session_id: int
    steps: list[TrajectoryStep]

class TrajectoryDataset(Protocol):
    def iter_sessions(
        self, *, max_sessions: int | None = None, seed: int | None = None
    ) -> Iterator[Session]: ...

@dataclass
class TrajectoryEvaluation:
    agent: str
    sessions: int
    total_steps: int
    avg_session_reward: float
    avg_discounted_return: float
    avg_session_length: float
    avg_session_hit_rate: float
    avg_per_step_ctr: float
    avg_per_step_ndcg: float
    avg_per_step_mrr: float
    seconds: float

    def as_dict(self) -> dict[str, float | int | str]: ...
```

`TrajectoryStep` keeps `logged_clicked_id == -1` as a no-click sentinel for forward compatibility with future loaders that encode no-click sessions. The finn-no-slate parquet already filters to clicked rows only at pipeline time, so the loader emits no `-1` values today.

## Replay Reward Rule

At step *t*, agent returns a slate of candidate **indices**. The replay reward is:

- If `logged_clicked_id == -1` (no logged click): reward = 0.
- Else if `obs.candidate_ids[slate]` contains `logged_clicked_id`: reward = `logged_reward`, with a `clicks` vector of length `slate_size` carrying 1 at the position(s) where the slate covers the logged item, 0 elsewhere.
- Else: reward = 0, `clicks` vector all zeros.

ID-overlap (not position-overlap) is the right semantic: agents pick from a candidate pool, slate ordering need not match the logged slate. ndcg/mrr/ctr are computed per step from the agent-aligned `clicks` vector and aggregated as a flat per-step mean across all steps in all sessions.

## Loader Behavior

`FinnNoSlateTrajectoryLoader(parquet_path, *, num_candidates, feature_dim, slate_size, seed=0)`:

1. Reads `slates.parquet` (columns: `request_id, user_id, clicks, slate`). Streams via `pandas.groupby('user_id')`.
2. Groups by `user_id`, sorts each group by `request_id`. Each group becomes one `Session`.
3. For each row: `logged_slate = row.slate`, `logged_clicked_id = row.slate[row.clicks]` (the parquet's `clicks` column stores the slate index of the click; the pipeline pre-filters to clicked rows only, so this is always valid).
4. Builds `RecObs` per step:
   - `candidate_ids`: `num_candidates` items sampled deterministically per step. **Must include the logged slate's items** so coverage is possible (otherwise replay is degenerate). Implementation: take `logged_slate` (size 25 in finn-no-slate) and pad with hashed-deterministic random items up to `num_candidates`. If `num_candidates < len(logged_slate)`, raise `ValueError`.
   - `candidate_features`: hashed-feature embedding of each item ID (consistent with `OpenBanditEventSampler`).
   - `user_features`: hashed embedding of `user_id`. **Per-user invariant across the session** — user-state evolution within a session is left for counterfactual-rollout mode (future work).

`iter_sessions(max_sessions, seed)` yields up to `max_sessions` sessions in shuffled order (seed-deterministic). `seed=None` keeps file order.

## Evaluator Loop

```python
def evaluate_trajectory_agent(
    dataset: TrajectoryDataset,
    agent: Agent,
    *,
    agent_name: str,
    max_sessions: int,
    seed: int,
    gamma: float = 0.95,
) -> TrajectoryEvaluation:
    started = perf_counter()
    session_rewards = []           # one float per session
    session_disc_returns = []
    session_lengths = []
    session_hits = []              # one bool per session
    per_step_ctrs, per_step_ndcgs, per_step_mrrs = [], [], []
    total_steps = 0

    for session in dataset.iter_sessions(max_sessions=max_sessions, seed=seed):
        rewards_per_step = []
        for step in session.steps:
            slate_indices = np.asarray(agent.select_slate(step.obs), dtype=np.int64)
            slate_ids = step.obs.candidate_ids[slate_indices]
            covered = step.logged_clicked_id != -1 and step.logged_clicked_id in slate_ids
            if covered:
                clicks = (slate_ids == step.logged_clicked_id).astype(np.float64)
                r = step.logged_reward
            else:
                clicks = np.zeros(len(slate_indices), dtype=np.float64)
                r = 0.0
            rewards_per_step.append(r)
            per_step_ctrs.append(ctr(clicks))
            per_step_ndcgs.append(ndcg_at_k(clicks))
            per_step_mrrs.append(mrr(clicks))
            total_steps += 1
        rewards_arr = np.array(rewards_per_step, dtype=np.float64)
        session_rewards.append(float(rewards_arr.sum()))
        session_disc_returns.append(discounted_return(rewards_arr, gamma=gamma))
        session_lengths.append(len(session.steps))
        session_hits.append(float(rewards_arr.sum() > 0.0))

    return TrajectoryEvaluation(
        agent=agent_name,
        sessions=len(session_rewards),
        total_steps=total_steps,
        avg_session_reward=float(np.mean(session_rewards)),
        avg_discounted_return=float(np.mean(session_disc_returns)),
        avg_session_length=float(np.mean(session_lengths)),
        avg_session_hit_rate=float(np.mean(session_hits)),
        avg_per_step_ctr=float(np.mean(per_step_ctrs)),
        avg_per_step_ndcg=float(np.mean(per_step_ndcgs)),
        avg_per_step_mrr=float(np.mean(per_step_mrrs)),
        seconds=float(perf_counter() - started),
    )
```

`agent.update()` is **not** called.

## Variance Wrapper Generalization

Refactor `rl_recsys/evaluation/variance.py`:

1. Drop `_SCALAR_KEYS` constant.
2. Add a free helper `_aggregate_runs(results) -> (mean_dict, std_dict)` that uses `dataclasses.fields()` to enumerate scalar (`int`/`float`) fields off any dataclass and computes mean/std across runs. Non-numeric fields (e.g., `agent: str`) are filtered out.
3. `evaluate_with_variance` keeps its signature; body delegates to `_aggregate_runs`. Behavior unchanged for `BanditEvaluation`-shaped results.
4. New `evaluate_trajectory_with_variance(make_dataset, make_agent, *, agent_name, max_sessions, n_seeds=5, base_seed=42, gamma=0.95)` mirrors the bandit version but calls `evaluate_trajectory_agent`.
5. `VarianceEvaluation` dataclass unchanged.

Two functions instead of one polymorphic one: env-factory and dataset-factory have different keyword arguments (`episodes` vs `max_sessions`); explicit pair beats union-type branching.

## Testing

**`tests/test_trajectory.py`** (new):

1. `test_replay_reward_when_slate_covers_logged_click` — agent's slate contains `logged_clicked_id` → reward equals `logged_reward`.
2. `test_replay_reward_zero_when_slate_misses_logged_click` — agent's slate doesn't contain it → reward = 0.
3. `test_evaluate_trajectory_agent_aggregates_per_session` — synthetic 2-session dataset (3 steps each), all replay-covered, `RandomAgent` configured to deterministically include the logged item; verify `avg_session_reward`, `avg_discounted_return`, `avg_session_length`, `total_steps`, `sessions` match hand-computed values.
4. `test_evaluate_trajectory_agent_handles_uncovered_steps` — synthetic dataset where half the steps are uncovered; per-session reward sums match the covered-only count.
5. `test_evaluate_trajectory_agent_does_not_mutate_agent_state` — `LinUCBAgent`; snapshot `agent.A`/`agent.b` before the eval, confirm identical after. Validates the "skip update()" decision.
6. `test_finn_no_slate_loader_emits_sessions` — synthetic `slates.parquet` (3 users, 2-3 rows each); instantiate loader, iterate sessions, verify session count, step count, `logged_clicked_id` correctness.

**`tests/test_variance.py`** (extend):

7. `test_evaluate_with_variance_introspects_dataclass_fields` — pass a custom dataclass-returning fake evaluator; verify mean/std cover numeric fields and skip string fields.
8. `test_evaluate_trajectory_with_variance_returns_finite_mean_and_std` — synthetic mini parquet, `n_seeds=3`, assert all means/stds finite and `n_seeds == 3`.

Existing 3 tests in `test_variance.py` must still pass after the `_SCALAR_KEYS` removal.

## Out of Scope (Future Work)

- Counterfactual rollout mode (requires click model / simulator).
- RL4RS pipeline fix to produce real multi-step `sessions.parquet`.
- Sequential DR estimator (requires per-step propensity in the data).
- KuaiRec trajectory loader (group by user_id sorted by timestamp).
- User-state evolution within a session (currently per-session-invariant `user_features`).
