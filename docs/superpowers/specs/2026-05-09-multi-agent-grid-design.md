# Multi-Agent Ablation Grid on RL4RS-B — Design

Date: 2026-05-09

## Context

Sub-project #1 of a multi-batch push that asked: "Implement a lot of agents
to really see which one works best on which data."

The full request decomposes into five sub-projects:

1. **Agent zoo + ablation harness on RL4RS-B** — this spec.
2. KuaiRec Sequential DR pipeline (depends on #1's harness).
3. finn-no-slate Sequential DR pipeline.
4. MovieLens Sequential DR pipeline.
5. Trained DM reward model for DR (cross-cutting; unblocks DR's variance
   reduction across all datasets).

Each downstream sub-project gets its own spec → plan → implementation
cycle. This spec covers only #1.

## Motivation

The repo currently has two agents (`Random`, `LinUCB`). The Sequential DR
benchmark on RL4RS-B is wired and discriminative-capable after the
offline LinUCB pretrain (`f9fe2d2`) and batched propensity precompute
(`be83da2`) batches. We do not know which agent family wins on RL4RS-B
because no other agents exist to compare against.

Sub-project #1 produces a **research-style ablation grid**: 13 agents
under a stable, dataset-agnostic interface, evaluated across the (seed ×
pretrained) core grid and aggregated into a single comparison table.
The result is a defensible answer to "which agent wins on RL4RS-B and
under what conditions" plus a reusable harness every later
sub-project can plug into.

## Goals

- Add 12 new agents to a stable `Agent` interface (existing `Random` and
  `LinUCB` carry forward unchanged).
- Keep agents dataset-agnostic — no agent imports anything dataset-specific.
- Extend the per-step observation with optional sequence history so DL
  and RL agents have a non-trivial state to consume on RL4RS-B.
- Build a harness that runs the (agent × seed × pretrained) core grid on
  RL4RS-B and produces a summary table.
- Produce a checked-in `summary.md` so reviewers can read the result
  without re-running the grid.

## Non-goals

- Trajectory loaders or behavior policies for KuaiRec, finn-no-slate,
  MovieLens (sub-projects #2–#4).
- Trained DM reward model (sub-project #5).
- Counterfactual rollout / full state-evolution rewrite. We ship a
  minimal optional `RecObs.history` field instead.
- Per-agent deep ablation sweeps. Core grid only (`seed ∈ {0, 1, 2}`,
  `pretrained ∈ {True, False}`, T fixed at 1.0). Boltzmann T sweep,
  hyperparameter sweeps, and selective-deep tiers are deferred to
  follow-up batches once we know which agents are even worth ablating.
- Refactoring the existing `LinUCBAgent`. It stays as-is and serves as
  the reference implementation for the linear bandit family.

## Agents added

| # | Family | Class | File | Notes |
|---|---|---|---|---|
| 1 | Heuristic | `MostPopularAgent` | `agents/most_popular.py` | Train-set click-count ranking |
| 2 | Heuristic | `LoggedReplayAgent` | `agents/logged_replay.py` | Replays logged slate; OPE sanity check (DR ≈ logged baseline) |
| 3 | Heuristic | `OracleClickAgent` | `agents/oracle_click.py` | Cheats with future clicks; upper bound; replay-mode only |
| 4 | Linear bandit | `LinTSAgent` | `agents/lin_ts.py` | Linear Thompson sampling |
| 5 | Linear bandit | `EpsGreedyLinearAgent` | `agents/eps_greedy_linear.py` | ε-greedy on linear regressor |
| 6 | Linear bandit | `BoltzmannLinearAgent` | `agents/boltzmann_linear.py` | Softmax-T on linear scores |
| 7 | Neural | `NeuralLinearAgent` | `agents/neural_linear.py` | MLP feature extractor + LinUCB head (Riquelme et al.) |
| 8 | Imitation | `BCAgent` | `agents/bc.py` | Wraps trained `BehaviorPolicy` |
| 9 | Imitation | `GBDTAgent` | `agents/gbdt.py` | LightGBM ranker on (context, item, click) |
| 10 | DL ranker | `SASRecAgent` | `agents/sasrec.py` | Self-attention over session history |
| 11 | Offline RL | `TopKReinforceAgent` | `agents/topk_reinforce.py` | Chen et al. 2019; reuses precomputed propensities |
| 12 | Offline RL | `DecisionTransformerAgent` | `agents/decision_transformer.py` | Chen et al. 2021; sequence-models (R, s, a) |

The existing `Random` and `LinUCB` are not re-implemented; they are
included in the harness's default agent set so the grid has 14 rows.

## Architecture

### Agent ABC (extended)

`rl_recsys/agents/base.py`:

```python
class Agent(ABC):
    @abstractmethod
    def select_slate(self, obs: RecObs) -> np.ndarray: ...

    @abstractmethod
    def update(
        self, obs: RecObs, slate: np.ndarray, reward: float,
        clicks: np.ndarray, next_obs: RecObs,
    ) -> dict[str, float]: ...

    def train_offline(
        self,
        source: LoggedTrajectorySource,
        *,
        seed: int = 0,
    ) -> dict[str, float]:
        """Default: per-step update via pretrain_agent_on_logged.
        DL and batch-trained agents override with their own routines.
        """
        from rl_recsys.training.offline_pretrain import pretrain_agent_on_logged
        return pretrain_agent_on_logged(self, source, seed=seed)

    def score_items(self, obs: RecObs) -> np.ndarray:
        """Per-candidate scores (used by Boltzmann shim and harness diagnostics).
        Default: zeros (uniform softmax)."""
        return np.zeros(len(obs.candidate_features), dtype=np.float64)
```

The `train_offline` default keeps the existing `LinUCB` pretrain working
unchanged. Heuristic agents that need no training override with a no-op
returning `{}`. DL agents override with batch training loops. The
default `score_items` lets `Random` and replay-only agents work
trivially with the Boltzmann target-policy shim already used by
Sequential DR.

### `RecObs` extension

`rl_recsys/environments/base.py`:

```python
@dataclass(frozen=True)
class HistoryStep:
    slate: np.ndarray   # (slate_size,) candidate indices
    clicks: np.ndarray  # (slate_size,) 0/1

@dataclass(frozen=True)
class RecObs:
    user_features: np.ndarray
    candidate_features: np.ndarray
    history: tuple[HistoryStep, ...] = ()
    # Replay-mode-only fields. None outside replay sources.
    logged_action: np.ndarray | None = None
    logged_clicks: np.ndarray | None = None
```

Defaults preserve the existing constructor signature for callers that
don't pass these (existing tests, agents `Random` and `LinUCB`).

`RL4RSTrajectoryOPESource.iter_trajectories` is updated to populate
`history` per step (cumulative within session) and
`logged_action`/`logged_clicks` (always populated in replay mode, which
is the only mode the loader supports today).

### Trajectory loader change

`rl_recsys/data/loaders/rl4rs_trajectory_ope.py` `iter_trajectories`
walks `self._ordered.groupby("session_id")`. Inside each session loop:

```python
history: list[HistoryStep] = []
for row_pos, row in group_with_pos:
    slate_indices = ...
    clicks = ...
    obs = RecObs(
        user_features=...,
        candidate_features=self._candidate_features,
        history=tuple(history),
        logged_action=slate_indices,
        logged_clicks=clicks,
    )
    yield LoggedTrajectoryStep(
        obs=obs,
        logged_action=slate_indices,
        logged_clicks=clicks,
        logged_reward=...,
        propensity=self._propensities[row_pos],
    )
    history.append(HistoryStep(slate=slate_indices, clicks=clicks))
```

`history` accumulates within the session, resets between sessions.
Existing tests must continue to pass; `LoggedTrajectoryStep.obs` now
carries non-trivial `history` after the first step of every session.

### Per-agent specs

#### A. Heuristics

**`MostPopularAgent`**
- Constructor: `(slate_size, num_candidates)`.
- State: `np.ndarray` `clicks_per_item` of shape `(num_candidates,)`,
  initialized to zero.
- `train_offline(source)`: iterate trajectories; for each step add
  `clicks` to `clicks_per_item[slate_indices]`. Returns
  `{"items_seen": ..., "clicks": ...}`.
- `select_slate(obs)`: `argsort(clicks_per_item)[-slate_size:][::-1]`.
- `score_items(obs)`: `clicks_per_item.astype(float)` (used by Boltzmann shim).
- `update`: no-op (returns `{}`).
- Assumes `obs.candidate_features` row order matches the global candidate
  index. The loader already enforces this via
  `_cand_id_to_idx`/`_candidate_features`.

**`LoggedReplayAgent`**
- Constructor: `(slate_size,)`. Carries no learnable state.
- `select_slate(obs)`: returns `obs.logged_action`. Raises `ValueError`
  if `obs.logged_action is None`.
- `score_items(obs)`: returns a vector of zeros except `+1.0` at the
  positions in `obs.logged_action`. Yields a softmax peaked on the
  logged slate after the Boltzmann shim.
- `train_offline`, `update`: no-ops.
- Validation: harness asserts `LoggedReplayAgent` is paired only with a
  replay-mode `LoggedTrajectorySource`.

**`OracleClickAgent`**
- Constructor: `(slate_size,)`.
- `select_slate(obs)`: ranks `obs.logged_action` items by their
  corresponding `obs.logged_clicks`, takes the top-`slate_size`. Pads
  with non-logged candidates if fewer than `slate_size` items were
  clicked. Raises if either field is `None`.
- `score_items(obs)`: vector with `obs.logged_clicks` at logged
  positions and zero elsewhere.
- `train_offline`, `update`: no-ops.
- Documented as eval-only upper bound; cheats by reading future labels.

#### B. Linear bandits

All three share a `LinearBanditBase` mixin (in `agents/_linear_base.py`)
that holds `A`, `b`, `feature_dim`, `_candidate_features`,
`_user_dim`, `_item_dim`, `_interaction_dim`. Same feature
construction as the existing `LinUCBAgent`. The mixin owns `update` and
`_candidate_features`; subclasses override `score_items` and (for
EpsGreedy only) `select_slate`.

`LinUCBAgent` is **not** retrofitted onto the mixin in this spec —
keeping the diff narrow. Mixin is a new module; the existing class is
untouched.

**`LinTSAgent`**
- Param: `sigma=1.0`.
- `score_items(obs)`: solve `A^{-1} b` and a Cholesky factor of `A^{-1}`
  (via `np.linalg.cholesky(np.linalg.inv(A))`); sample `θ ~ N(A^{-1} b,
  σ² A^{-1})` once per `select_slate` call; return `features @ θ`.
- `update`: same as `LinUCB.update`.
- Determinism: tests pass an explicit `numpy.random.Generator`.

**`EpsGreedyLinearAgent`**
- Param: `epsilon=0.1`.
- `select_slate(obs)`: with probability `epsilon`, return
  `rng.choice(num_candidates, size=slate_size, replace=False)`.
  Otherwise top-K of `score_items`.
- `score_items(obs)`: linear point estimate `features @ A^{-1} b`. No
  exploration bonus.
- `update`: same as `LinUCB.update`.

**`BoltzmannLinearAgent`**
- Param: `temperature=1.0`.
- `score_items(obs)`: `features @ A^{-1} b` (no bonus).
- `select_slate(obs)`: Plackett-Luce sample of size `slate_size` without
  replacement from `softmax(scores / temperature)`. Implementation:
  Gumbel-top-K trick — `argsort(scores/T + Gumbel(0,1))[-slate_size:]`.
- `update`: same as `LinUCB.update`.

#### C. Neural contextual

**`NeuralLinearAgent`** — Riquelme et al. NeuralLinear.

Architecture:
- `self._mlp = nn.Sequential(nn.Linear(user_dim+item_dim, hidden_dim),
  nn.ReLU(), nn.Linear(hidden_dim, embedding_dim))`.
- UCB head lives in embedding space: `A ∈ R^{D×D}`, `b ∈ R^D` with
  `D = embedding_dim`.

Training (`train_offline`):
1. Build a regression dataset over the train trajectories: rows are
   `(concat(user, item), click)` for every (step, slate position).
2. Train the MLP with MSE loss to predict click ∈ {0, 1} for `mlp_epochs`
   epochs, batch size 4096, Adam lr 1e-3. CUDA preferred.
3. Freeze the MLP. Walk the training trajectories one more time;
   compute per-(step, slate position) embedding `φ(user, item)` and
   accumulate `A += φφ^T`, `b += click · φ`.

Inference (`select_slate`):
- Forward all candidates: `φ(user, candidate_k)` → embedding matrix
  `Φ ∈ R^{N×D}`.
- UCB scores: `Φ @ A^{-1} b + alpha · sqrt(diag(Φ A^{-1} Φ^T))`.
- Top-K.

Params: `hidden_dim=64`, `embedding_dim=32`, `mlp_epochs=5`,
`alpha=1.0`. CUDA preferred; `--cpu-fallback` allowed for unit tests.

#### D. Imitation

**`BCAgent`**
- Constructor: `(slate_size, behavior_policy: BehaviorPolicy | None,
  candidate_id_to_idx, candidate_features)`.
- `train_offline(source)`:
  - If `behavior_policy is not None`, no-op (already trained); return
    `{}`.
  - Else build a `BehaviorPolicy` and call `.fit(...)` on the source's
    materialized (user, slate) training dataset. Reuses existing
    `BehaviorPolicy` infra.
- `select_slate(obs)`:
  - Compute per-position score logits via
    `BehaviorPolicy._score_batch(obs.user_features[None, :],
    candidate_features, position=k)` for each position `k ∈
    [0, slate_size)`.
  - Sum logits across positions to get per-candidate scores.
  - Top-K argsort.
- `score_items(obs)`: same per-candidate sum-of-position-logits.
- `update`: no-op.

**`GBDTAgent`** — LightGBM ranker.
- Dependency: `lightgbm` added to `requirements.txt`.
- Constructor: `(slate_size, candidate_features)` plus hyperparams.
- `train_offline(source)`:
  - Build `(features, click)` regression dataset where `features =
    concat(user_features, candidate_features[item_idx])` for every
    (step, slate position).
  - Fit `lgb.LGBMRegressor(n_estimators=100, max_depth=6,
    learning_rate=0.05)`.
- `select_slate(obs)`: predict score per candidate, top-K.
- `score_items(obs)`: same predictions.
- `update`: no-op. CPU-only — no GPU contention.

#### E. Sequential / DL

**`SASRecAgent`** — Kang & McAuley 2018, adapted for slate scoring.

Encoder:
- `nn.Embedding(num_candidates+1, hidden_dim)` (item embedding; +1 for
  pad token).
- Positional encoding: learned, `nn.Embedding(max_history_len,
  hidden_dim)`.
- `n_blocks` self-attention blocks (`n_heads` heads each), causal mask.
- Pool the last position's hidden state → `h_session ∈ R^{hidden_dim}`.

Click signal injection: each history step contributes `slate_size`
tokens, one per slate position, each token's embedding scaled by
`(1 + click_t)` so clicked items get amplified. Empty-history sessions
get a learned `cls` sentinel token.

Scorer:
- `score(item) = h_session · W_out @ item_embedding + bias`.
- `select_slate(obs)`: top-K on this score across all candidates.

Training (`train_offline`):
- Standard sequential rec loss adapted for slates: at each step `t`
  with non-empty `clicks_{t+1}`, predict the clicked items at `t+1`
  from history `[1..t]` via cross-entropy over the candidate universe
  (positives = clicked items at t+1; negatives = sampled from the rest
  with k=64 negatives per positive).
- Loss summed over all `t` in the session.
- Optimizer: Adam lr 1e-3, batch size 256 sessions, `epochs=10`.
- CUDA preferred. Falls back to CPU with a warning when
  `--cpu-fallback` is set.

Params: `hidden_dim=64`, `n_heads=2`, `n_blocks=2`, `max_history_len=20`,
`epochs=10`.

**`TopKReinforceAgent`** — Chen et al. 2019 top-K off-policy correction.

Architecture:
- Same SASRec-style encoder as `SASRecAgent` (separately instantiated;
  no shared module yet — see "Out of scope" note below).
- Scoring head emits a per-position softmax distribution over the
  candidate universe: for each position `k ∈ [0, slate_size)`, a linear
  layer `W_k @ h_session` → logits over `num_candidates`.

Training (`train_offline`):
- For each logged step, compute target-policy probability
  `π_θ(slate | s_t) = Π_k softmax(logits_k)[slate[k]]`.
- Importance weight: `ρ_t = π_θ(slate | s_t) / μ(slate | s_t)` where μ is
  the precomputed propensity from `RL4RSTrajectoryOPESource._propensities`
  (load-bearing reuse of the prior batch's perf fix).
- Top-K correction factor (paper Eq. 8): `λ_K(slate) = K · (1 -
  π_θ(slate))^{K-1}` for the top-k inclusion-probability adjustment.
  Clamped to `[1, K]` to avoid pathologies on near-deterministic π_θ.
- Loss: `-Σ_t [clip(ρ_t, 0, c) · λ_K · Σ_k log π_θ(slate[k] | s_t)] ·
  reward_t`.
- Clip threshold `c=10`. Adam lr 1e-3, batch size 256 sessions,
  `epochs=10`.
- CUDA preferred.

Inference: `select_slate(obs)` runs the encoder + per-position head,
then takes the top-K of the **summed** per-position logits across
positions (treats it as a single ranking head, sidestepping
position-specific decoding for inference simplicity).

Params: `hidden_dim=64`, `n_heads=2`, `n_blocks=2`,
`max_history_len=20`, `clip_c=10`, `epochs=10`.

**`DecisionTransformerAgent`** — Chen et al. 2021 (NeurIPS).

Architecture:
- Causal transformer over interleaved sequence `[R_1, s_1, a_1, R_2, s_2,
  a_2, ...]`.
- `R_t`: scalar return-to-go projected to `hidden_dim` via a linear
  layer.
- `s_t`: encoded by a small MLP over `(user_features ||
  cumulative_click_count_features)`.
- `a_t`: pooled mean of item embeddings of the slate.
- `n_blocks` causal-attention blocks.

Training (`train_offline`):
- For each session compute return-to-go `R_t = Σ_{u≥t} γ^{u-t} r_u`. DT
  reads γ from `EvalConfig.gamma` (default `0.95` to match
  `evaluate_trajectory_agent`'s default), so DT's return-to-go matches
  the Sequential DR discount.
- Predict `a_t` from `[R_1, s_1, a_1, ..., R_t, s_t]` via MSE on the
  pooled action embedding plus an auxiliary cross-entropy against the
  slate's individual items (auxiliary helps decoding).
- Adam lr 1e-3, batch size 64 sessions, `epochs=10`. CUDA preferred.

Inference (`select_slate(obs)`):
- Condition on `target_return = max return observed in train`.
- The state for the current step is built from `obs.history`.
- Greedy-decode the slate: predict the pooled action embedding, then
  pick the top-`slate_size` candidates closest to it under cosine
  similarity. (Token-by-token autoregressive decoding adds complexity
  not worth paying for here; cosine-similarity decoding is the simpler
  variant used by several DT-for-rec papers.)

Params: `hidden_dim=64`, `n_blocks=3`, `context_window=20`,
`target_return` set at training time to the train-set max,
`epochs=10`.

**Caveat documented in the results table:** with constant
`user_features` on RL4RS-B, "state evolution" in DT comes solely from
`obs.history`. Until full state evolution lands, expect TopK and DT
results to be a preliminary lower bound, not the agent's true ceiling.

## Harness

### CLI

`scripts/benchmark_agent_grid.py`:

```
python scripts/benchmark_agent_grid.py \
    --dataset rl4rs_b \
    --agents linucb,lints,sasrec   # or 'all'
    --seeds 0,1,2 \
    --pretrained-modes both         # both | true | false
    --max-trajectories 5000 \
    --boltzmann-T 1.0 \
    --output-dir results/agent_grid/2026-05-09 \
    --resume \
    --cpu-fallback
```

Behavior:
- `--agents all` expands via `agents.factory.AGENT_REGISTRY.keys()`
  minus an opt-out set (`OracleClickAgent` is opt-in only — must be
  named explicitly). The existing `factory.py` if/elif dispatch is
  refactored to an `AGENT_REGISTRY: dict[str, Callable[[AgentConfig,
  EnvConfig], Agent]]`. `build_agent` becomes a one-line
  `AGENT_REGISTRY[name.lower()](agent_cfg, env_cfg)` with a
  `KeyError → ValueError("Unknown agent")` translation. Adding new
  agents is then "register one entry."
- `--resume` makes the grid restartable: per-run JSON files have a
  deterministic name `{agent}_seed{seed}_pretrained{T|F}.json`; runner
  skips runs whose file already exists.
- For each (agent, seed, pretrained) tuple:
  1. Build train/eval sources via the existing 50/50 session split.
  2. Build the agent via factory.
  3. If `pretrained`: `agent.train_offline(train_source, seed=seed)`.
  4. Run `evaluate_trajectory_agent(agent, eval_source,
     temperature=<from --boltzmann-T>)`. (The function's existing
     `temperature` kwarg is the Boltzmann shim. CLI flag is named
     `--boltzmann-T` for UX clarity.)
  5. Write `{run_id}.json` to `output-dir`.
- Failed runs write `{run_id}.failed.json` with the exception trace and
  agent config; aggregator skips these but counts them in summary.

### GPU / CPU detection

- Harness reads `torch.cuda.is_available()` at startup. If `False` and
  any agent requires CUDA, harness either skips those agents (default)
  or runs them on CPU (`--cpu-fallback`).
- Harness reads `nvidia-smi --query-gpu=memory.free --format=csv,nounits`
  before launching DL runs. If free < 4 GB, abort with a clear error.
  Echoes the paper2/bet1_surprise contention pattern from the previous
  batch.

### Aggregator

`rl_recsys/training/results_aggregator.py`:

- `aggregate(results_dir: Path) -> pd.DataFrame`: reads all `*.json`
  (skipping `*.failed.json`), one row per run, columns include all
  metrics and the run config.
- `to_summary_md(df) -> str`: pivot table with rows = agent, columns =
  pretrained ∈ {True, False}, cells = `{avg_seq_dr_value:.3f} ±
  {std_seq_dr_value:.3f}`. Sorted by `pretrained=True` cell descending.
- `to_summary_csv(df, path)`: full long-form CSV.
- CLI: `python -m rl_recsys.training.results_aggregator
  results/agent_grid/<dir>` writes `summary.csv` and `summary.md` next
  to the input.

### Configuration

Each agent gets a default config block in YAML at
`configs/agents/<agent_name>.yaml` (~10 lines each). Harness loads
these by default; CLI flag `--agent-config-dir` overrides.

`rl_recsys.config.AgentConfig` extends with optional fields used by
specific agents (`alpha`, `epsilon`, `temperature`, `sigma`,
`hidden_dim`, `epochs`, `embedding_dim`, `n_estimators`, `n_heads`,
`n_blocks`, `max_history_len`, `clip_c`, `target_return`). Fields
ignored by agents that don't consume them. No agent reads dataset-
specific config.

### Run artifact format

`results/agent_grid/<dir>/{run_id}.json`:

```json
{
  "agent": "linucb",
  "seed": 0,
  "pretrained": true,
  "config": {"alpha": 1.0, "boltzmann_T": 1.0},
  "metrics": {
    "avg_seq_dr_value": 10.612,
    "std_seq_dr_value": 0.094,
    "avg_logged_discounted_return": 10.572,
    "std_logged_discounted_return": 0.082,
    "n_eval_trajectories": 5000,
    "wall_seconds": 312.4,
    "train_seconds": 41.7
  },
  "git_sha": "be83da2",
  "timestamp": "2026-05-09T17:42:11Z"
}
```

## Dataset-agnostic invariants

- No agent imports `rl_recsys.data.loaders.rl4rs_*` or any dataset-
  specific module. CI lint check via `tests/test_agents_dataset_agnostic.py`
  scans agent modules' import graphs.
- All per-step inputs flow through `RecObs`. All offline training uses
  the abstract `LoggedTrajectorySource` interface.
- All slate-size, user-dim, item-dim, num-candidates parameters come
  from `EnvConfig` at agent construction. No agent hardcodes 9, 283,
  or any dataset-specific magic number.

## Error handling

- `LoggedReplayAgent.select_slate` raises `ValueError("LoggedReplayAgent
  requires replay-mode source — obs.logged_action is None")` when called
  without replay context.
- `OracleClickAgent.select_slate` raises the analogous error if either
  `logged_action` or `logged_clicks` is `None`.
- DL agents raise `RuntimeError("CUDA required for {AgentName}; pass
  --cpu-fallback to run on CPU")` at construction if `cuda` requested
  but unavailable.
- Harness aborts with `RuntimeError` if `LoggedReplayAgent` is paired
  with a non-replay source.
- GBDT raises `ImportError` with a clear "pip install lightgbm" message
  if the optional dep is missing.
- `RecObs.history` is always a tuple (possibly empty). Agents must not
  assume non-emptiness. SASRec / TopK / DT all treat empty history via
  a learned sentinel.

## Testing

### Per-agent unit tests

`tests/test_agents_heuristic.py`:
- `test_most_popular_ranks_by_train_clicks`
- `test_logged_replay_returns_logged_slate`
- `test_logged_replay_raises_without_logged_action`
- `test_oracle_click_picks_clicked_items_first`
- `test_oracle_click_pads_with_nonlogged_when_underclicked`

`tests/test_agents_linear_bandit.py`:
- `test_lints_sample_is_deterministic_with_rng`
- `test_eps_greedy_explores_at_eps_one`
- `test_eps_greedy_exploits_at_eps_zero`
- `test_boltzmann_collapses_to_argmax_at_low_T`
- `test_linear_bandit_base_update_matches_linucb`

`tests/test_agents_neural.py`:
- `test_neural_linear_train_offline_completes_on_tiny_dataset`
- `test_neural_linear_score_shape`
- `test_neural_linear_alpha_zero_drops_to_point_estimate`

`tests/test_agents_imitation.py`:
- `test_bc_score_matches_behavior_policy_softmax_sum`
- `test_bc_train_offline_calls_fit_when_no_policy_provided`
- `test_gbdt_train_offline_returns_metrics`
- `test_gbdt_score_shape`
- `test_gbdt_raises_clear_error_when_lightgbm_missing`

`tests/test_agents_dl.py`:
- `test_sasrec_forward_shape`
- `test_sasrec_handles_empty_history_via_sentinel`
- `test_topk_reinforce_loss_decreases_on_overfit_batch`
- `test_topk_reinforce_uses_precomputed_propensities`
- `test_dt_decode_top_k_returns_unique_items`
- `test_dt_handles_empty_history_via_sentinel`

All DL tests run on CPU with tiny configs (`hidden_dim=8`, 1 epoch, 4
sessions) to keep CI fast.

### Harness / aggregator tests

`tests/test_agent_grid_runner.py`:
- `test_runner_writes_one_artifact_per_run`
- `test_runner_resume_skips_existing_artifacts`
- `test_runner_failed_run_writes_failed_json_and_continues`
- `test_runner_skips_dl_agents_without_cuda_unless_fallback`
- `test_runner_validates_replay_mode_for_logged_replay_agent`

`tests/test_results_aggregator.py`:
- `test_aggregator_pivot_shape`
- `test_aggregator_skips_failed_json`
- `test_aggregator_summary_md_renders_pretrained_columns`

### `RecObs` / loader tests

`tests/test_recobs.py` (new file):
- `test_recobs_history_default_empty_tuple`
- `test_recobs_legacy_constructor_still_works`
- `test_history_step_shape_validation`

`tests/test_rl4rs_trajectory_ope.py` (extend):
- `test_loader_history_accumulates_within_session`
- `test_loader_history_resets_between_sessions`
- `test_loader_obs_logged_action_matches_step_logged_action`

### Dataset-agnostic lint

`tests/test_agents_dataset_agnostic.py`:
- Walks every module under `rl_recsys.agents`, parses imports via `ast`,
  asserts no module imports anything matching
  `rl_recsys.data.loaders.*` or `rl_recsys.environments.rl4rs_*` /
  `rl_recsys.environments.kuairec_*` etc.

### End-to-end smoke

`tests/test_agent_grid_smoke.py`:
- Synthetic 50-session parquet on disk, `slate_size=3`,
  `num_candidates=20`.
- Run harness with `--agents random,linucb,bc --seeds 0
  --pretrained-modes true --max-trajectories 50 --output-dir
  <tmpdir>`.
- Assert 3 JSON artifacts written, all with finite `avg_seq_dr_value`,
  and `summary.md` exists with 3 rows × 1 col.

### Test count

Current 197 tests + ~45 new = ~242. CPU-only; no CUDA required for any
test in the suite.

## Acceptance

- All tests pass on CPU.
- `python scripts/benchmark_agent_grid.py --dataset rl4rs_b --agents all
  --seeds 0,1,2 --pretrained-modes both --max-trajectories 5000` runs
  end-to-end and produces a `summary.md` with 14 rows × 2 columns.
  (`OracleClickAgent` opt-in only; if added, 15 rows.)
- Sanity checks pass on the produced grid:
  - `LoggedReplayAgent` row matches `avg_logged_discounted_return` to
    within IS noise (< 1% relative).
  - `MostPopularAgent` and `RandomAgent` rows lie below the linear
    bandit family.
  - `OracleClickAgent` (when run) strictly dominates every other row.
- `summary.md` is checked into `results/agent_grid/2026-05-09/` so
  reviewers can read the answer without re-running the grid.
- The harness is restartable via `--resume`; partial failures don't
  invalidate the rest of the grid.

## Risk / open questions

- **TopK and DT may produce noisy or weak results** until full state
  evolution lands. The spec calls this out in the results table; the
  acceptance check does not require these agents to outperform any
  baseline. If they look broken (loss not decreasing), we flag it
  rather than chase it within this batch.
- **GBDT adds an optional dependency** (`lightgbm`). Falling back to
  XGBoost is a swap if `lightgbm` doesn't pin cleanly with our
  `requirements.txt` numpy/torch versions.
- **GPU contention** could block DL agents (paper2 has been holding
  the RTX 5080 in past batches). The harness's `--resume` flag and
  preflight `nvidia-smi` check minimize the cost when this happens —
  CPU-trainable agents complete; DL runs error cleanly and can be
  retried later.
- **Boltzmann temperature is fixed at T=1.0**. The "Cause #3" from
  TODO.md (T sweep) is intentionally deferred to a follow-up batch
  since core grid shipping speed matters more here.

## Out of scope

- Refactoring `LinUCBAgent` onto the `LinearBanditBase` mixin.
- Sharing a `SequenceEncoder` module across SASRec / TopK / DT. Their
  training objectives differ; abstracting prematurely would lock in a
  shape we may regret. Extract later if patterns repeat.
- Per-agent hyperparam sweep YAML schemas (one config per agent is
  enough until we know which agents win).
- Multi-dataset comparison plots — sub-projects #2–#4 produce per-
  dataset grids; cross-dataset visualization gets its own follow-up.

## Decomposition map (for context only)

This spec is sub-project #1 of:

1. **Multi-Agent Ablation Grid on RL4RS-B** ← this spec.
2. KuaiRec Sequential DR pipeline (uses #1's harness as-is, adds
   loader + behavior policy).
3. finn-no-slate Sequential DR pipeline (same shape).
4. MovieLens Sequential DR pipeline (same shape).
5. Trained DM reward model for DR (cross-cutting, unblocks DR's
   variance reduction across datasets).

Each sub-project gets its own design + plan + implementation cycle.
