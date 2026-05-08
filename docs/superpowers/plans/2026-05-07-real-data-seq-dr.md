# Real-Data Sequential DR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `seq_dr_value` to RL4RS dataset B end-to-end: extend the pipeline to emit a multi-step parquet, fit an estimated behavior policy, and report Sequential DR with variance for our agents.

**Architecture:** Slate-as-action with factorized propensity. A small per-position softmax MLP estimates `μ`. A Boltzmann-temperature shim turns deterministic agents (`LinUCB`, `Random`) into stochastic targets so `π/μ` ratios are well-defined. Existing `seq_dr_value` formula is unchanged — we just feed it real per-step propensity / target_prob arrays.

**Tech Stack:** Python 3.12, NumPy, pandas, pyarrow, PyTorch, pytest. Project venv at `.venv/`.

---

## Conventions

- **Run tests with `.venv/bin/python -m pytest ...`** (per repo CLAUDE.md memory). Never use system `python3`/`pip3`.
- **One commit per task.** Co-author footer matches existing commits: `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.
- **TDD strictly:** write failing test → run to confirm fail → implement → run to confirm pass → commit.
- **Branch:** the work happens on `main` (this repo's flow), or in a worktree if the executor prefers — both are fine.
- **Spec reference:** `docs/superpowers/specs/2026-05-07-real-data-seq-dr-design.md`.

## File map

| File | Action | Touched by task |
|---|---|---|
| `rl_recsys/agents/linucb.py` | rename method | T1 |
| `rl_recsys/agents/random.py` | add method | T1 |
| `tests/test_agents.py` | update + add tests | T1 |
| `rl_recsys/evaluation/ope_trajectory.py` | rewrite `_target_probability`, retype `LoggedTrajectoryStep`, add `temperature`, contract check, ESS | T2, T3 |
| `tests/test_ope_trajectory.py` | update existing tests + add Boltzmann + ESS tests | T2, T3 |
| `rl_recsys/data/schema.py` | add `rl_sessions_b` schema | T4 |
| `tests/test_schema.py` | new file (or extend existing) | T4 |
| `rl_recsys/data/pipelines/rl4rs.py` | add `process_b()`, register `rl4rs_b` | T5 |
| `tests/test_rl4rs_pipeline.py` | new file | T5 |
| `rl_recsys/evaluation/behavior_policy.py` | new module: model, fit, calibration | T6, T7, T8 |
| `tests/test_behavior_policy.py` | new file | T6, T7, T8 |
| `rl_recsys/data/loaders/rl4rs_trajectory_ope.py` | new loader | T9 |
| `tests/test_rl4rs_trajectory_ope.py` | new file | T9 |
| `rl_recsys/evaluation/variance.py` | add `evaluate_trajectory_ope_with_variance` | T10 |
| `tests/test_variance.py` | add smoke test | T10 |
| `scripts/benchmark_rl4rs_b_seq_dr.py` | new script | T11 |
| `rl_recsys/evaluation/__init__.py` | export new variance wrapper | T10 |
| `TODO.md` | remove "Real-data Sequential DR" entry from Next up | T11 |

---

## Task 1: Add `score_items` agent contract

**Files:**
- Modify: `rl_recsys/agents/linucb.py:61` (rename `score_candidates` → `score_items`)
- Modify: `rl_recsys/agents/linucb.py:34` (call site for the rename)
- Modify: `rl_recsys/agents/random.py` (add `score_items` method)
- Modify: `tests/test_agents.py:68,77` (call site updates)
- Modify: `tests/test_agents.py` (add RandomAgent.score_items test)

- [ ] **Step 1: Write failing test for RandomAgent.score_items**

Add to `tests/test_agents.py`:

```python
def test_random_agent_score_items_is_uniform() -> None:
    agent = RandomAgent(slate_size=2)
    scores = agent.score_items(_obs())

    assert scores.shape == (3,)
    assert np.allclose(scores, scores[0])  # all equal → uniform softmax
```

- [ ] **Step 2: Run to confirm fail**

Run: `.venv/bin/python -m pytest tests/test_agents.py::test_random_agent_score_items_is_uniform -v`
Expected: FAIL with `AttributeError: 'RandomAgent' object has no attribute 'score_items'`

- [ ] **Step 3: Add `score_items` to RandomAgent**

Modify `rl_recsys/agents/random.py`. Add this method to the class (above `update`):

```python
    def score_items(self, obs: RecObs) -> np.ndarray:
        """Uniform scores → softmax produces uniform 1/n distribution."""
        return np.zeros(len(obs.candidate_features), dtype=np.float64)
```

- [ ] **Step 4: Run to confirm pass**

Run: `.venv/bin/python -m pytest tests/test_agents.py::test_random_agent_score_items_is_uniform -v`
Expected: PASS

- [ ] **Step 5: Rename LinUCB.score_candidates → score_items**

In `rl_recsys/agents/linucb.py`:

Line 34 — change:
```python
        scores = self.score_candidates(obs)
```
to:
```python
        scores = self.score_items(obs)
```

Line 61 — change:
```python
    def score_candidates(self, obs: RecObs) -> np.ndarray:
        """Return LinUCB scores for every candidate in the observation."""
```
to:
```python
    def score_items(self, obs: RecObs) -> np.ndarray:
        """Return LinUCB UCB scores for every candidate in the observation."""
```

- [ ] **Step 6: Update test_agents.py call sites**

In `tests/test_agents.py` lines 68 and 77, change `agent.score_candidates(obs)` → `agent.score_items(obs)` (both call sites).

- [ ] **Step 7: Run full agents test file to confirm green**

Run: `.venv/bin/python -m pytest tests/test_agents.py -v`
Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add rl_recsys/agents/linucb.py rl_recsys/agents/random.py tests/test_agents.py
git commit -m "$(cat <<'EOF'
feat: add score_items contract to LinUCB and Random agents

Renames LinUCBAgent.score_candidates to score_items (the canonical
contract name used by the upcoming Boltzmann target-probability shim
in sequential OPE). Adds score_items to RandomAgent returning zeros so
softmax(scores/T) yields a uniform distribution over candidates.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Boltzmann `_target_probability` + slate-action `LoggedTrajectoryStep`

This task makes the breaking change to the public dataclass shipped earlier and rewrites the target-probability function for slate-as-action. After this task, every existing `evaluate_trajectory_ope_agent_*` test must still pass under the new shape.

**Files:**
- Modify: `rl_recsys/evaluation/ope_trajectory.py` (`LoggedTrajectoryStep`, `_target_probability`, `evaluate_trajectory_ope_agent`)
- Modify: `tests/test_ope_trajectory.py` (update fixtures + existing tests, add Boltzmann test)

- [ ] **Step 1: Write failing test for the new Boltzmann target_prob signature**

Add to `tests/test_ope_trajectory.py`:

```python
def test_target_probability_boltzmann_factorized_over_slate() -> None:
    from rl_recsys.evaluation.ope_trajectory import _target_probability

    class _ScoredAgent:
        def score_items(self, obs):
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        def select_slate(self, obs):
            return np.array([0, 1], dtype=np.int64)

    obs = _make_obs(num_candidates=4)
    agent = _ScoredAgent()
    # softmax([1,0,0,0]/1.0) = [e/(e+3), 1/(e+3), 1/(e+3), 1/(e+3)]
    e = float(np.e)
    p0 = e / (e + 3.0)
    p1 = 1.0 / (e + 3.0)
    # logged_slate=[0,1] → target prob = p0 * p1
    expected = p0 * p1

    result = _target_probability(
        agent, obs,
        agent_slate=np.array([0, 1], dtype=np.int64),
        logged_slate=np.array([0, 1], dtype=np.int64),
        temperature=1.0,
    )

    assert result == pytest.approx(expected)


def test_target_probability_raises_on_nonpositive_temperature() -> None:
    from rl_recsys.evaluation.ope_trajectory import _target_probability

    class _ScoredAgent:
        def score_items(self, obs):
            return np.zeros(4, dtype=np.float64)

    obs = _make_obs(num_candidates=4)
    with pytest.raises(ValueError, match="temperature"):
        _target_probability(
            _ScoredAgent(), obs,
            agent_slate=np.array([0], dtype=np.int64),
            logged_slate=np.array([0], dtype=np.int64),
            temperature=0.0,
        )


def test_evaluate_trajectory_ope_agent_raises_when_agent_lacks_score_items() -> None:
    class _NoScoresAgent:
        def select_slate(self, obs):
            return np.array([0], dtype=np.int64)

    obs = _make_obs(num_candidates=4)
    step = LoggedTrajectoryStep(
        obs=obs,
        logged_action=np.array([0], dtype=np.int64),
        logged_reward=1.0,
        propensity=0.5,
    )
    source = _SyntheticTrajectorySource(trajectories=[[step]])

    with pytest.raises(AttributeError, match="score_items"):
        evaluate_trajectory_ope_agent(
            source, _NoScoresAgent(), agent_name="x",
            max_trajectories=1, seed=0,
        )
```

- [ ] **Step 2: Run to confirm fail**

Run: `.venv/bin/python -m pytest tests/test_ope_trajectory.py::test_target_probability_boltzmann_factorized_over_slate -v`
Expected: FAIL — signature mismatch (current `_target_probability` takes `top_action: int, logged_action: int`).

- [ ] **Step 3: Rewrite `LoggedTrajectoryStep` and `_target_probability`**

In `rl_recsys/evaluation/ope_trajectory.py`:

Replace the `LoggedTrajectoryStep` dataclass:

```python
@dataclass(frozen=True)
class LoggedTrajectoryStep:
    obs: RecObs
    logged_action: np.ndarray  # shape (slate_size,) — the logged slate, item ids
    logged_reward: float
    propensity: float          # μ(slate | obs) = Π_k μ(slate[k] | obs, k)
```

Replace `_target_probability` with:

```python
def _target_probability(
    agent: Agent,
    obs: RecObs,
    agent_slate: np.ndarray,
    logged_slate: np.ndarray,
    *,
    temperature: float,
) -> float:
    """π_target(logged_slate | obs) under a temperature-softmax shim of agent.

    π(item_k | obs) = softmax(agent.score_items(obs) / T)[item_k]
    π(slate | obs) = Π_k π(slate[k] | obs)
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    scores = np.asarray(agent.score_items(obs), dtype=np.float64)
    # Numerically stable softmax: subtract max before exp.
    z = scores / temperature
    z = z - z.max()
    exp_z = np.exp(z)
    probs = exp_z / exp_z.sum()
    logged = np.asarray(logged_slate, dtype=np.int64)
    return float(np.prod(probs[logged]))
```

- [ ] **Step 4: Update `evaluate_trajectory_ope_agent` signature and body**

In `rl_recsys/evaluation/ope_trajectory.py`, replace the function body:

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
    temperature: float = 1.0,
) -> TrajectoryOPEEvaluation:
    """Sequential DR off-policy evaluator with slate-as-action.

    For each trajectory, the agent picks a slate per step. Per-step target
    probability uses a Boltzmann shim over agent.score_items: π(slate | obs) =
    Π_k softmax(scores / T)[slate[k]]. agent.update() is NOT called.

    Empty trajectories yielded by ``source`` are silently skipped — they have
    no steps to score and would make ``seq_dr_value`` raise. If every yielded
    trajectory is empty (or ``source`` yields nothing), the run raises
    ``ValueError`` because the resulting averages would be undefined.
    """
    if max_trajectories <= 0:
        raise ValueError("max_trajectories must be positive")
    if not hasattr(agent, "score_items"):
        raise AttributeError(
            f"agent {agent_name!r} must implement score_items(obs) -> np.ndarray "
            "for slate OPE under the Boltzmann target-probability shim"
        )

    started = perf_counter()
    seq_dr_per_traj: list[float] = []
    logged_returns: list[float] = []
    total_steps = 0

    for traj in source.iter_trajectories(max_trajectories=max_trajectories, seed=seed):
        if not traj:
            continue
        rewards: list[float] = []
        target_probs: list[float] = []
        propensities: list[float] = []
        for step in traj:
            agent_slate = np.asarray(agent.select_slate(step.obs), dtype=np.int64)
            if len(agent_slate) == 0:
                raise ValueError("agent returned an empty slate")
            target_probs.append(
                _target_probability(
                    agent, step.obs,
                    agent_slate=agent_slate,
                    logged_slate=np.asarray(step.logged_action, dtype=np.int64),
                    temperature=temperature,
                )
            )
            rewards.append(float(step.logged_reward))
            propensities.append(float(step.propensity))
        rewards_arr = np.asarray(rewards, dtype=np.float64)
        target_arr = np.asarray(target_probs, dtype=np.float64)
        prop_arr = np.asarray(propensities, dtype=np.float64)
        seq_dr_per_traj.append(
            seq_dr_value(
                rewards_arr, target_arr, prop_arr,
                gamma=gamma, reward_model=reward_model, clip=clip,
            )
        )
        logged_returns.append(discounted_return(rewards_arr, gamma=gamma))
        total_steps += len(traj)

    n = len(seq_dr_per_traj)
    if n == 0:
        raise ValueError("source produced zero trajectories")

    return TrajectoryOPEEvaluation(
        agent=agent_name,
        trajectories=n,
        total_steps=total_steps,
        avg_seq_dr_value=float(np.mean(seq_dr_per_traj)),
        avg_logged_discounted_return=float(np.mean(logged_returns)),
        seconds=float(perf_counter() - started),
    )
```

Also delete the `RandomAgent` import from this file (no longer used in `_target_probability`).

- [ ] **Step 5: Update existing tests for `logged_action: ndarray` shape**

In `tests/test_ope_trajectory.py`:

- Update `_DetAgent` to expose `score_items`:

```python
class _DetAgent:
    """Always picks slate=[0] — top action is candidate index 0.

    score_items returns a flat-zero score so target probability under the
    Boltzmann shim is uniform 1/num_candidates per position. This keeps
    expected values hand-computable in aggregation tests.
    """

    def __init__(self, slate_size: int = 1) -> None:
        self._slate_size = slate_size

    def select_slate(self, obs: RecObs) -> np.ndarray:
        return np.arange(self._slate_size, dtype=np.int64)

    def score_items(self, obs: RecObs) -> np.ndarray:
        return np.zeros(len(obs.candidate_features), dtype=np.float64)

    def update(self, obs, slate, reward, clicks, next_obs):
        return {}
```

- Update `test_evaluate_trajectory_ope_agent_aggregates_per_trajectory` body. Replace the entire test with:

```python
def test_evaluate_trajectory_ope_agent_aggregates_per_trajectory() -> None:
    # _DetAgent (flat scores) under T=1.0 → softmax uniform = 1/4 per position.
    # slate_size=1 → π(slate | obs) = 0.25 every step.
    # Trajectory A (2 steps): rewards=[1, 0], propensity=[0.5, 0.5]
    #   w = clip(0.25/0.5) = 0.5; W = [0.5, 0.25]; b = mean([1,0]) = 0.5
    #   V_A(γ=0.9) = 1*(0.5*0.5+0.5) + 0.9*(0.25*-0.5+0.5) = 0.75 + 0.9*0.375 = 1.0875
    # Trajectory B (1 step): rewards=[2], propensity=[1.0]
    #   w = clip(0.25/1.0) = 0.25; b = 2.0
    #   V_B = 1*(0.25*0+2) = 2.0
    obs = _make_obs(num_candidates=4)
    traj_a = [
        LoggedTrajectoryStep(
            obs=obs, logged_action=np.array([0], dtype=np.int64),
            logged_reward=1.0, propensity=0.5,
        ),
        LoggedTrajectoryStep(
            obs=obs, logged_action=np.array([0], dtype=np.int64),
            logged_reward=0.0, propensity=0.5,
        ),
    ]
    traj_b = [
        LoggedTrajectoryStep(
            obs=obs, logged_action=np.array([0], dtype=np.int64),
            logged_reward=2.0, propensity=1.0,
        ),
    ]
    source = _SyntheticTrajectorySource(trajectories=[traj_a, traj_b])
    agent = _DetAgent(slate_size=1)

    result = evaluate_trajectory_ope_agent(
        source, agent, agent_name="det",
        max_trajectories=2, seed=0, gamma=0.9, temperature=1.0,
    )

    assert isinstance(result, TrajectoryOPEEvaluation)
    assert result.trajectories == 2
    assert result.total_steps == 3
    assert result.avg_seq_dr_value == pytest.approx((1.0875 + 2.0) / 2)
    assert result.avg_logged_discounted_return == pytest.approx((1.0 + 2.0) / 2)
```

- Update `test_evaluate_trajectory_ope_agent_uses_uniform_target_prob_for_random_agent` — replace the LoggedTrajectoryStep constructions with `logged_action=np.array([0], dtype=np.int64)` instead of `logged_action=0`.

- Update `test_evaluate_trajectory_ope_agent_does_not_mutate_agent_state` — change `logged_action=0` → `logged_action=np.array([0], dtype=np.int64)`.

- Update `test_evaluate_trajectory_ope_agent_raises_when_all_trajectories_empty` — no LoggedTrajectoryStep constructions, only empty lists, no change needed.

- Update `test_evaluate_trajectory_ope_agent_raises_on_nonpositive_max_trajectories` — no LoggedTrajectoryStep constructions, no change needed.

- [ ] **Step 6: Run full ope_trajectory tests**

Run: `.venv/bin/python -m pytest tests/test_ope_trajectory.py -v`
Expected: all tests pass (12 tests now: original 10 + 2 new Boltzmann tests + 1 contract test = 13 tests).

- [ ] **Step 7: Run full suite to confirm nothing else regressed**

Run: `.venv/bin/python -m pytest -q`
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add rl_recsys/evaluation/ope_trajectory.py tests/test_ope_trajectory.py
git commit -m "$(cat <<'EOF'
feat: slate-action LoggedTrajectoryStep + Boltzmann target probability

Breaking change to the public LoggedTrajectoryStep dataclass shipped in
the previous batch: logged_action is now a slate ndarray, and target
probability is computed as a Boltzmann shim over agent.score_items
factorized across slate positions. evaluate_trajectory_ope_agent gains
a temperature parameter (default 1.0) and a score_items contract check.

This is the data-shape change required to drive the estimator from
real RL4RS slate logs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: ESS reporting in `evaluate_trajectory_ope_agent`

Adds effective sample size as a non-aggregating field on `TrajectoryOPEEvaluation`. Logs a warning if ESS / total_steps < 0.01.

**Files:**
- Modify: `rl_recsys/evaluation/ope_trajectory.py` (compute ESS, add field, warn)
- Modify: `tests/test_ope_trajectory.py` (test ESS computation + warning)

- [ ] **Step 1: Write failing test for ESS computation**

Add to `tests/test_ope_trajectory.py`:

```python
def test_evaluate_trajectory_ope_agent_reports_ess() -> None:
    # _DetAgent flat scores → π = 0.25 per step at T=1, slate_size=1.
    # propensity=0.5 → w=0.5 (clipped within [0.1,10]). With 4 such steps,
    # W per step is cumprod within trajectory. Per-trajectory ESS contribution
    # is computed by the implementation; we just verify ess is finite, in
    # (0, total_steps], and equals (sum W)^2 / sum W^2 across all step weights.
    obs = _make_obs(num_candidates=4)
    traj = [
        LoggedTrajectoryStep(
            obs=obs, logged_action=np.array([0], dtype=np.int64),
            logged_reward=1.0, propensity=0.5,
        )
        for _ in range(4)
    ]
    source = _SyntheticTrajectorySource(trajectories=[traj])
    agent = _DetAgent(slate_size=1)

    result = evaluate_trajectory_ope_agent(
        source, agent, agent_name="det",
        max_trajectories=1, seed=0, gamma=0.9, temperature=1.0,
    )

    # W within trajectory = cumprod([0.5,0.5,0.5,0.5]) = [0.5, 0.25, 0.125, 0.0625]
    expected_ess = (0.5 + 0.25 + 0.125 + 0.0625) ** 2 / (
        0.5**2 + 0.25**2 + 0.125**2 + 0.0625**2
    )
    assert result.ess == pytest.approx(expected_ess)
    assert 0.0 < result.ess <= result.total_steps
```

- [ ] **Step 2: Run to confirm fail**

Run: `.venv/bin/python -m pytest tests/test_ope_trajectory.py::test_evaluate_trajectory_ope_agent_reports_ess -v`
Expected: FAIL with `AttributeError: 'TrajectoryOPEEvaluation' object has no attribute 'ess'`.

- [ ] **Step 3: Add ess field and computation**

In `rl_recsys/evaluation/ope_trajectory.py`:

Update `TrajectoryOPEEvaluation`:

```python
@dataclass
class TrajectoryOPEEvaluation:
    agent: str
    trajectories: int = field(metadata={"aggregate": False})
    total_steps: int = field(metadata={"aggregate": False})
    avg_seq_dr_value: float
    avg_logged_discounted_return: float
    ess: float = field(metadata={"aggregate": False})
    seconds: float = field(metadata={"aggregate": False})

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "agent": self.agent,
            "trajectories": self.trajectories,
            "total_steps": self.total_steps,
            "avg_seq_dr_value": self.avg_seq_dr_value,
            "avg_logged_discounted_return": self.avg_logged_discounted_return,
            "ess": self.ess,
            "seconds": self.seconds,
        }
```

In `evaluate_trajectory_ope_agent`, accumulate weights across steps and compute ESS at the end. Replace the function body's per-trajectory loop with:

```python
    started = perf_counter()
    seq_dr_per_traj: list[float] = []
    logged_returns: list[float] = []
    total_steps = 0
    all_weights: list[np.ndarray] = []

    for traj in source.iter_trajectories(max_trajectories=max_trajectories, seed=seed):
        if not traj:
            continue
        rewards: list[float] = []
        target_probs: list[float] = []
        propensities: list[float] = []
        for step in traj:
            agent_slate = np.asarray(agent.select_slate(step.obs), dtype=np.int64)
            if len(agent_slate) == 0:
                raise ValueError("agent returned an empty slate")
            target_probs.append(
                _target_probability(
                    agent, step.obs,
                    agent_slate=agent_slate,
                    logged_slate=np.asarray(step.logged_action, dtype=np.int64),
                    temperature=temperature,
                )
            )
            rewards.append(float(step.logged_reward))
            propensities.append(float(step.propensity))
        rewards_arr = np.asarray(rewards, dtype=np.float64)
        target_arr = np.asarray(target_probs, dtype=np.float64)
        prop_arr = np.asarray(propensities, dtype=np.float64)
        ratios = np.clip(target_arr / prop_arr, clip[0], clip[1])
        cum_weights = np.cumprod(ratios)
        all_weights.append(cum_weights)
        seq_dr_per_traj.append(
            seq_dr_value(
                rewards_arr, target_arr, prop_arr,
                gamma=gamma, reward_model=reward_model, clip=clip,
            )
        )
        logged_returns.append(discounted_return(rewards_arr, gamma=gamma))
        total_steps += len(traj)

    n = len(seq_dr_per_traj)
    if n == 0:
        raise ValueError("source produced zero trajectories")

    weights = np.concatenate(all_weights)
    ess = float((weights.sum() ** 2) / (np.sum(weights ** 2)))
    if ess / total_steps < 0.01:
        import warnings
        warnings.warn(
            f"effective sample size {ess:.2f} is < 1% of total steps "
            f"({total_steps}); estimator may be unreliable",
            stacklevel=2,
        )

    return TrajectoryOPEEvaluation(
        agent=agent_name,
        trajectories=n,
        total_steps=total_steps,
        avg_seq_dr_value=float(np.mean(seq_dr_per_traj)),
        avg_logged_discounted_return=float(np.mean(logged_returns)),
        ess=ess,
        seconds=float(perf_counter() - started),
    )
```

- [ ] **Step 4: Run new test to confirm pass**

Run: `.venv/bin/python -m pytest tests/test_ope_trajectory.py::test_evaluate_trajectory_ope_agent_reports_ess -v`
Expected: PASS.

- [ ] **Step 5: Run full ope_trajectory file (other tests reference TrajectoryOPEEvaluation)**

Run: `.venv/bin/python -m pytest tests/test_ope_trajectory.py -v`
Expected: all pass. (Existing tests don't reference `ess` so they're unaffected.)

- [ ] **Step 6: Run full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add rl_recsys/evaluation/ope_trajectory.py tests/test_ope_trajectory.py
git commit -m "$(cat <<'EOF'
feat: report effective sample size from sequential DR evaluator

Adds ess (= (Σ W)^2 / Σ W^2 across all evaluated step weights) as a
non-aggregating field on TrajectoryOPEEvaluation. Warns when
ess / total_steps < 0.01 to flag runs where the estimator is dominated
by a tiny number of trajectories and likely unreliable.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add `rl_sessions_b` schema

**Files:**
- Modify: `rl_recsys/data/schema.py` (add to `REQUIRED_COLUMNS`)
- Create: `tests/test_schema.py` (if not present; otherwise extend)

- [ ] **Step 1: Check whether `tests/test_schema.py` exists**

Run: `ls tests/test_schema.py 2>/dev/null`
- If file exists: extend it.
- If not: create it.

- [ ] **Step 2: Write failing test**

Create or append to `tests/test_schema.py`:

```python
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rl_recsys.data.schema import validate_parquet_schema


def _write_b_fixture(path: Path, *, missing: str | None = None) -> None:
    columns = {
        "session_id": [1, 1],
        "sequence_id": [1, 2],
        "user_state": [[1.0], [1.0]],
        "slate": [[10, 11], [12, 13]],
        "item_features": [[[0.0], [1.0]], [[0.5], [0.7]]],
        "user_feedback": [[1, 0], [0, 1]],
        "candidate_ids": [[10, 11, 12, 13], [10, 11, 12, 13]],
        "candidate_features": [
            [[0.0], [1.0], [0.5], [0.7]],
            [[0.0], [1.0], [0.5], [0.7]],
        ],
    }
    if missing:
        del columns[missing]
    pd.DataFrame(columns).to_parquet(path, index=False)


def test_validate_parquet_schema_accepts_rl_sessions_b(tmp_path: Path) -> None:
    p = tmp_path / "sessions_b.parquet"
    _write_b_fixture(p)
    validate_parquet_schema(p, "rl_sessions_b")  # must not raise


def test_validate_parquet_schema_rejects_rl_sessions_b_missing_candidate_ids(
    tmp_path: Path,
) -> None:
    p = tmp_path / "sessions_b.parquet"
    _write_b_fixture(p, missing="candidate_ids")
    with pytest.raises(ValueError, match="candidate_ids"):
        validate_parquet_schema(p, "rl_sessions_b")
```

- [ ] **Step 3: Run to confirm fail**

Run: `.venv/bin/python -m pytest tests/test_schema.py -v`
Expected: FAIL with `Unknown schema type 'rl_sessions_b'`.

- [ ] **Step 4: Add schema entry**

In `rl_recsys/data/schema.py`, add to `REQUIRED_COLUMNS`:

```python
    "rl_sessions_b": {
        "session_id", "sequence_id", "user_state", "slate", "item_features",
        "user_feedback", "candidate_ids", "candidate_features",
    },
```

- [ ] **Step 5: Run to confirm pass**

Run: `.venv/bin/python -m pytest tests/test_schema.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add rl_recsys/data/schema.py tests/test_schema.py
git commit -m "$(cat <<'EOF'
feat: add rl_sessions_b schema for multi-step RL4RS sessions

Defines required columns for the dataset-B parquet that the next task
will start writing: adds candidate_ids and candidate_features alongside
slate / user_feedback so loaders have a fixed candidate universe per
step without needing to materialise it on the fly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `RL4RSPipeline.process_b()`

Reads `rl4rs_dataset_b_rl.csv`, groups rows by `(session_id, sequence_id)` ordered by `sequence_id`, builds a fixed candidate universe (all unique items from the slate column), and writes `sessions_b.parquet`.

**Files:**
- Modify: `rl_recsys/data/pipelines/rl4rs.py` (add `process_b`, register `rl4rs_b` target)
- Create: `tests/test_rl4rs_pipeline.py`

- [ ] **Step 1: Write failing test using a tiny in-memory CSV fixture**

Create `tests/test_rl4rs_pipeline.py`:

```python
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from rl_recsys.data.pipelines.rl4rs import RL4RSPipeline


def _write_b_fixture_csv(raw_dir: Path) -> None:
    """Mimics rl4rs_dataset_b_rl.csv with two sessions."""
    csv_path = raw_dir / "rl4rs-dataset" / "rl4rs_dataset_b_rl.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        # session 1, two steps
        {"timestamp": 1, "session_id": 1, "sequence_id": 1,
         "exposed_items": "10,11,12", "user_feedback": "1,0,0",
         "user_seqfeature": "1,2", "user_protrait": "0.1,0.2",
         "item_feature": "0.0,1.0;0.1,0.9;0.2,0.8",
         "behavior_policy_id": 1},
        {"timestamp": 2, "session_id": 1, "sequence_id": 2,
         "exposed_items": "11,13,14", "user_feedback": "0,1,1",
         "user_seqfeature": "1,2,3", "user_protrait": "0.1,0.2",
         "item_feature": "0.1,0.9;0.3,0.7;0.4,0.6",
         "behavior_policy_id": 1},
        # session 2, one step
        {"timestamp": 3, "session_id": 2, "sequence_id": 1,
         "exposed_items": "10,12,15", "user_feedback": "0,0,0",
         "user_seqfeature": "5", "user_protrait": "0.4,0.5",
         "item_feature": "0.0,1.0;0.2,0.8;0.5,0.5",
         "behavior_policy_id": 1},
    ]
    pd.DataFrame(rows).to_csv(csv_path, sep="@", index=False)


def test_process_b_emits_multistep_parquet_with_required_columns(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    proc = tmp_path / "proc"
    _write_b_fixture_csv(raw)

    pipeline = RL4RSPipeline(raw_dir=raw, processed_dir=proc)
    pipeline.process_b()

    out = proc / "sessions_b.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    expected_cols = {
        "session_id", "sequence_id", "user_state", "slate", "item_features",
        "user_feedback", "candidate_ids", "candidate_features",
    }
    assert expected_cols.issubset(set(df.columns))
    # 3 rows = 3 steps total across both sessions
    assert len(df) == 3
    # Session 1 has 2 sequence_ids
    assert set(df[df["session_id"] == 1]["sequence_id"]) == {1, 2}
    # candidate_ids universe = all unique items {10,11,12,13,14,15}
    universe = set()
    for ids in df["candidate_ids"]:
        universe.update(ids)
    assert universe == {10, 11, 12, 13, 14, 15}
    # All rows share the same candidate universe (same length, sorted)
    first = list(df["candidate_ids"].iloc[0])
    for row_ids in df["candidate_ids"]:
        assert list(row_ids) == first
```

- [ ] **Step 2: Run to confirm fail**

Run: `.venv/bin/python -m pytest tests/test_rl4rs_pipeline.py -v`
Expected: FAIL with `AttributeError: 'RL4RSPipeline' object has no attribute 'process_b'`.

- [ ] **Step 3: Implement `process_b`**

In `rl_recsys/data/pipelines/rl4rs.py`, add a method to the `RL4RSPipeline` class (just after `process`):

```python
    def process_b(self) -> None:
        """Process dataset B (multi-step) into sessions_b.parquet.

        Groups raw rows by (session_id, sequence_id), parses CSV columns,
        and attaches a fixed candidate universe (all unique items in the
        slate column) to every row so loaders have a stable candidate set.
        """
        rl_file = self.raw_dir / "rl4rs-dataset" / "rl4rs_dataset_b_rl.csv"
        if not rl_file.exists():
            raise FileNotFoundError(
                f"Not found: {rl_file}. Run --download first."
            )
        df = pd.read_csv(rl_file, sep="@")
        df["slate"] = df["exposed_items"].apply(
            lambda s: [int(x) for x in str(s).split(",")]
        )
        df["user_feedback"] = df["user_feedback"].apply(
            lambda s: [int(x) for x in str(s).split(",")]
        )
        df["user_state"] = df["user_protrait"].apply(
            lambda s: [float(x) for x in str(s).split(",")]
        )
        df["item_features"] = df["item_feature"].apply(
            lambda s: [
                [float(v) for v in vec.split(",")]
                for vec in str(s).split(";")
            ]
        )

        universe_set: set[int] = set()
        for slate in df["slate"]:
            universe_set.update(slate)
        universe = sorted(universe_set)
        # Build a parallel feature lookup: pick each item's first observed feature vector.
        feature_for: dict[int, list[float]] = {}
        for slate, item_feats in zip(df["slate"], df["item_features"]):
            for item_id, feat in zip(slate, item_feats):
                if item_id not in feature_for:
                    feature_for[item_id] = list(feat)
        candidate_features = [feature_for[i] for i in universe]
        df["candidate_ids"] = [list(universe)] * len(df)
        df["candidate_features"] = [list(candidate_features)] * len(df)

        df = df.sort_values(["session_id", "sequence_id"], kind="stable")
        out_df = df[
            [
                "session_id", "sequence_id", "user_state", "slate",
                "item_features", "user_feedback",
                "candidate_ids", "candidate_features",
            ]
        ].reset_index(drop=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        out = self.processed_dir / "sessions_b.parquet"
        out_df.to_parquet(out, index=False)
        validate_parquet_schema(out, "rl_sessions_b")
        print(
            f"Saved {len(out_df):,} rows "
            f"({out_df['session_id'].nunique():,} sessions) to {out}"
        )
```

Note: this materialises the full candidate universe in every row. With ~380 unique items in dataset B that's a few MB of duplication — acceptable for now. A later task could switch to per-step candidate sampling if memory pressure arises.

- [ ] **Step 4: Run to confirm pass**

Run: `.venv/bin/python -m pytest tests/test_rl4rs_pipeline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rl_recsys/data/pipelines/rl4rs.py tests/test_rl4rs_pipeline.py
git commit -m "$(cat <<'EOF'
feat: process RL4RS dataset B into multi-step sessions_b.parquet

Adds RL4RSPipeline.process_b() which reads rl4rs_dataset_b_rl.csv
(multi-step, mean 1.78 steps/session), parses comma/semicolon-encoded
slate / feedback / state / item_feature columns, and attaches a fixed
candidate universe (all unique slate items, with their feature vectors)
to every row. The fixed universe lets the upcoming Sequential DR
loader yield a stable candidate set per step without per-step sampling.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `BehaviorPolicy` model + `slate_propensity`

A small PyTorch MLP that scores `(user_features, item_features, position_index)` triples and produces a softmax over the candidate vocabulary at each position. The model is constructed and used; training comes in Task 7.

**Files:**
- Create: `rl_recsys/evaluation/behavior_policy.py`
- Create: `tests/test_behavior_policy.py`

- [ ] **Step 1: Write failing test for `slate_propensity`**

Create `tests/test_behavior_policy.py`:

```python
import numpy as np
import pytest
import torch

from rl_recsys.evaluation.behavior_policy import BehaviorPolicy


def test_slate_propensity_returns_product_of_position_softmax() -> None:
    # 3 candidate items, slate_size=2, feature_dim=2.
    # Construct a model and override its scorer to return known logits so
    # the softmax probabilities are hand-computable.
    model = BehaviorPolicy(
        user_dim=2, item_dim=2, slate_size=2, num_items=3,
        hidden_dim=4, seed=0,
    )

    def fake_score(user_feat, candidate_feats, position):
        # Return position-dependent logits per candidate.
        if position == 0:
            return torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        return torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)

    model._score_position = fake_score  # monkey-patched scorer

    user = np.array([0.1, 0.2], dtype=np.float64)
    cand = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float64)
    slate = np.array([0, 1], dtype=np.int64)

    e = float(np.e)
    p_pos0_item0 = e / (e + 2.0)  # softmax([1,0,0])[0]
    p_pos1_item1 = e / (e + 2.0)  # softmax([0,1,0])[1]
    expected = p_pos0_item0 * p_pos1_item1

    result = model.slate_propensity(user, cand, slate)
    assert result == pytest.approx(expected)
```

- [ ] **Step 2: Run to confirm fail**

Run: `.venv/bin/python -m pytest tests/test_behavior_policy.py::test_slate_propensity_returns_product_of_position_softmax -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `BehaviorPolicy` (model + slate_propensity)**

Create `rl_recsys/evaluation/behavior_policy.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class BehaviorPolicy(nn.Module):
    """Per-position softmax classifier estimating μ(item | context, position).

    Forward pass scores every (candidate, position) pair given a context.
    slate_propensity returns Π_k softmax(scores)[slate[k]] across positions.
    """

    def __init__(
        self,
        *,
        user_dim: int,
        item_dim: int,
        slate_size: int,
        num_items: int,
        hidden_dim: int = 64,
        seed: int = 0,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self._user_dim = user_dim
        self._item_dim = item_dim
        self._slate_size = slate_size
        self._num_items = num_items
        self._mlp = nn.Sequential(
            nn.Linear(user_dim + item_dim + slate_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).double()

    def _score_position(
        self,
        user_feat: torch.Tensor,
        candidate_feats: torch.Tensor,
        position: int,
    ) -> torch.Tensor:
        """Returns logits of shape (num_candidates,) for the given position."""
        n = candidate_feats.shape[0]
        position_onehot = torch.zeros(self._slate_size, dtype=torch.float64)
        position_onehot[position] = 1.0
        user_tile = user_feat.unsqueeze(0).expand(n, -1)
        position_tile = position_onehot.unsqueeze(0).expand(n, -1)
        x = torch.cat([user_tile, candidate_feats, position_tile], dim=1)
        return self._mlp(x).squeeze(-1)

    def slate_propensity(
        self,
        user_features: np.ndarray,
        candidate_features: np.ndarray,
        slate: np.ndarray,
    ) -> float:
        """π_b(slate | context) = Π_k softmax(score(·, k))[slate[k]]."""
        user = torch.as_tensor(user_features, dtype=torch.float64)
        cand = torch.as_tensor(candidate_features, dtype=torch.float64)
        slate_t = torch.as_tensor(np.asarray(slate, dtype=np.int64))
        log_prob_total = 0.0
        with torch.no_grad():
            for k in range(int(slate_t.shape[0])):
                logits = self._score_position(user, cand, k)
                log_probs = torch.log_softmax(logits, dim=-1)
                log_prob_total += float(log_probs[int(slate_t[k])].item())
        result = float(np.exp(log_prob_total))
        if result <= 0.0:
            raise ValueError("zero propensity in logged slate")
        return result
```

- [ ] **Step 4: Run to confirm pass**

Run: `.venv/bin/python -m pytest tests/test_behavior_policy.py::test_slate_propensity_returns_product_of_position_softmax -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rl_recsys/evaluation/behavior_policy.py tests/test_behavior_policy.py
git commit -m "$(cat <<'EOF'
feat: add BehaviorPolicy network + slate_propensity API

Per-position softmax MLP scoring (user_features, item_features,
position_onehot) -> logit per candidate. slate_propensity returns
Π_k softmax(scores)[slate[k]] in log-space for numerical stability.
Defensive guard raises if the product underflows to zero.

Training loop and calibration helpers come in following tasks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `fit_behavior_policy` training loop

Trains the model on logged `(context, position, item_id_chosen)` tuples derived from a `sessions_b.parquet`-shaped DataFrame. Returns the fitted `BehaviorPolicy`.

**Files:**
- Modify: `rl_recsys/evaluation/behavior_policy.py` (add `fit_behavior_policy`)
- Modify: `tests/test_behavior_policy.py` (add fit-recovery test)

- [ ] **Step 1: Write failing test — synthetic 2-context recovery**

Add to `tests/test_behavior_policy.py`:

```python
import pandas as pd


def test_fit_behavior_policy_recovers_context_dependent_distribution(
    tmp_path,
) -> None:
    # Construct a synthetic logged dataset where context A always picks item 0
    # at position 0 and context B always picks item 1 at position 0. After
    # training, the model should put >0.5 probability on the correct item
    # given each context.
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(200):
        # Context A: user_state = [1, 0]
        rows.append({
            "session_id": rng.integers(1, 1000),
            "sequence_id": 1,
            "user_state": [1.0, 0.0],
            "slate": [0, 1, 2],
            "user_feedback": [1, 0, 0],
            "item_features": [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]],
            "candidate_ids": [0, 1, 2],
            "candidate_features": [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]],
        })
        # Context B: user_state = [0, 1]
        rows.append({
            "session_id": rng.integers(1000, 2000),
            "sequence_id": 1,
            "user_state": [0.0, 1.0],
            "slate": [1, 0, 2],
            "user_feedback": [1, 0, 0],
            "item_features": [[1.0, 0.0], [0.0, 0.0], [0.5, 0.5]],
            "candidate_ids": [0, 1, 2],
            "candidate_features": [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]],
        })
    df = pd.DataFrame(rows)
    parquet = tmp_path / "synth_b.parquet"
    df.to_parquet(parquet, index=False)

    from rl_recsys.evaluation.behavior_policy import fit_behavior_policy
    model = fit_behavior_policy(
        parquet, user_dim=2, item_dim=2, slate_size=3, num_items=3,
        epochs=20, batch_size=64, seed=0,
    )

    # In context A, position-0 prob for item 0 should exceed item 1's prob.
    user_a = np.array([1.0, 0.0], dtype=np.float64)
    cand = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float64)
    p_a_item0 = model.slate_propensity(user_a, cand, np.array([0], dtype=np.int64))
    p_a_item1 = model.slate_propensity(user_a, cand, np.array([1], dtype=np.int64))
    assert p_a_item0 > p_a_item1

    user_b = np.array([0.0, 1.0], dtype=np.float64)
    p_b_item0 = model.slate_propensity(user_b, cand, np.array([0], dtype=np.int64))
    p_b_item1 = model.slate_propensity(user_b, cand, np.array([1], dtype=np.int64))
    assert p_b_item1 > p_b_item0
```

- [ ] **Step 2: Run to confirm fail**

Run: `.venv/bin/python -m pytest tests/test_behavior_policy.py::test_fit_behavior_policy_recovers_context_dependent_distribution -v`
Expected: FAIL with `ImportError: cannot import name 'fit_behavior_policy'`.

- [ ] **Step 3: Implement `fit_behavior_policy`**

Append to `rl_recsys/evaluation/behavior_policy.py`:

```python
def fit_behavior_policy(
    parquet_path: Path,
    *,
    user_dim: int,
    item_dim: int,
    slate_size: int,
    num_items: int,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-2,
    seed: int = 0,
    hidden_dim: int = 64,
) -> BehaviorPolicy:
    """Fit a per-position softmax classifier on logged slate placements.

    Each row of `parquet_path` is expanded into `slate_size` training tuples:
    (user_state, candidate_features, position k, target = candidate-index of slate[k]).
    """
    import pandas as pd  # local import to keep module-level deps minimal

    df = pd.read_parquet(parquet_path)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Build training tensors. Every row contributes slate_size examples.
    users: list[np.ndarray] = []
    cands: list[np.ndarray] = []
    positions: list[int] = []
    targets: list[int] = []
    for _, row in df.iterrows():
        user = np.asarray(row["user_state"], dtype=np.float64)
        cand = np.asarray(row["candidate_features"], dtype=np.float64)
        cand_ids = list(row["candidate_ids"])
        slate = list(row["slate"])
        for k in range(min(slate_size, len(slate))):
            target_item_id = int(slate[k])
            try:
                target_idx = cand_ids.index(target_item_id)
            except ValueError:
                continue  # logged item not in candidate universe — skip
            users.append(user)
            cands.append(cand)
            positions.append(k)
            targets.append(target_idx)

    if not users:
        raise ValueError("no training tuples derivable from parquet")

    user_t = torch.as_tensor(np.stack(users), dtype=torch.float64)
    cand_t = torch.as_tensor(np.stack(cands), dtype=torch.float64)
    pos_t = torch.as_tensor(np.array(positions), dtype=torch.long)
    target_t = torch.as_tensor(np.array(targets), dtype=torch.long)

    model = BehaviorPolicy(
        user_dim=user_dim, item_dim=item_dim, slate_size=slate_size,
        num_items=num_items, hidden_dim=hidden_dim, seed=seed,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n = user_t.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = order[start : start + batch_size]
            losses: list[torch.Tensor] = []
            for i in batch_idx:
                logits = model._score_position(
                    user_t[i], cand_t[i], int(pos_t[i].item())
                )
                log_probs = torch.log_softmax(logits, dim=-1)
                losses.append(-log_probs[int(target_t[i].item())])
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
```

- [ ] **Step 4: Run to confirm pass**

Run: `.venv/bin/python -m pytest tests/test_behavior_policy.py::test_fit_behavior_policy_recovers_context_dependent_distribution -v`
Expected: PASS. (If the test is flaky, raise epochs to 30 — synthetic recovery should be very stable.)

- [ ] **Step 5: Run full behavior_policy test file**

Run: `.venv/bin/python -m pytest tests/test_behavior_policy.py -v`
Expected: both tests pass.

- [ ] **Step 6: Commit**

```bash
git add rl_recsys/evaluation/behavior_policy.py tests/test_behavior_policy.py
git commit -m "$(cat <<'EOF'
feat: train BehaviorPolicy on logged slate placements

fit_behavior_policy expands each parquet row into slate_size training
tuples (user, candidates, position, target=candidate-index-of-slate[k])
and trains the per-position softmax MLP via cross-entropy. Returns the
fitted model. Synthetic-recovery test verifies the model picks the
right item per context.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Calibration helpers + threshold gate

Adds `held_out_nll` and a wrapper `fit_behavior_policy_with_calibration` that splits the parquet, fits, evaluates NLL on the held-out, and raises if the model is worse than `2 * log(num_items)` (twice uniform).

**Files:**
- Modify: `rl_recsys/evaluation/behavior_policy.py` (add `held_out_nll`, `fit_behavior_policy_with_calibration`)
- Modify: `tests/test_behavior_policy.py` (calibration tests)

- [ ] **Step 1: Write failing test for `held_out_nll`**

Add to `tests/test_behavior_policy.py`:

```python
def test_held_out_nll_returns_average_neg_log_prob(tmp_path) -> None:
    from rl_recsys.evaluation.behavior_policy import (
        BehaviorPolicy, held_out_nll,
    )
    # Build a model with a deterministic scorer for which NLL is hand-computable.
    model = BehaviorPolicy(
        user_dim=2, item_dim=2, slate_size=1, num_items=3,
        hidden_dim=4, seed=0,
    )
    def fake_score(user_feat, candidate_feats, position):
        return torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    model._score_position = fake_score

    df = pd.DataFrame([
        {"user_state": [0.1, 0.2],
         "candidate_features": [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]],
         "candidate_ids": [0, 1, 2],
         "slate": [0]},  # target = position 0, item id 0
    ])
    e = float(np.e)
    expected = -np.log(e / (e + 2.0))  # softmax([1,0,0])[0]

    result = held_out_nll(model, df)
    assert result == pytest.approx(expected, rel=1e-6)


def test_fit_behavior_policy_with_calibration_raises_on_bad_nll(
    tmp_path, monkeypatch,
) -> None:
    from rl_recsys.evaluation import behavior_policy as bp_module

    # Force fit_behavior_policy to return a degenerate model whose NLL on
    # held-out exceeds 2*log(num_items). We monkey-patch fit_behavior_policy
    # to a stub that returns a model with random weights only (no training).
    def stub_fit(*args, **kwargs):
        return bp_module.BehaviorPolicy(
            user_dim=kwargs["user_dim"], item_dim=kwargs["item_dim"],
            slate_size=kwargs["slate_size"], num_items=kwargs["num_items"],
            hidden_dim=4, seed=0,
        )
    monkeypatch.setattr(bp_module, "fit_behavior_policy", stub_fit)

    rows = [
        {"session_id": i, "sequence_id": 1, "user_state": [1.0, 0.0],
         "slate": [(i % 3)], "user_feedback": [1],
         "item_features": [[0.0, 0.0]],
         "candidate_ids": [0, 1, 2],
         "candidate_features": [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]]}
        for i in range(50)
    ]
    df = pd.DataFrame(rows)
    parquet = tmp_path / "noisy_b.parquet"
    df.to_parquet(parquet, index=False)

    # NLL threshold gate: with the stub returning an untrained model, NLL on
    # noisy held-out should exceed 2*log(3) ≈ 2.197 only if the model is
    # severely biased. To FORCE a fail, set threshold ratio to a tiny value.
    with pytest.raises(ValueError, match="behavior policy NLL exceeds threshold"):
        bp_module.fit_behavior_policy_with_calibration(
            parquet, user_dim=2, item_dim=2, slate_size=1, num_items=3,
            epochs=1, batch_size=8, seed=0, nll_threshold=0.01,
        )
```

- [ ] **Step 2: Run to confirm fail**

Run: `.venv/bin/python -m pytest tests/test_behavior_policy.py::test_held_out_nll_returns_average_neg_log_prob tests/test_behavior_policy.py::test_fit_behavior_policy_with_calibration_raises_on_bad_nll -v`
Expected: both FAIL with `ImportError: cannot import name 'held_out_nll'`.

- [ ] **Step 3: Implement `held_out_nll` and `fit_behavior_policy_with_calibration`**

Append to `rl_recsys/evaluation/behavior_policy.py`:

```python
def held_out_nll(model: BehaviorPolicy, df) -> float:
    """Average negative log-likelihood over (row, slate-position) tuples.

    `df` rows must include `user_state`, `candidate_features`, `candidate_ids`,
    `slate`. Returns mean -log p(slate[k] | context, k) across all positions.
    """
    losses: list[float] = []
    with torch.no_grad():
        for _, row in df.iterrows():
            user = torch.as_tensor(np.asarray(row["user_state"]), dtype=torch.float64)
            cand = torch.as_tensor(
                np.asarray(row["candidate_features"]), dtype=torch.float64
            )
            cand_ids = list(row["candidate_ids"])
            slate = list(row["slate"])
            for k in range(len(slate)):
                target_item_id = int(slate[k])
                try:
                    target_idx = cand_ids.index(target_item_id)
                except ValueError:
                    continue
                logits = model._score_position(user, cand, k)
                log_probs = torch.log_softmax(logits, dim=-1)
                losses.append(-float(log_probs[target_idx].item()))
    if not losses:
        raise ValueError("no held-out tuples; cannot compute NLL")
    return float(np.mean(losses))


def fit_behavior_policy_with_calibration(
    parquet_path: Path,
    *,
    user_dim: int,
    item_dim: int,
    slate_size: int,
    num_items: int,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-2,
    seed: int = 0,
    hidden_dim: int = 64,
    nll_threshold: float | None = None,
    held_out_fraction: float = 0.1,
) -> BehaviorPolicy:
    """Fit + held-out NLL gate.

    Splits parquet 90/10 (deterministic via seed), fits on the 90, evaluates
    NLL on the 10, and raises if NLL > nll_threshold.
    Default threshold = 2 * log(num_items)  (twice uniform NLL).
    """
    import pandas as pd

    if nll_threshold is None:
        nll_threshold = 2.0 * float(np.log(num_items))

    df = pd.read_parquet(parquet_path)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    n_held = max(1, int(len(df) * held_out_fraction))
    held_idx = perm[:n_held]
    train_idx = perm[n_held:]

    train_path = parquet_path.with_name(parquet_path.stem + "_train.parquet")
    df.iloc[train_idx].to_parquet(train_path, index=False)
    try:
        model = fit_behavior_policy(
            train_path, user_dim=user_dim, item_dim=item_dim,
            slate_size=slate_size, num_items=num_items,
            epochs=epochs, batch_size=batch_size,
            learning_rate=learning_rate, seed=seed, hidden_dim=hidden_dim,
        )
    finally:
        train_path.unlink(missing_ok=True)

    held_df = df.iloc[held_idx]
    nll = held_out_nll(model, held_df)
    if nll > nll_threshold:
        raise ValueError(
            f"behavior policy NLL exceeds threshold: {nll:.4f} > {nll_threshold:.4f}"
        )
    print(f"Behavior policy held-out NLL = {nll:.4f} (threshold {nll_threshold:.4f})")
    return model
```

- [ ] **Step 4: Run new tests**

Run: `.venv/bin/python -m pytest tests/test_behavior_policy.py -v`
Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add rl_recsys/evaluation/behavior_policy.py tests/test_behavior_policy.py
git commit -m "$(cat <<'EOF'
feat: add behavior-policy calibration gate

held_out_nll computes mean negative log-likelihood across logged slate
positions on a held-out DataFrame. fit_behavior_policy_with_calibration
splits 90/10, fits on the 90, evaluates NLL on the 10, and raises if NLL
exceeds threshold (default = 2 * log(num_items), i.e. twice the uniform
NLL). Catches catastrophic training failures before downstream OPE
estimators consume the propensities.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `RL4RSTrajectoryOPESource` loader

Loads `sessions_b.parquet`, attaches a fitted `BehaviorPolicy`, and yields `LoggedTrajectoryStep` objects.

**Files:**
- Create: `rl_recsys/data/loaders/rl4rs_trajectory_ope.py`
- Create: `tests/test_rl4rs_trajectory_ope.py`

- [ ] **Step 1: Write failing test for the loader**

Create `tests/test_rl4rs_trajectory_ope.py`:

```python
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from rl_recsys.evaluation.behavior_policy import BehaviorPolicy
from rl_recsys.evaluation.ope_trajectory import LoggedTrajectoryStep


def _fixture_b_parquet(tmp_path: Path) -> Path:
    rows = [
        {"session_id": 1, "sequence_id": 1, "user_state": [1.0, 0.0],
         "slate": [10, 11], "user_feedback": [1, 0],
         "item_features": [[0.0, 0.0], [1.0, 0.0]],
         "candidate_ids": [10, 11, 12],
         "candidate_features": [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]]},
        {"session_id": 1, "sequence_id": 2, "user_state": [1.0, 0.0],
         "slate": [11, 12], "user_feedback": [0, 1],
         "item_features": [[1.0, 0.0], [0.5, 0.5]],
         "candidate_ids": [10, 11, 12],
         "candidate_features": [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]]},
        {"session_id": 2, "sequence_id": 1, "user_state": [0.0, 1.0],
         "slate": [10, 12], "user_feedback": [0, 0],
         "item_features": [[0.0, 0.0], [0.5, 0.5]],
         "candidate_ids": [10, 11, 12],
         "candidate_features": [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]]},
    ]
    p = tmp_path / "sessions_b.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    return p


def test_loader_emits_trajectories_grouped_by_session(tmp_path: Path) -> None:
    from rl_recsys.data.loaders.rl4rs_trajectory_ope import (
        RL4RSTrajectoryOPESource,
    )
    parquet = _fixture_b_parquet(tmp_path)

    model = BehaviorPolicy(
        user_dim=2, item_dim=2, slate_size=2, num_items=3,
        hidden_dim=4, seed=0,
    )

    source = RL4RSTrajectoryOPESource(
        parquet_path=parquet, behavior_policy=model, slate_size=2,
    )
    trajectories = list(source.iter_trajectories(max_trajectories=10, seed=0))

    assert len(trajectories) == 2  # 2 unique session_ids
    s1 = next(t for t in trajectories if len(t) == 2)
    s2 = next(t for t in trajectories if len(t) == 1)

    assert all(isinstance(s, LoggedTrajectoryStep) for s in s1)
    # logged_action shape == (slate_size,)
    assert s1[0].logged_action.shape == (2,)
    np.testing.assert_array_equal(s1[0].logged_action, np.array([10, 11]))
    np.testing.assert_array_equal(s1[1].logged_action, np.array([11, 12]))
    np.testing.assert_array_equal(s2[0].logged_action, np.array([10, 12]))
    # rewards = sum(user_feedback)
    assert s1[0].logged_reward == 1.0
    assert s1[1].logged_reward == 1.0
    assert s2[0].logged_reward == 0.0
    # propensity in (0, 1]
    for t in trajectories:
        for step in t:
            assert 0.0 < step.propensity <= 1.0
```

- [ ] **Step 2: Run to confirm fail**

Run: `.venv/bin/python -m pytest tests/test_rl4rs_trajectory_ope.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the loader**

Create `rl_recsys/data/loaders/rl4rs_trajectory_ope.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from rl_recsys.environments.base import RecObs
from rl_recsys.evaluation.behavior_policy import BehaviorPolicy
from rl_recsys.evaluation.ope_trajectory import LoggedTrajectoryStep


class RL4RSTrajectoryOPESource:
    """LoggedTrajectorySource over RL4RS dataset B sessions_b.parquet.

    Groups rows by session_id ordered by sequence_id and yields one
    LoggedTrajectoryStep per row. Reward = sum(user_feedback). Propensity is
    computed by a pre-fitted BehaviorPolicy.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        behavior_policy: BehaviorPolicy,
        *,
        slate_size: int,
    ) -> None:
        self._df = pd.read_parquet(parquet_path)
        self._policy = behavior_policy
        self._slate_size = int(slate_size)

    def iter_trajectories(
        self, *, max_trajectories: int | None = None, seed: int | None = None
    ) -> Iterator[list[LoggedTrajectoryStep]]:
        ordered = self._df.sort_values(["session_id", "sequence_id"], kind="stable")
        groups = ordered.groupby("session_id", sort=False)
        session_ids = list(groups.groups.keys())
        rng = np.random.default_rng(0 if seed is None else seed)
        if seed is not None:
            rng.shuffle(session_ids)

        emitted = 0
        for sid in session_ids:
            if max_trajectories is not None and emitted >= max_trajectories:
                break
            group = groups.get_group(sid)
            steps: list[LoggedTrajectoryStep] = []
            for _, row in group.iterrows():
                user_features = np.asarray(row["user_state"], dtype=np.float64)
                candidate_features = np.asarray(
                    row["candidate_features"], dtype=np.float64
                )
                candidate_ids = np.asarray(row["candidate_ids"], dtype=np.int64)
                logged_slate = np.asarray(row["slate"], dtype=np.int64)
                logged_reward = float(np.sum(row["user_feedback"]))
                propensity = self._policy.slate_propensity(
                    user_features, candidate_features, logged_slate,
                )
                obs = RecObs(
                    user_features=user_features,
                    candidate_features=candidate_features,
                    candidate_ids=candidate_ids,
                )
                steps.append(
                    LoggedTrajectoryStep(
                        obs=obs,
                        logged_action=logged_slate,
                        logged_reward=logged_reward,
                        propensity=propensity,
                    )
                )
            yield steps
            emitted += 1
```

- [ ] **Step 4: Run to confirm pass**

Run: `.venv/bin/python -m pytest tests/test_rl4rs_trajectory_ope.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rl_recsys/data/loaders/rl4rs_trajectory_ope.py tests/test_rl4rs_trajectory_ope.py
git commit -m "$(cat <<'EOF'
feat: RL4RSTrajectoryOPESource yields per-session sequential DR steps

Loads sessions_b.parquet, groups by session_id ordered by sequence_id,
attaches a pre-fitted BehaviorPolicy, and yields LoggedTrajectoryStep
sequences with reward = sum(user_feedback) and propensity from the
behavior model. Supports deterministic session-order shuffling via seed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: `evaluate_trajectory_ope_with_variance`

**Files:**
- Modify: `rl_recsys/evaluation/variance.py` (new function + import)
- Modify: `rl_recsys/evaluation/__init__.py` (export)
- Modify: `tests/test_variance.py` (smoke test)

- [ ] **Step 1: Write failing smoke test**

Add to `tests/test_variance.py`:

```python
def test_evaluate_trajectory_ope_with_variance_smoke() -> None:
    import numpy as np

    from rl_recsys.environments.base import RecObs
    from rl_recsys.evaluation import (
        evaluate_trajectory_ope_with_variance, LoggedTrajectoryStep,
    )

    obs = RecObs(
        user_features=np.zeros(2, dtype=np.float64),
        candidate_features=np.zeros((3, 2), dtype=np.float64),
        candidate_ids=np.arange(3, dtype=np.int64),
    )
    traj = [
        LoggedTrajectoryStep(
            obs=obs,
            logged_action=np.array([0, 1], dtype=np.int64),
            logged_reward=1.0,
            propensity=0.3,
        ),
        LoggedTrajectoryStep(
            obs=obs,
            logged_action=np.array([0, 1], dtype=np.int64),
            logged_reward=0.0,
            propensity=0.3,
        ),
    ]

    class _FixedSource:
        def iter_trajectories(self, *, max_trajectories=None, seed=None):
            yield traj

    class _FlatScoreAgent:
        def select_slate(self, obs):
            return np.array([0, 1], dtype=np.int64)
        def score_items(self, obs):
            return np.zeros(3, dtype=np.float64)
        def update(self, *args, **kwargs):
            return {}

    result = evaluate_trajectory_ope_with_variance(
        make_source=lambda: _FixedSource(),
        make_agent=lambda: _FlatScoreAgent(),
        agent_name="flat",
        max_trajectories=1,
        n_seeds=3,
        base_seed=42,
        gamma=0.9,
        temperature=1.0,
    )

    assert "avg_seq_dr_value" in result.mean
    assert "avg_logged_discounted_return" in result.mean
    assert isinstance(result.mean["avg_seq_dr_value"], float)
    assert "avg_seq_dr_value" in result.std
    assert result.n_seeds == 3
    # Deterministic source → std == 0 across seeds
    assert result.std["avg_seq_dr_value"] == 0.0
```

- [ ] **Step 2: Run to confirm fail**

Run: `.venv/bin/python -m pytest tests/test_variance.py::test_evaluate_trajectory_ope_with_variance_smoke -v`
Expected: FAIL with `ImportError: cannot import name 'evaluate_trajectory_ope_with_variance'`.

- [ ] **Step 3: Implement the wrapper**

In `rl_recsys/evaluation/variance.py`:

Add to imports at the top:

```python
from rl_recsys.evaluation.ope_trajectory import (
    LoggedTrajectorySource, evaluate_trajectory_ope_agent,
)
```

Append the new function:

```python
def evaluate_trajectory_ope_with_variance(
    make_source: Callable[[], LoggedTrajectorySource],
    make_agent: Callable[[], Agent],
    *,
    agent_name: str,
    max_trajectories: int,
    n_seeds: int = 5,
    base_seed: int = 42,
    gamma: float = 0.95,
    reward_model: Callable[[int], float] | None = None,
    clip: tuple[float, float] = (0.1, 10.0),
    temperature: float = 1.0,
) -> VarianceEvaluation:
    """Run evaluate_trajectory_ope_agent n_seeds times; return mean ± std."""
    results = [
        evaluate_trajectory_ope_agent(
            make_source(),
            make_agent(),
            agent_name=agent_name,
            max_trajectories=max_trajectories,
            seed=base_seed + i,
            gamma=gamma,
            reward_model=reward_model,
            clip=clip,
            temperature=temperature,
        )
        for i in range(n_seeds)
    ]
    mean, std = _aggregate_runs(results)
    return VarianceEvaluation(mean=mean, std=std, n_seeds=n_seeds)
```

- [ ] **Step 4: Export from package**

In `rl_recsys/evaluation/__init__.py`:

Add `evaluate_trajectory_ope_with_variance` to the `from rl_recsys.evaluation.variance import (...)` block, and add it to `__all__` (alphabetical placement: between `evaluate_trajectory_ope_agent` and `evaluate_trajectory_with_variance`).

- [ ] **Step 5: Run to confirm pass**

Run: `.venv/bin/python -m pytest tests/test_variance.py -v`
Expected: PASS.

- [ ] **Step 6: Run full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add rl_recsys/evaluation/variance.py rl_recsys/evaluation/__init__.py tests/test_variance.py
git commit -m "$(cat <<'EOF'
feat: add evaluate_trajectory_ope_with_variance

Variance wrapper around evaluate_trajectory_ope_agent. Mirrors the two
existing siblings (bandit, trajectory) — runs n_seeds times with fresh
source/agent factories per seed, aggregates scalar metrics via mean/std.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: End-to-end smoke + benchmark script + TODO.md cleanup

**Files:**
- Modify: `tests/test_rl4rs_trajectory_ope.py` (add integration smoke)
- Create: `scripts/benchmark_rl4rs_b_seq_dr.py`
- Modify: `TODO.md` (remove "Real-data Sequential DR" entry)

- [ ] **Step 1: Write end-to-end smoke test**

Add to `tests/test_rl4rs_trajectory_ope.py`:

```python
def test_end_to_end_seq_dr_on_synthetic_b_fixture(tmp_path: Path) -> None:
    # Write a tiny multi-step parquet, fit BehaviorPolicy, build the loader,
    # run evaluate_trajectory_ope_agent, assert avg_seq_dr_value is finite
    # and ess is reported.
    rng = np.random.default_rng(0)
    rows = []
    for sid in range(40):
        for seq in range(2):
            rows.append({
                "session_id": sid,
                "sequence_id": seq,
                "user_state": rng.standard_normal(2).tolist(),
                "slate": [rng.integers(0, 3), rng.integers(0, 3)],
                "user_feedback": [int(rng.integers(0, 2)), int(rng.integers(0, 2))],
                "item_features": [[0.0, 0.0], [1.0, 0.0]],
                "candidate_ids": [0, 1, 2],
                "candidate_features": [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]],
            })
    parquet = tmp_path / "sessions_b.parquet"
    pd.DataFrame(rows).to_parquet(parquet, index=False)

    from rl_recsys.evaluation.behavior_policy import (
        fit_behavior_policy_with_calibration,
    )
    from rl_recsys.data.loaders.rl4rs_trajectory_ope import (
        RL4RSTrajectoryOPESource,
    )
    from rl_recsys.evaluation import evaluate_trajectory_ope_agent
    from rl_recsys.environments.base import RecObs

    model = fit_behavior_policy_with_calibration(
        parquet, user_dim=2, item_dim=2, slate_size=2, num_items=3,
        epochs=5, batch_size=16, seed=0, nll_threshold=10.0,
    )

    class _FlatScoreAgent:
        def select_slate(self, obs):
            return np.array([0, 1], dtype=np.int64)
        def score_items(self, obs):
            return np.zeros(len(obs.candidate_features), dtype=np.float64)
        def update(self, *a, **kw):
            return {}

    source = RL4RSTrajectoryOPESource(
        parquet_path=parquet, behavior_policy=model, slate_size=2,
    )
    result = evaluate_trajectory_ope_agent(
        source, _FlatScoreAgent(), agent_name="flat",
        max_trajectories=40, seed=0, gamma=0.9, temperature=1.0,
    )

    assert np.isfinite(result.avg_seq_dr_value)
    assert np.isfinite(result.avg_logged_discounted_return)
    assert result.ess > 0.0
    assert result.trajectories > 0
```

- [ ] **Step 2: Run end-to-end smoke to confirm pass**

Run: `.venv/bin/python -m pytest tests/test_rl4rs_trajectory_ope.py::test_end_to_end_seq_dr_on_synthetic_b_fixture -v`
Expected: PASS. (May print a "behavior policy NLL = ..." line; that's fine.)

- [ ] **Step 3: Create benchmark script**

Create `scripts/benchmark_rl4rs_b_seq_dr.py`:

```python
"""Real-data Sequential DR benchmark on RL4RS dataset B.

Run after `python -m rl_recsys.data.cli rl4rs --download` and the
process_b step has produced data/processed/rl4rs/sessions_b.parquet.

Usage:
    .venv/bin/python scripts/benchmark_rl4rs_b_seq_dr.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rl_recsys.agents import LinUCBAgent, RandomAgent
from rl_recsys.data.loaders.rl4rs_trajectory_ope import RL4RSTrajectoryOPESource
from rl_recsys.evaluation import evaluate_trajectory_ope_with_variance
from rl_recsys.evaluation.behavior_policy import (
    fit_behavior_policy_with_calibration,
)


def main() -> None:
    parquet = Path("data/processed/rl4rs/sessions_b.parquet")
    if not parquet.exists():
        raise SystemExit(
            f"missing {parquet}. Run RL4RSPipeline.process_b() first "
            "(see rl_recsys.data.cli)."
        )

    # Auto-detect shapes from the parquet — avoids hardcoding numbers that
    # would drift if RL4RSPipeline emits different dims later.
    import pandas as pd
    df = pd.read_parquet(
        parquet, columns=["user_state", "item_features", "slate", "candidate_ids"]
    )
    USER_DIM = len(df["user_state"].iloc[0])
    ITEM_DIM = len(df["item_features"].iloc[0][0])
    SLATE_SIZE = len(df["slate"].iloc[0])
    num_items = len(df["candidate_ids"].iloc[0])
    print(
        f"detected dims: user={USER_DIM}, item={ITEM_DIM}, "
        f"slate={SLATE_SIZE}, num_items={num_items}"
    )

    model = fit_behavior_policy_with_calibration(
        parquet, user_dim=USER_DIM, item_dim=ITEM_DIM,
        slate_size=SLATE_SIZE, num_items=num_items,
        epochs=5, batch_size=512, seed=0,
    )

    def make_source() -> RL4RSTrajectoryOPESource:
        return RL4RSTrajectoryOPESource(
            parquet_path=parquet, behavior_policy=model, slate_size=SLATE_SIZE,
        )

    print("\n--- LinUCB ---")
    linucb_result = evaluate_trajectory_ope_with_variance(
        make_source=make_source,
        make_agent=lambda: LinUCBAgent(
            slate_size=SLATE_SIZE, user_dim=USER_DIM, item_dim=ITEM_DIM, alpha=1.0,
        ),
        agent_name="linucb",
        max_trajectories=5000, n_seeds=3, base_seed=42, gamma=0.95, temperature=1.0,
    )
    print(linucb_result)

    print("\n--- Random ---")
    random_result = evaluate_trajectory_ope_with_variance(
        make_source=make_source,
        make_agent=lambda: RandomAgent(slate_size=SLATE_SIZE, seed=0),
        agent_name="random",
        max_trajectories=5000, n_seeds=3, base_seed=42, gamma=0.95, temperature=1.0,
    )
    print(random_result)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run script (dry-run skip if parquet missing)**

Run: `.venv/bin/python scripts/benchmark_rl4rs_b_seq_dr.py`
Expected: either runs to completion OR exits with the "missing parquet" message — both are acceptable for plan completion. If `data/processed/rl4rs/sessions_b.parquet` exists, the script should print numbers; if not, the SystemExit is the expected behavior.

If running, the printed result is informational — no specific value to assert. Note any warnings about ESS < 1% in the output and surface them in the commit message.

- [ ] **Step 5: Update TODO.md**

In `TODO.md`, remove the entire "### Real-data Sequential DR" section under "## Next up". The "## Next up" header may now be empty — if so, replace it with a stub note: `(Backlog continues below.)`.

Actual edit: locate the block from `### Real-data Sequential DR` through the end of its bullet list (the last line before `## Loader / data`) and delete those lines.

- [ ] **Step 6: Run full test suite one final time**

Run: `.venv/bin/python -m pytest -q`
Expected: all green. Report the final test count and the deltas vs. the 165 baseline.

- [ ] **Step 7: Commit**

```bash
git add tests/test_rl4rs_trajectory_ope.py scripts/benchmark_rl4rs_b_seq_dr.py TODO.md
git commit -m "$(cat <<'EOF'
feat: end-to-end RL4RS sequential DR benchmark + smoke test

Wires the full pipeline: BehaviorPolicy fit + calibration gate ->
RL4RSTrajectoryOPESource -> evaluate_trajectory_ope_with_variance.
Adds an integration smoke test on a synthetic 40-session fixture and
a runnable benchmark script for the real dataset B parquet. Removes
the "Real-data Sequential DR" entry from TODO.md Next up.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review notes for the executor

After completing all 11 tasks, before declaring done:

1. **Spec coverage.** Cross-reference each numbered decision in the spec ("Decisions" 1-6) against the tasks above:
   - Decision 1 (slate-as-action) — Task 2.
   - Decision 2 (Boltzmann shim) — Task 2.
   - Decision 3 (per-position softmax classifier) — Tasks 6-8.
   - Decision 4 (reward = sum(user_feedback)) — Task 9.
   - Decision 5 (score_items contract) — Tasks 1, 2.
   - Decision 6 (dataset A path untouched) — confirmed: no task touches `process()`.

2. **Test count delta.** Baseline was 165. Expected gain: 1 (T1 RandomAgent.score_items) + 3 (T2 Boltzmann + contract) + 1 (T3 ESS) + 2 (T4 schema) + 1 (T5 pipeline) + 4 (T6-T8 behavior policy) + 1 (T9 loader) + 1 (T10 variance) + 1 (T11 e2e smoke) = **15 new tests → 180 total**.

3. **No leftovers.** `git status` should be clean. No `*.bak` files, no commented-out blocks.

4. **Final report to the user.** Include: total commits, final test count, and any surfaced ESS warnings from the optional benchmark run.
