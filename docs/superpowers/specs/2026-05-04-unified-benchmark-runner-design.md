# Unified Benchmark Runner Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the two diverged experiment scripts (`run_synthetic.py`, `run_dataset_bandit.py`) with a single Hydra-based `run.py` that benchmarks Random and LinUCB agents across all environments (synthetic, KuaiRec, FinnNoSlate, RL4RS, and logged-interaction datasets), with MLflow tracking and Hydra multirun for ablations.

**Architecture:** A thin `experiments/run.py` entry point delegates to an env factory (`rl_recsys/environments/factory.py`) and the existing `trainer.train()`. The trainer loop is extended from single-step to `while not done` so session-based envs (KuaiRec, RL4RS) work alongside bandit envs (FinnNoSlate, synthetic). New Hydra env configs under `conf/env/` cover each env type.

**Tech stack:** Python, Hydra 1.3, MLflow (SQLite backend), pandas/parquet, existing `rl_recsys` agent and training infrastructure.

---

## Env Factory

`rl_recsys/environments/factory.py` exposes a single function `build_env(cfg: DictConfig) -> RecEnv`. It switches on `cfg.env.type`:

| `type` | Class | Data loaded |
|---|---|---|
| `synthetic` | `SyntheticEnv` | none (generated) |
| `kuairec` | `KuaiRecEnv` | `interactions.parquet` + `item_features.parquet` from `processed_dir` |
| `finn_no_slate` | `FinnNoSlateEnv` | `interactions.parquet` + optional `item_features.parquet` from `processed_dir` |
| `rl4rs` | `RL4RSEnv` | `sessions.parquet` from `processed_dir` (path passed directly) |
| `logged` | `LoggedInteractionEnv` | `{processed_dir}/{filename}` |

The factory raises `FileNotFoundError` with a clear message if required parquet files are missing — no silent fallback. Data preparation is a separate step (`scripts/prepare_data.py`); the runner does not download or process data.

## Trainer Loop Change

`rl_recsys/training/trainer.py` currently does a single `select_slate` + `step` per episode. This is replaced with a `while not done` inner loop:

```python
obs = env.reset(seed=...)
episode_rewards, episode_clicks = [], []
done = False
while not done:
    slate = agent.select_slate(obs)
    step = env.step(slate)
    agent.update(obs, slate, step.reward, step.clicks, step.obs)
    episode_rewards.append(step.reward)
    episode_clicks.append(step.clicks)
    obs = step.obs
    done = step.done
```

This is backward-compatible: bandit and synthetic envs set `done=True` on the first step, so their behavior is unchanged.

## Entry Point

`experiments/run.py` is a `@hydra.main` app that:
1. Calls `build_env(cfg.env)` → `RecEnv`
2. Calls `build_agent(cfg.agent, env)` — extracts `slate_size`, `user_dim`, `item_dim` from `env` properties and `alpha` from `cfg.agent`; wraps the existing `AgentConfig`-based factory
3. Calls `trainer.train(env, agent, cfg)` → history
4. Prints a one-line summary per run

The old scripts (`run_synthetic.py`, `run_dataset_bandit.py`) gain a deprecation notice comment pointing to `run.py` but are otherwise left intact.

## Config Structure

**New files under `conf/env/`:**

```yaml
# conf/env/kuairec.yaml
type: kuairec
processed_dir: data/processed/kuairec
slate_size: 6
feature_dim: 32
feature_source: native
seed: 42

# conf/env/finn_no_slate.yaml
type: finn_no_slate
processed_dir: data/processed/finn-no-slate
slate_size: 5
feature_dim: 16
seed: 42

# conf/env/rl4rs.yaml
type: rl4rs
processed_dir: data/processed/rl4rs
slate_size: 6
feature_dim: 32
feature_source: native
seed: 42

# conf/env/logged.yaml
type: logged
dataset_key: movielens-100k
processed_dir: data/processed/movielens
filename: ratings_100k.parquet
slate_size: 10
num_candidates: 50
feature_dim: 16
rating_threshold: 4.0
seed: 42
```

**`conf/train.yaml`** updated to set `defaults` pointing at `run.py` conventions (env, agent, train, runtime, mlflow).

**`conf/mlflow/local.yaml`** (new): enables MLflow by default with `tracking_uri: sqlite:///mlflow.db` and `experiment_name: rl-recsys-benchmark`. The existing `default.yaml` (disabled) is unchanged.

**Unchanged:** `conf/env/synthetic.yaml`, `conf/agent/random.yaml`, `conf/agent/linucb.yaml`, `conf/train/default.yaml`, `conf/runtime/default.yaml`.

## Example Commands

```bash
# single run
python experiments/run.py env=rl4rs agent=linucb

# full 8-run benchmark matrix
python experiments/run.py --multirun \
  env=synthetic,kuairec,finn_no_slate,rl4rs \
  agent=random,linucb

# override hyperparams on the fly
python experiments/run.py env=kuairec agent=linucb \
  train.num_episodes=500 agent.alpha=0.5

# view results
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## MLflow Experiment Structure

- Experiment name: `rl-recsys-benchmark`
- Each run tagged with `env_type` and `agent_name`
- Per-episode metrics logged: `reward`, `ctr`, `ndcg`, `mrr`
- Summary logged at end: `avg_reward`, `final_ctr`
- Existing `finish_mlflow` artifact (history CSV) retained

## Migration

| Old command | New equivalent |
|---|---|
| `python experiments/run_synthetic.py` | `python experiments/run.py env=synthetic agent=linucb` |
| `python experiments/run_dataset_bandit.py --datasets movielens-100k` | `python experiments/run.py env=logged` |
| `python experiments/run_dataset_bandit.py --datasets kuairec` | `python experiments/run.py env=kuairec` |

## Testing

- `tests/test_factory.py`: factory builds each env type from a minimal config; raises `FileNotFoundError` for missing data
- `tests/test_trainer.py`: extend existing tests to cover multi-step episodes (mock session env with 3 steps)
- `tests/test_run.py`: smoke-test `run.py` end-to-end with synthetic env (no MLflow, in-memory)
- Existing 124 tests must continue to pass
