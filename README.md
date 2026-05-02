# rl-recsys

RL-based recommendation ranking scaffold with random and shared LinUCB baselines.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Python Environment

Activate the virtualenv once before running repo commands:

```bash
source .venv/bin/activate
```

The examples below assume the virtualenv is active, so they use `python` and
`pytest` directly. For non-interactive automation, use `.venv/bin/python` and
`.venv/bin/pytest` if activating the shell first is not practical.

## Usage

```bash
# Run synthetic experiment
python experiments/run_synthetic.py

# Run the shared contextual LinUCB baseline
python experiments/run_synthetic.py agent=linucb

# Compare random and LinUCB on sampled logged interactions
python experiments/run_dataset_bandit.py \
  --datasets movielens-100k movielens-1m \
  --download --process

# Run Open Bandit off-policy evaluation with replay, IPS, and SNIPS
python experiments/run_ope_benchmark.py \
  --dataset open-bandit \
  --download --process

# Compare against the older hashed-feature path
python experiments/run_ope_benchmark.py \
  --dataset open-bandit \
  --feature-source hashed

# Run across every processed Open Bandit policy and campaign split
python experiments/run_ope_benchmark.py \
  --dataset open-bandit \
  --policy any \
  --campaign any

# Inspect the effective Hydra config without running
python experiments/run_synthetic.py --cfg job --resolve

# Override values from the CLI
python experiments/run_synthetic.py train.num_episodes=25 env.slate_size=5 train.seed=7

# Enable offline W&B logging
python experiments/run_synthetic.py wandb.enabled=true wandb.mode=offline

# Enable local MLflow tracking with a local SQLite backend
python experiments/run_synthetic.py mlflow.enabled=true mlflow.tracking_uri=sqlite:///mlflow.db

# Connect to an existing W&B server
export WANDB_API_KEY=your_api_key
python experiments/run_synthetic.py \
  wandb.enabled=true \
  wandb.mode=online \
  wandb.base_url=http://your-wandb-server:8080

# Verify the server before sending online runs
python /home/vishsangale/workspace/latent-superpowers/core/wandb/scripts/check_wandb_server.py \
  --base-url http://your-wandb-server:8080 --json

# Run tests
pytest tests/

# Validate the repo surface
python tools/validate_repo.py
```

## Datasets

Dataset preparation is registry-driven through `scripts/prepare_data.py`.

```bash
# List available dataset keys
python scripts/prepare_data.py --help

# Download and process one dataset
python scripts/prepare_data.py --dataset movielens-100k --download --process

# Override storage locations
python scripts/prepare_data.py \
  --dataset amazon-books \
  --raw-dir /data/raw/amazon/Books \
  --processed-dir /data/processed/amazon/Books \
  --download --process
```

The current registry has 17 dataset keys across CF, session, slate RL, and
off-policy evaluation data. See [docs/datasets.md](docs/datasets.md) for the
full catalog and guidance on which dataset to use.

Open Bandit is the first supported logged-policy benchmark path. It preserves
`propensity_score`, `policy`, `campaign`, and native context columns, so
`experiments/run_ope_benchmark.py` can report replay, IPS, and self-normalized
IPS estimates for top-1 target policies on one slice or across all processed
splits. Native Open Bandit context is the default feature source; use
`--feature-source hashed` for the previous grouped/hash baseline.

## Architecture

- `rl_recsys/config.py` — dataclass configs (EnvConfig, AgentConfig, TrainConfig)
- `rl_recsys/environments/` — recommendation environments (synthetic latent-factor)
- `rl_recsys/agents/` — recommendation agents (random, LinUCB)
- `rl_recsys/rewards/` — reward models (click-sum, DCG)
- `rl_recsys/networks/` — neural network components (MLP, ItemScorer)
- `rl_recsys/data/` — replay buffer and data utilities
- `rl_recsys/training/` — training loop and metrics (NDCG, MRR, CTR)
- `experiments/` — runnable experiment scripts
