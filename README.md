# rl-recsys

RL-based recommendation ranking with LinUCB, REINFORCE (Plackett-Luce), and Slate-Q.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
# Run synthetic experiment
python experiments/run_synthetic.py

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
```

## Architecture

- `rl_recsys/config.py` — dataclass configs (EnvConfig, AgentConfig, TrainConfig)
- `rl_recsys/environments/` — recommendation environments (synthetic latent-factor)
- `rl_recsys/agents/` — RL agents (LinUCB, REINFORCE, Slate-Q)
- `rl_recsys/rewards/` — reward models (click-sum, DCG)
- `rl_recsys/networks/` — neural network components (MLP, ItemScorer)
- `rl_recsys/data/` — replay buffer and data utilities
- `rl_recsys/training/` — training loop and metrics (NDCG, MRR, CTR)
- `experiments/` — runnable experiment scripts
