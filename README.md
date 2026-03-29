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
