"""Real-data Sequential DR benchmark on RL4RS dataset B.

Run after `python -m rl_recsys.data.cli rl4rs --download` and the
process_b step has produced data/processed/rl4rs/sessions_b.parquet.

Usage:
    .venv/bin/python scripts/benchmark_rl4rs_b_seq_dr.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from rl_recsys.agents import LinUCBAgent, RandomAgent
from rl_recsys.data.loaders.rl4rs_trajectory_ope import RL4RSTrajectoryOPESource
from rl_recsys.evaluation import evaluate_trajectory_ope_with_variance
from rl_recsys.evaluation.behavior_policy import (
    fit_behavior_policy_with_calibration,
)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parquet = Path("data/processed/rl4rs/sessions_b.parquet")
    if not parquet.exists():
        raise SystemExit(
            f"missing {parquet}. Run RL4RSPipeline.process_b() first "
            "(see rl_recsys.data.cli)."
        )

    # Auto-detect shapes from the parquet — avoids hardcoding numbers that
    # would drift if RL4RSPipeline emits different dims later.
    import pandas as pd
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    df = pd.read_parquet(parquet, columns=["user_state", "item_features", "slate"])
    USER_DIM = len(df["user_state"].iloc[0])
    ITEM_DIM = len(df["item_features"].iloc[0][0])
    SLATE_SIZE = len(df["slate"].iloc[0])

    # Derive num_items from the unique items seen across all logged slates.
    slate_table = pq.read_table(parquet, columns=["slate"])
    flat_items = slate_table["slate"].combine_chunks().flatten()
    num_items = int(pc.count_distinct(flat_items).as_py())
    print(
        f"detected dims: user={USER_DIM}, item={ITEM_DIM}, "
        f"slate={SLATE_SIZE}, num_items={num_items}",
        flush=True,
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

    print("\n--- LinUCB ---", flush=True)
    linucb_result = evaluate_trajectory_ope_with_variance(
        make_source=make_source,
        make_agent=lambda: LinUCBAgent(
            slate_size=SLATE_SIZE, user_dim=USER_DIM, item_dim=ITEM_DIM, alpha=1.0,
        ),
        agent_name="linucb",
        max_trajectories=5000, n_seeds=3, base_seed=42, gamma=0.95, temperature=1.0,
    )
    print(linucb_result, flush=True)

    print("\n--- Random ---", flush=True)
    random_result = evaluate_trajectory_ope_with_variance(
        make_source=make_source,
        make_agent=lambda: RandomAgent(slate_size=SLATE_SIZE, seed=0),
        agent_name="random",
        max_trajectories=5000, n_seeds=3, base_seed=42, gamma=0.95, temperature=1.0,
    )
    print(random_result, flush=True)


if __name__ == "__main__":
    main()
