from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rl_recsys.data.loaders.rl4rs_trajectory_ope import RL4RSTrajectoryOPESource
from rl_recsys.evaluation.behavior_policy import BehaviorPolicy
from rl_recsys.training.agent_grid_runner import GridRun, run_grid
from rl_recsys.training.results_aggregator import aggregate, to_summary_md


def _write_synthetic_parquet(
    path: Path,
    *,
    num_sessions: int = 10,
    steps_per_session: int = 3,
    slate_size: int = 3,
    num_candidates: int = 20,
    user_dim: int = 4,
    item_dim: int = 3,
) -> None:
    rng = np.random.default_rng(0)
    rows: list[dict] = []
    for sid in range(num_sessions):
        for seq in range(steps_per_session):
            slate = rng.choice(num_candidates, size=slate_size, replace=False)
            rows.append({
                "session_id": sid,
                "sequence_id": seq,
                "user_state": np.zeros(user_dim, dtype=np.float64).tolist(),
                "item_features": np.eye(num_candidates, item_dim)[slate].tolist(),
                "slate": slate.astype(np.int64).tolist(),
                "user_feedback": rng.integers(0, 2, size=slate_size).astype(np.int64).tolist(),
            })
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def test_agent_grid_smoke_end_to_end(tmp_path):
    parquet = tmp_path / "synth.parquet"
    _write_synthetic_parquet(parquet)

    behavior = BehaviorPolicy(
        slate_size=3, user_dim=4, item_dim=3, num_items=20, device="cpu",
    )
    src = RL4RSTrajectoryOPESource(parquet, behavior, slate_size=3)

    runs = [
        GridRun(agent_name="random", seed=0, pretrained=False),
        GridRun(agent_name="linucb", seed=0, pretrained=True),
    ]
    written = run_grid(
        runs,
        train_source_factory=lambda s: src,
        eval_source_factory=lambda s: src,
        env_kwargs=dict(slate_size=3, user_dim=4, item_dim=3, num_candidates=20),
        output_dir=tmp_path / "out",
        max_trajectories=10,
    )
    assert len(written) == 2

    df = aggregate(tmp_path / "out")
    assert {"random", "linucb"} <= set(df["agent"])
    md = to_summary_md(df)
    assert "random" in md and "linucb" in md
