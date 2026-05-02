from __future__ import annotations

import pandas as pd
import pytest

from rl_recsys.data.pipelines.rl4rs import RL4RSPipeline


def _write_rl_csv(path, n_sessions=2, n_steps=3):
    """Write a minimal rl4rs_dataset_a_rl.csv with 2 user feats, 3 items, 2 item feats."""
    import io, random
    random.seed(0)
    rows = []
    for sid in range(n_sessions):
        for _ in range(n_steps):
            row = {"session_id": sid}
            row["user_feat_0"] = round(random.random(), 4)
            row["user_feat_1"] = round(random.random(), 4)
            for k in range(3):
                row[f"item_id_{k}"] = random.randint(100, 999)
                row[f"item_{k}_feat_0"] = round(random.random(), 4)
                row[f"item_{k}_feat_1"] = round(random.random(), 4)
                row[f"click_{k}"] = random.randint(0, 1)
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_process_produces_sessions_parquet(tmp_path):
    raw_dir = tmp_path / "raw" / "rl4rs-dataset"
    raw_dir.mkdir(parents=True)
    _write_rl_csv(raw_dir / "rl4rs_dataset_a_rl.csv", n_sessions=2, n_steps=3)
    proc_dir = tmp_path / "proc"
    p = RL4RSPipeline(raw_dir=str(tmp_path / "raw"), processed_dir=str(proc_dir))
    p.process()

    out = proc_dir / "sessions.parquet"
    assert out.exists()


def test_sessions_parquet_schema(tmp_path):
    raw_dir = tmp_path / "raw" / "rl4rs-dataset"
    raw_dir.mkdir(parents=True)
    _write_rl_csv(raw_dir / "rl4rs_dataset_a_rl.csv", n_sessions=2, n_steps=3)
    proc_dir = tmp_path / "proc"
    RL4RSPipeline(raw_dir=str(tmp_path / "raw"), processed_dir=str(proc_dir)).process()

    df = pd.read_parquet(proc_dir / "sessions.parquet")
    assert set(df.columns) >= {"session_id", "step", "user_state", "slate", "item_features", "clicks"}


def test_sessions_parquet_step_indices(tmp_path):
    raw_dir = tmp_path / "raw" / "rl4rs-dataset"
    raw_dir.mkdir(parents=True)
    _write_rl_csv(raw_dir / "rl4rs_dataset_a_rl.csv", n_sessions=2, n_steps=3)
    proc_dir = tmp_path / "proc"
    RL4RSPipeline(raw_dir=str(tmp_path / "raw"), processed_dir=str(proc_dir)).process()

    df = pd.read_parquet(proc_dir / "sessions.parquet")
    for sid, grp in df.groupby("session_id"):
        assert sorted(grp["step"].tolist()) == list(range(len(grp)))


def test_sessions_parquet_list_shapes(tmp_path):
    raw_dir = tmp_path / "raw" / "rl4rs-dataset"
    raw_dir.mkdir(parents=True)
    _write_rl_csv(raw_dir / "rl4rs_dataset_a_rl.csv", n_sessions=2, n_steps=3)
    proc_dir = tmp_path / "proc"
    RL4RSPipeline(raw_dir=str(tmp_path / "raw"), processed_dir=str(proc_dir)).process()

    df = pd.read_parquet(proc_dir / "sessions.parquet")
    row = df.iloc[0]
    assert len(row["user_state"]) == 2    # 2 user features in mock
    assert len(row["slate"]) == 3         # 3 items in mock
    assert len(row["item_features"]) == 3  # 3 items
    assert len(row["item_features"][0]) == 2  # 2 features per item
    assert len(row["clicks"]) == 3        # 3 click labels


def test_rl4rs_is_registered():
    import rl_recsys.data.pipelines.rl4rs  # noqa: F401
    from rl_recsys.data.registry import _REGISTRY
    assert "rl4rs" in _REGISTRY
