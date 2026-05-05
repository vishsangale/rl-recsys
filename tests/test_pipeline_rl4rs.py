from __future__ import annotations

import pandas as pd
import pytest

from rl_recsys.data.pipelines.rl4rs import RL4RSPipeline


def _write_rl_csv(path, n_sessions=2, n_steps=3):
    """Write a minimal rl4rs_dataset_a_rl.csv in the real @-delimited format.

    Uses 2 user features, 3 items, 2 item features per item.
    """
    import random
    random.seed(0)
    rows = []
    for sid in range(n_sessions):
        for step in range(n_steps):
            item_ids = [random.randint(100, 999) for _ in range(3)]
            user_feats = [round(random.random(), 4) for _ in range(2)]
            item_feats = [[round(random.random(), 4) for _ in range(2)] for _ in range(3)]
            clicks = [random.randint(0, 1) for _ in range(3)]
            rows.append({
                "timestamp": 1000 * sid + step,
                "session_id": sid,
                "sequence_id": step,
                "exposed_items": ",".join(str(i) for i in item_ids),
                "user_feedback": ",".join(str(c) for c in clicks),
                "user_seqfeature": "0,0",
                "user_protrait": ",".join(str(f) for f in user_feats),
                "item_feature": ";".join(",".join(str(f) for f in vec) for vec in item_feats),
                "behavior_policy_id": 1,
                # store raw values for value-wiring test
                "_item_ids": item_ids,
                "_user_feats": user_feats,
                "_item_feats": item_feats,
                "_clicks": clicks,
            })
    meta = pd.DataFrame(rows)
    meta.drop(columns=[c for c in meta.columns if c.startswith("_")]).to_csv(
        path, index=False, sep="@"
    )
    return meta  # returned for value-wiring assertions


def _setup(tmp_path, n_sessions=2, n_steps=3):
    raw_dir = tmp_path / "raw" / "rl4rs-dataset"
    raw_dir.mkdir(parents=True)
    meta = _write_rl_csv(raw_dir / "rl4rs_dataset_a_rl.csv", n_sessions=n_sessions, n_steps=n_steps)
    proc_dir = tmp_path / "proc"
    RL4RSPipeline(raw_dir=str(tmp_path / "raw"), processed_dir=str(proc_dir)).process()
    return pd.read_parquet(proc_dir / "sessions.parquet"), meta


def test_process_produces_sessions_parquet(tmp_path):
    raw_dir = tmp_path / "raw" / "rl4rs-dataset"
    raw_dir.mkdir(parents=True)
    _write_rl_csv(raw_dir / "rl4rs_dataset_a_rl.csv")
    proc_dir = tmp_path / "proc"
    RL4RSPipeline(raw_dir=str(tmp_path / "raw"), processed_dir=str(proc_dir)).process()
    assert (proc_dir / "sessions.parquet").exists()


def test_sessions_parquet_schema(tmp_path):
    df, _ = _setup(tmp_path)
    assert set(df.columns) >= {"session_id", "step", "user_state", "slate", "item_features", "clicks"}


def test_sessions_parquet_step_indices(tmp_path):
    df, _ = _setup(tmp_path)
    for sid, grp in df.groupby("session_id"):
        assert sorted(grp["step"].tolist()) == list(range(len(grp)))


def test_sessions_parquet_list_shapes(tmp_path):
    df, _ = _setup(tmp_path)
    row = df.iloc[0]
    assert len(row["user_state"]) == 2    # 2 user features in mock
    assert len(row["slate"]) == 3         # 3 items in mock
    assert len(row["item_features"]) == 3  # 3 items
    assert len(row["item_features"][0]) == 2  # 2 features per item
    assert len(row["clicks"]) == 3        # 3 click labels


def test_sessions_parquet_value_wiring(tmp_path):
    df, meta = _setup(tmp_path)
    first = df.iloc[0]
    first_meta = meta.iloc[0]

    assert first["slate"][0] == first_meta["_item_ids"][0]
    assert first["user_state"][0] == pytest.approx(first_meta["_user_feats"][0])
    assert first["item_features"][0][0] == pytest.approx(first_meta["_item_feats"][0][0])
    assert first["clicks"][0] == first_meta["_clicks"][0]


def test_rl4rs_is_registered():
    import rl_recsys.data.pipelines.rl4rs  # noqa: F401
    from rl_recsys.data.registry import _REGISTRY
    assert "rl4rs" in _REGISTRY
