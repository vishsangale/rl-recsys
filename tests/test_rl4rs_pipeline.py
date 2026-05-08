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
