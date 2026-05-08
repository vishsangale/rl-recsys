from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rl_recsys.data.schema import validate_parquet_schema


def test_validate_passes_for_correct_interactions(tmp_path):
    path = tmp_path / "ok.parquet"
    pd.DataFrame({"user_id": [0], "item_id": [1], "timestamp": [0]}).to_parquet(path)
    validate_parquet_schema(path, "interactions")  # must not raise


def test_validate_fails_for_missing_column(tmp_path):
    path = tmp_path / "bad.parquet"
    pd.DataFrame({"user_id": [0], "item_id": [1]}).to_parquet(path)
    with pytest.raises(ValueError, match="missing columns"):
        validate_parquet_schema(path, "interactions")


def test_validate_fails_for_unknown_schema_type(tmp_path):
    path = tmp_path / "any.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(path)
    with pytest.raises(ValueError, match="Unknown schema type"):
        validate_parquet_schema(path, "nonexistent")


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
