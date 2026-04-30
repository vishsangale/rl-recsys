import pandas as pd
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
