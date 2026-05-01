import pandas as pd
import pytest

from experiments.run_ope_benchmark import _load_open_bandit_interactions


def _interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [0, 0, 0],
            "item_id": [1, 2, 3],
            "rating": [1.0, 0.0, 1.0],
            "timestamp": [1, 2, 3],
            "propensity_score": [0.5, 0.25, 0.75],
            "policy": ["random", "bts", "bts"],
            "campaign": ["all", "women", "men"],
        }
    )


def test_open_bandit_loader_filters_policy_and_campaign(tmp_path):
    path = tmp_path / "interactions.parquet"
    _interactions().to_parquet(path, index=False)

    df = _load_open_bandit_interactions(path, policy="bts", campaign="women")

    assert len(df) == 1
    assert df.iloc[0]["item_id"] == 2
    assert set(df["policy"]) == {"bts"}
    assert set(df["campaign"]) == {"women"}


def test_open_bandit_loader_supports_any_filter(tmp_path):
    path = tmp_path / "interactions.parquet"
    _interactions().to_parquet(path, index=False)

    df = _load_open_bandit_interactions(path, policy="bts", campaign="any")

    assert len(df) == 2
    assert set(df["campaign"]) == {"men", "women"}


def test_open_bandit_loader_handles_legacy_random_all_file(tmp_path):
    path = tmp_path / "interactions.parquet"
    _interactions().drop(columns=["policy", "campaign"]).to_parquet(path, index=False)

    df = _load_open_bandit_interactions(path, policy="random", campaign="all")

    assert len(df) == 3
    assert set(df["policy"]) == {"random"}
    assert set(df["campaign"]) == {"all"}


def test_open_bandit_loader_rejects_non_default_filter_for_legacy_file(tmp_path):
    path = tmp_path / "interactions.parquet"
    _interactions().drop(columns=["policy", "campaign"]).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="no policy/campaign columns"):
        _load_open_bandit_interactions(path, policy="bts", campaign="all")
