import pandas as pd
import pytest

from rl_recsys.data.pipelines.open_bandit import OpenBanditPipeline


def test_instantiation_sets_dirs(tmp_path):
    p = OpenBanditPipeline(raw_dir=str(tmp_path / "raw"),
                           processed_dir=str(tmp_path / "proc"))
    assert p.raw_dir.name == "raw"
    assert p.processed_dir.name == "proc"


def test_process_produces_correct_schema(tmp_path):
    raw_dir = tmp_path / "raw"
    random_all_dir = raw_dir / "open_bandit_dataset" / "random" / "all"
    random_all_dir.mkdir(parents=True)
    (random_all_dir / "all.csv").write_text(
        "timestamp,item_id,position,click,propensity_score,user_feature_0\n"
        "1609459200,42,0,1,0.33,0.5\n"
        "1609459260,43,1,0,0.33,0.7\n"
    )
    bts_women_dir = raw_dir / "open_bandit_dataset" / "bts" / "women"
    bts_women_dir.mkdir(parents=True)
    (bts_women_dir / "women.csv").write_text(
        "timestamp,item_id,position,click,propensity_score,user_feature_0\n"
        "1609459320,44,2,1,0.25,0.2\n"
    )
    proc_dir = tmp_path / "proc"
    p = OpenBanditPipeline(
        raw_dir=str(raw_dir),
        processed_dir=str(proc_dir),
        chunksize=1,
    )
    p.process()

    out = proc_dir / "interactions.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert set(df.columns) >= {"user_id", "item_id", "rating", "timestamp"}
    assert set(df.columns) >= {"propensity_score", "position", "policy", "campaign"}
    assert len(df) == 3
    assert (df["rating"].isin([0, 1])).all()
    assert (df["user_id"] == 0).all()
    assert set(df["policy"]) == {"random", "bts"}
    assert set(df["campaign"]) == {"all", "women"}


def test_open_bandit_is_registered():
    import rl_recsys.data.pipelines.open_bandit  # noqa: F401
    from rl_recsys.data.registry import _REGISTRY
    assert "open-bandit" in _REGISTRY
