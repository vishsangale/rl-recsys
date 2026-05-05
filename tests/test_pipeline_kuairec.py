import pandas as pd
import pytest

from rl_recsys.data.pipelines.kuairec import KuaiRecPipeline


def test_instantiation_sets_dirs(tmp_path):
    p = KuaiRecPipeline(raw_dir=str(tmp_path / "raw"),
                        processed_dir=str(tmp_path / "proc"))
    assert p.raw_dir.name == "raw"
    assert p.processed_dir.name == "proc"


def test_process_produces_correct_schema(tmp_path):
    raw_dir = tmp_path / "raw"
    extracted = raw_dir / "KuaiRec 2.0" / "data"
    extracted.mkdir(parents=True)
    (extracted / "big_matrix.csv").write_text(
        "user_id,video_id,play_duration,video_duration,time,date,timestamp,watch_ratio\n"
        "0,100,30.0,60.0,1609459200,2021-01-01,1609459200,0.5\n"
        "1,101,45.0,90.0,1609459260,2021-01-01,1609459260,0.5\n"
    )
    proc_dir = tmp_path / "proc"
    p = KuaiRecPipeline(raw_dir=str(raw_dir), processed_dir=str(proc_dir))
    p.process()

    out = proc_dir / "interactions.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert set(df.columns) >= {"user_id", "item_id", "rating", "timestamp"}
    assert df["user_id"].dtype.kind in ("i", "u")
    assert df["item_id"].dtype.kind in ("i", "u")
    assert df["rating"].ge(0.0).all()
    assert df["rating"].le(1.0).all()


def test_kuairec_is_registered():
    import rl_recsys.data.pipelines.kuairec  # noqa: F401
    from rl_recsys.data.registry import _REGISTRY
    assert "kuairec" in _REGISTRY


def test_process_emits_item_features_parquet(tmp_path):
    raw_dir = tmp_path / "raw"
    extracted = raw_dir / "KuaiRec 2.0" / "data"
    extracted.mkdir(parents=True)
    (extracted / "big_matrix.csv").write_text(
        "user_id,video_id,play_duration,video_duration,time,date,timestamp,watch_ratio\n"
        "0,100,30.0,60.0,1609459200,2021-01-01,1609459200,0.5\n"
        "1,101,45.0,90.0,1609459260,2021-01-01,1609459260,0.5\n"
    )
    (extracted / "item_categories.csv").write_text(
        "video_id,feat\n"
        '100,"[1, 2]"\n'
        '101,"[3, 4]"\n'
    )
    proc_dir = tmp_path / "proc"
    p = KuaiRecPipeline(raw_dir=str(raw_dir), processed_dir=str(proc_dir))
    p.process()

    out = proc_dir / "item_features.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert "item_id" in df.columns
    assert len(df) == 2
    feat_cols = [c for c in df.columns if c != "item_id"]
    assert set(feat_cols) == {"cat_1", "cat_2", "cat_3", "cat_4"}
    row100 = df[df["item_id"] == 100].iloc[0]
    assert row100["cat_1"] == 1 and row100["cat_2"] == 1
    assert row100["cat_3"] == 0 and row100["cat_4"] == 0


def test_process_item_features_handles_missing_feat(tmp_path):
    raw_dir = tmp_path / "raw"
    extracted = raw_dir / "KuaiRec 2.0" / "data"
    extracted.mkdir(parents=True)
    (extracted / "big_matrix.csv").write_text(
        "user_id,video_id,play_duration,video_duration,time,date,timestamp,watch_ratio\n"
        "0,100,30.0,60.0,1609459200,2021-01-01,1609459200,0.5\n"
    )
    (extracted / "item_categories.csv").write_text(
        "video_id,feat\n"
        '100,"[1, 2]"\n'
        "101,\n"  # missing feat — pandas reads as NaN
    )
    proc_dir = tmp_path / "proc"
    p = KuaiRecPipeline(raw_dir=str(raw_dir), processed_dir=str(proc_dir))
    p.process()
    df = pd.read_parquet(proc_dir / "item_features.parquet")
    assert len(df) == 2
    row101 = df[df["item_id"] == 101].iloc[0]
    assert row101["cat_1"] == 0 and row101["cat_2"] == 0


def test_process_skips_item_features_when_categories_absent(tmp_path):
    raw_dir = tmp_path / "raw"
    extracted = raw_dir / "KuaiRec 2.0" / "data"
    extracted.mkdir(parents=True)
    (extracted / "big_matrix.csv").write_text(
        "user_id,video_id,play_duration,video_duration,time,date,timestamp,watch_ratio\n"
        "0,100,30.0,60.0,1609459200,2021-01-01,1609459200,0.5\n"
    )
    # no item_categories.csv — should not raise
    proc_dir = tmp_path / "proc"
    p = KuaiRecPipeline(raw_dir=str(raw_dir), processed_dir=str(proc_dir))
    p.process()
    assert not (proc_dir / "item_features.parquet").exists()
