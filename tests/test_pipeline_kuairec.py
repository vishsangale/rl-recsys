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
        "user_id,video_id,play_duration,video_duration,time,date,watch_ratio\n"
        "0,100,30.0,60.0,1609459200,2021-01-01,0.5\n"
        "1,101,45.0,90.0,1609459260,2021-01-01,0.5\n"
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


def test_kuairec_is_registered():
    import rl_recsys.data.pipelines.kuairec  # noqa: F401
    from rl_recsys.data.registry import _REGISTRY
    assert "kuairec" in _REGISTRY
