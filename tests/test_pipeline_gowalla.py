import gzip

import pandas as pd
import pytest

from rl_recsys.data.pipelines.gowalla import GowallaPipeline


def test_instantiation_sets_dirs(tmp_path):
    p = GowallaPipeline(raw_dir=str(tmp_path / "raw"),
                        processed_dir=str(tmp_path / "proc"))
    assert p.raw_dir.name == "raw"
    assert p.processed_dir.name == "proc"


def test_process_produces_correct_schema(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    content = (
        "196514\t2010-10-19T23:55:27Z\t30.2359091\t-97.7951395\t145064\n"
        "196514\t2010-10-18T22:17:43Z\t30.2691029\t-97.7493953\t1275991\n"
        "196515\t2010-10-19T20:20:17Z\t30.2749767\t-97.7403954\t145064\n"
    )
    gz_path = raw_dir / "loc-gowalla_totalCheckins.txt.gz"
    gz_path.write_bytes(gzip.compress(content.encode()))
    proc_dir = tmp_path / "proc"
    p = GowallaPipeline(raw_dir=str(raw_dir), processed_dir=str(proc_dir))
    p.process()

    out = proc_dir / "sessions.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert set(df.columns) >= {"session_id", "user_id", "item_id", "timestamp"}
    assert len(df) == 3
    assert df["user_id"].dtype.kind in ("i", "u")
    assert df["item_id"].dtype.kind in ("i", "u")
    assert df["timestamp"].dtype.kind in ("i", "u")
    assert df[df["user_id"] == 196514]["item_id"].nunique() == 2


def test_gowalla_is_registered():
    import rl_recsys.data.pipelines.gowalla  # noqa: F401
    from rl_recsys.data.registry import _REGISTRY
    assert "gowalla" in _REGISTRY
