import gzip

import pandas as pd
import pytest

from rl_recsys.data.pipelines.steam import SteamPipeline


def test_instantiation_sets_dirs(tmp_path):
    p = SteamPipeline(raw_dir=str(tmp_path / "raw"),
                      processed_dir=str(tmp_path / "proc"))
    assert p.raw_dir.name == "raw"
    assert p.processed_dir.name == "proc"


def test_process_produces_correct_schema(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    lines = [
        "{'username': 'user1', 'user_id': 'u1', 'product_id': 'g1', 'hours': 10.5, 'date': 'Oct 1, 2011'}",
        "{'username': 'user2', 'user_id': 'u2', 'product_id': 'g2', 'hours': 0.0, 'date': 'Jan 5, 2013'}",
        "bad line that should be skipped",
    ]
    content = "\n".join(lines).encode()
    (raw_dir / "steam_reviews.json.gz").write_bytes(gzip.compress(content))
    proc_dir = tmp_path / "proc"
    p = SteamPipeline(raw_dir=str(raw_dir), processed_dir=str(proc_dir))
    p.process()

    out = proc_dir / "interactions.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert set(df.columns) >= {"user_id", "item_id", "rating", "timestamp"}
    assert len(df) == 2
    assert df["user_id"].dtype.kind in ("i", "u")
    assert df["item_id"].dtype.kind in ("i", "u")
    assert df["rating"].dtype.kind == "f"
    assert df["rating"].iloc[0] == pytest.approx(10.5)


def test_steam_is_registered():
    import rl_recsys.data.pipelines.steam  # noqa: F401
    from rl_recsys.data.registry import _REGISTRY
    assert "steam" in _REGISTRY
