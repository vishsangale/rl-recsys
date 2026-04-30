import gzip
import json

import pandas as pd
import pytest

from rl_recsys.data.pipelines.amazon import AmazonPipeline


def _write_fake_reviews(path, n=3):
    lines = [
        json.dumps({
            "overall": float(i % 5 + 1),
            "reviewerID": f"R{i}",
            "asin": f"B00{i:03d}",
            "unixReviewTime": 1600000000 + i * 1000,
        })
        for i in range(n)
    ]
    path.write_bytes(gzip.compress("\n".join(lines).encode()))


def test_instantiation_default_category(tmp_path):
    p = AmazonPipeline(raw_dir=str(tmp_path / "raw"),
                       processed_dir=str(tmp_path / "proc"))
    assert p.category == "Books"


def test_instantiation_custom_category(tmp_path):
    p = AmazonPipeline(category="Electronics",
                       raw_dir=str(tmp_path / "raw"),
                       processed_dir=str(tmp_path / "proc"))
    assert p.category == "Electronics"


def test_process_produces_correct_schema(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_fake_reviews(raw_dir / "Books_5.json.gz")
    proc_dir = tmp_path / "proc"
    p = AmazonPipeline(category="Books",
                       raw_dir=str(raw_dir), processed_dir=str(proc_dir))
    p.process()

    out = proc_dir / "interactions.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert set(df.columns) >= {"user_id", "item_id", "rating", "timestamp"}
    assert df["user_id"].dtype.kind in ("i", "u")
    assert df["item_id"].dtype.kind in ("i", "u")
    assert df["rating"].between(1.0, 5.0).all()
    assert (df["timestamp"] > 0).all()


def test_amazon_categories_registered():
    import rl_recsys.data.pipelines.amazon  # noqa: F401
    from rl_recsys.data.registry import _REGISTRY
    for key in ("amazon-books", "amazon-movies", "amazon-electronics", "amazon-video-games"):
        assert key in _REGISTRY
