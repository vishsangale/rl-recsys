import pytest
from rl_recsys.data.pipelines.movielens import MovieLensPipeline


def test_default_variant_is_100k(tmp_path):
    p = MovieLensPipeline(raw_dir=str(tmp_path), processed_dir=str(tmp_path))
    assert p._variant == "100k"
    assert "ml-100k" in p._url


def test_variant_1m_sets_correct_url(tmp_path):
    p = MovieLensPipeline(variant="1m", raw_dir=str(tmp_path), processed_dir=str(tmp_path))
    assert "ml-1m" in p._url


def test_unknown_variant_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown variant"):
        MovieLensPipeline(variant="999m", raw_dir=str(tmp_path), processed_dir=str(tmp_path))


def test_all_five_variants_registered():
    import rl_recsys.data.pipelines.movielens  # noqa: F401 — triggers self-registration
    from rl_recsys.data.registry import _REGISTRY
    for key in ["movielens-100k", "movielens-1m", "movielens-10m", "movielens-20m", "movielens-25m"]:
        assert key in _REGISTRY, f"{key} not in registry"


def test_process_100k_produces_correct_schema(tmp_path):
    import pandas as pd
    raw_dir = tmp_path / "raw" / "ml-100k"
    raw_dir.mkdir(parents=True)
    # Synthesize the u.data file that MovieLens-100k uses (tab-separated)
    (raw_dir / "u.data").write_text("1\t1\t5\t881250949\n2\t1\t3\t891717742\n")

    p = MovieLensPipeline(
        variant="100k",
        raw_dir=str(tmp_path / "raw"),
        processed_dir=str(tmp_path / "proc"),
    )
    p.process()

    out = tmp_path / "proc" / "ratings_100k.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert set(df.columns) >= {"user_id", "item_id", "rating", "timestamp"}
    assert df["user_id"].tolist() == [0, 1]  # 1-indexed → 0-indexed
