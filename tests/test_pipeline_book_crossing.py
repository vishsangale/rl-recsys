import pandas as pd
import pytest
from rl_recsys.data.pipelines.book_crossing import BookCrossingPipeline


def test_instantiation_sets_dirs(tmp_path):
    p = BookCrossingPipeline(raw_dir=str(tmp_path / "raw"),
                             processed_dir=str(tmp_path / "proc"))
    assert p.raw_dir.name == "raw"
    assert p.processed_dir.name == "proc"


def test_process_produces_correct_schema(tmp_path):
    raw_dir = tmp_path / "raw"
    extracted = raw_dir / "BX-CSV-Dump"
    extracted.mkdir(parents=True)
    (extracted / "BX-Book-Ratings.csv").write_text(
        '"User-ID";"ISBN";"Book-Rating"\n'
        '276725;"034545104X";0\n'
        '276726;"0155061224";5\n'
    )
    proc_dir = tmp_path / "proc"
    p = BookCrossingPipeline(raw_dir=str(raw_dir), processed_dir=str(proc_dir))
    p.process()

    out = proc_dir / "ratings.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert set(df.columns) >= {"user_id", "item_id", "rating", "timestamp"}
    assert df["user_id"].dtype.kind in ("i", "u")
    assert df["item_id"].dtype.kind in ("i", "u")
    assert (df["timestamp"] == 0).all()


def test_book_crossing_is_registered():
    import rl_recsys.data.pipelines.book_crossing  # noqa: F401
    from rl_recsys.data.registry import _REGISTRY
    assert "book-crossing" in _REGISTRY
