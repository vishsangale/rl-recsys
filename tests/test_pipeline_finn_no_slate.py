import numpy as np
import pandas as pd
import pytest

from rl_recsys.data.pipelines.finn_no_slate import FinnNoSlatePipeline


def _write_fake_npz(path, n=3):
    np.savez(
        path,
        userId=np.arange(n, dtype=np.int64),
        click=np.zeros(n, dtype=np.int64),
        slate=np.arange(n * 25, dtype=np.int64).reshape(n, 25),
        timestamps=np.array([1600000000 + i * 1000 for i in range(n)], dtype=np.int64),
        interaction_type=np.zeros(n, dtype=np.int64),
    )


def test_instantiation_sets_dirs(tmp_path):
    p = FinnNoSlatePipeline(raw_dir=str(tmp_path / "raw"),
                            processed_dir=str(tmp_path / "proc"))
    assert p.raw_dir.name == "raw"
    assert p.processed_dir.name == "proc"


def test_process_produces_correct_schema(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_fake_npz(raw_dir / "train.npz")
    _write_fake_npz(raw_dir / "test.npz")
    proc_dir = tmp_path / "proc"
    p = FinnNoSlatePipeline(raw_dir=str(raw_dir), processed_dir=str(proc_dir))
    p.process()

    out = proc_dir / "slates.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert set(df.columns) >= {"request_id", "user_id", "slate", "clicks", "timestamp"}
    assert len(df) == 6
    assert df["user_id"].dtype.kind in ("i", "u")
    assert isinstance(df["slate"].iloc[0], (list, np.ndarray))
    assert len(df["slate"].iloc[0]) == 25


def test_finn_no_slate_is_registered():
    import rl_recsys.data.pipelines.finn_no_slate  # noqa: F401
    from rl_recsys.data.registry import _REGISTRY
    assert "finn-no-slate" in _REGISTRY
