import numpy as np
import pandas as pd

from rl_recsys.data.pipelines.finn_no_slate import FinnNoSlatePipeline


def _write_fake_data_npz(path, n_users=3, n_interactions=20, slate_size=25):
    """Write a minimal data.npz matching the current finn-no/recsys_slates_dataset format.

    Shapes: userId[N], click[N,T], click_idx[N,T], slate_lengths[N,T],
            slate[N,T,S], interaction_type[N,T].
    click=0 → PAD, click=1 → noClick, click>1 → real click.
    We set click>1 for every (user, interaction) so all rows survive the filter.
    """
    rng = np.random.default_rng(0)
    user_ids = np.arange(n_users, dtype=np.int32)
    slates = rng.integers(2, 1000, size=(n_users, n_interactions, slate_size), dtype=np.int64)
    # click > 1 so all rows are kept; use item ID 5 as clicked item
    click = np.full((n_users, n_interactions), 5, dtype=np.int64)
    click_idx = np.zeros((n_users, n_interactions), dtype=np.int64)  # position 0
    slate_lengths = np.full((n_users, n_interactions), slate_size, dtype=np.int64)
    interaction_type = np.ones((n_users, n_interactions), dtype=np.int64)
    np.savez(
        path,
        userId=user_ids,
        click=click,
        click_idx=click_idx,
        slate_lengths=slate_lengths,
        slate=slates,
        interaction_type=interaction_type,
    )


def test_instantiation_sets_dirs(tmp_path):
    p = FinnNoSlatePipeline(raw_dir=str(tmp_path / "raw"),
                            processed_dir=str(tmp_path / "proc"))
    assert p.raw_dir.name == "raw"
    assert p.processed_dir.name == "proc"


def test_process_produces_correct_schema(tmp_path):
    n_users, n_interactions = 3, 20
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_fake_data_npz(raw_dir / "data.npz", n_users=n_users, n_interactions=n_interactions)
    proc_dir = tmp_path / "proc"
    p = FinnNoSlatePipeline(raw_dir=str(raw_dir), processed_dir=str(proc_dir))
    p.process()

    out = proc_dir / "slates.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert set(df.columns) >= {"request_id", "user_id", "slate", "clicks"}
    # all n_users * n_interactions rows survive (click > 1 everywhere)
    assert len(df) == n_users * n_interactions
    assert df["user_id"].dtype.kind in ("i", "u")
    assert isinstance(df["slate"].iloc[0], (list, np.ndarray))
    assert len(df["slate"].iloc[0]) == 25
    assert df["clicks"].between(0, 24).all()


def test_finn_no_slate_is_registered():
    import rl_recsys.data.pipelines.finn_no_slate  # noqa: F401
    from rl_recsys.data.registry import _REGISTRY
    assert "finn-no-slate" in _REGISTRY
