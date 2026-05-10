from pathlib import Path

import pandas as pd
import pytest

from rl_recsys.training.session_split import split_session_ids


def _make_parquet(tmp_path: Path, n_sessions: int = 1000) -> Path:
    rows = []
    for sid in range(n_sessions):
        rows.append({
            "session_id": sid, "sequence_id": 1,
            "user_state": [1.0, 0.0],
            "slate": [10, 11], "user_feedback": [1, 0],
            "item_features": [[0.0, 0.0], [1.0, 0.0]],
        })
    p = tmp_path / "sessions_b.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    return p


def test_split_is_deterministic(tmp_path: Path) -> None:
    p = _make_parquet(tmp_path, n_sessions=200)
    a_train, a_eval = split_session_ids(p, train_fraction=0.5, seed=42)
    b_train, b_eval = split_session_ids(p, train_fraction=0.5, seed=42)
    assert a_train == b_train
    assert a_eval == b_eval


def test_split_is_disjoint_and_complete(tmp_path: Path) -> None:
    p = _make_parquet(tmp_path, n_sessions=200)
    train, evl = split_session_ids(p, train_fraction=0.5, seed=42)
    assert train.isdisjoint(evl)
    assert train | evl == set(range(200))


def test_split_fraction_approximate(tmp_path: Path) -> None:
    p = _make_parquet(tmp_path, n_sessions=1000)
    train, evl = split_session_ids(p, train_fraction=0.5, seed=42)
    assert 450 <= len(train) <= 550
    assert 450 <= len(evl) <= 550


def test_split_rejects_invalid_fractions(tmp_path: Path) -> None:
    p = _make_parquet(tmp_path, n_sessions=10)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="train_fraction"):
            split_session_ids(p, train_fraction=bad, seed=0)


def test_split_rejects_empty_split(tmp_path: Path) -> None:
    rows = [
        {"session_id": 1, "sequence_id": 1, "user_state": [1.0, 0.0],
         "slate": [10, 11], "user_feedback": [1, 0],
         "item_features": [[0.0, 0.0], [1.0, 0.0]]},
        {"session_id": 2, "sequence_id": 1, "user_state": [1.0, 0.0],
         "slate": [10, 11], "user_feedback": [0, 0],
         "item_features": [[0.0, 0.0], [1.0, 0.0]]},
    ]
    p = tmp_path / "sessions_b.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)

    # With train_fraction=0.001 and n=2, both sessions almost always land
    # in the eval side (threshold=1, only bucket==0 is train). Sweep seeds
    # to find one where the train side is empty and confirm the raise.
    raised = False
    for seed in range(50):
        try:
            split_session_ids(p, train_fraction=0.001, seed=seed)
        except ValueError as e:
            if "empty" in str(e):
                raised = True
                break
    assert raised, "expected at least one seed/fraction combo to trigger empty-split error"
