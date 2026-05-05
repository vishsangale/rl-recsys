from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from rl_recsys.environments.factory import build_env
from rl_recsys.environments.finn_no_slate import FinnNoSlateEnv
from rl_recsys.environments.kuairec import KuaiRecEnv
from rl_recsys.environments.logged import LoggedInteractionEnv
from rl_recsys.environments.rl4rs import RL4RSEnv
from rl_recsys.environments.synthetic import SyntheticEnv


# ── fixtures ──────────────────────────────────────────────────────────────────

def _make_kuairec(tmp_path, n_rows: int = 30, n_items: int = 60, n_feats: int = 4) -> None:
    rng = np.random.default_rng(0)
    pd.DataFrame({
        "user_id": rng.integers(0, 5, n_rows),
        "item_id": rng.integers(0, n_items, n_rows),
        "rating": rng.uniform(0, 1, n_rows).astype(np.float32),
        "timestamp": np.arange(n_rows),
    }).to_parquet(tmp_path / "interactions.parquet", index=False)
    feat_cols = {f"feat_{i}": rng.standard_normal(n_items).astype(np.float32) for i in range(n_feats)}
    pd.DataFrame({"item_id": np.arange(n_items), **feat_cols}).to_parquet(
        tmp_path / "item_features.parquet", index=False
    )


def _make_finn(tmp_path, n_rows: int = 30) -> None:
    rng = np.random.default_rng(0)
    rows = [
        {
            "request_id": i,
            "user_id": i % 5,
            "slate": list(range(i * 25, i * 25 + 25)),
            "clicks": int(rng.integers(0, 25)),
            "timestamp": 1_600_000_000 + i * 1000,
        }
        for i in range(n_rows)
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "slates.parquet", index=False)


def _make_rl4rs(tmp_path, n_sessions: int = 3, n_steps: int = 3,
                slate_size: int = 3, n_feats: int = 4) -> None:
    rng = np.random.default_rng(0)
    rows = [
        {
            "session_id": sid,
            "step": step,
            "user_state": rng.standard_normal(n_feats).tolist(),
            "slate": rng.integers(0, 100, size=slate_size).tolist(),
            "item_features": rng.standard_normal((slate_size, n_feats)).tolist(),
            "clicks": rng.integers(0, 2, size=slate_size).tolist(),
        }
        for sid in range(n_sessions)
        for step in range(n_steps)
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "sessions.parquet", index=False)


def _make_logged(tmp_path, n_rows: int = 100) -> None:
    rng = np.random.default_rng(0)
    pd.DataFrame({
        "user_id": rng.integers(0, 20, n_rows),
        "item_id": rng.integers(0, 200, n_rows),
        "rating": rng.uniform(1, 5, n_rows).astype(np.float32),
        "timestamp": np.arange(n_rows),
    }).to_parquet(tmp_path / "interactions.parquet", index=False)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_build_synthetic():
    cfg = OmegaConf.create({
        "type": "synthetic", "num_items": 100, "num_candidates": 10,
        "slate_size": 3, "user_dim": 8, "item_dim": 8, "position_bias_decay": 0.5,
    })
    env = build_env(cfg)
    assert isinstance(env, SyntheticEnv)
    assert env.slate_size == 3
    assert env.num_candidates == 10


def test_build_kuairec(tmp_path):
    _make_kuairec(tmp_path)
    cfg = OmegaConf.create({
        "type": "kuairec", "processed_dir": str(tmp_path),
        "slate_size": 3, "num_candidates": 10, "feature_dim": 4,
        "feature_source": "native", "seed": 0,
    })
    env = build_env(cfg)
    assert isinstance(env, KuaiRecEnv)
    assert env.slate_size == 3


def test_build_finn_no_slate(tmp_path):
    _make_finn(tmp_path)
    cfg = OmegaConf.create({
        "type": "finn_no_slate", "processed_dir": str(tmp_path),
        "slate_size": 5, "feature_dim": 8, "feature_source": "hashed", "seed": 0,
    })
    env = build_env(cfg)
    assert isinstance(env, FinnNoSlateEnv)
    assert env.slate_size == 5


def test_build_rl4rs(tmp_path):
    _make_rl4rs(tmp_path)
    cfg = OmegaConf.create({
        "type": "rl4rs", "processed_dir": str(tmp_path),
        "slate_size": 3, "feature_dim": 4, "feature_source": "native", "seed": 0,
    })
    env = build_env(cfg)
    assert isinstance(env, RL4RSEnv)
    assert env.slate_size == 3


def test_build_logged(tmp_path):
    _make_logged(tmp_path)
    cfg = OmegaConf.create({
        "type": "logged", "processed_dir": str(tmp_path),
        "filename": "interactions.parquet",
        "slate_size": 5, "num_candidates": 20, "feature_dim": 8,
        "rating_threshold": 3.0, "seed": 0,
    })
    env = build_env(cfg)
    assert isinstance(env, LoggedInteractionEnv)
    assert env.slate_size == 5


def test_unknown_type_raises():
    cfg = OmegaConf.create({"type": "not_a_real_env"})
    with pytest.raises(ValueError, match="Unknown env type"):
        build_env(cfg)


def test_logged_missing_file_raises(tmp_path):
    cfg = OmegaConf.create({
        "type": "logged", "processed_dir": str(tmp_path),
        "filename": "missing.parquet",
        "slate_size": 5, "num_candidates": 20, "feature_dim": 8,
        "rating_threshold": 3.0, "seed": 0,
    })
    with pytest.raises(FileNotFoundError):
        build_env(cfg)
