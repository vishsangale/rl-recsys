from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from rl_recsys.config import EnvConfig
from rl_recsys.environments.base import RecEnv
from rl_recsys.environments.finn_no_slate import FinnNoSlateEnv
from rl_recsys.environments.kuairec import KuaiRecEnv
from rl_recsys.environments.logged import LoggedInteractionEnv
from rl_recsys.environments.rl4rs import RL4RSEnv
from rl_recsys.environments.synthetic import SyntheticEnv


def build_env(cfg: DictConfig) -> RecEnv:
    """Build a RecEnv from a Hydra/OmegaConf config dict.

    The ``cfg.type`` field selects which environment to instantiate.
    Known types: ``synthetic``, ``kuairec``, ``finn_no_slate``, ``rl4rs``, ``logged``.
    """
    env_type = str(cfg.get("type", "synthetic"))

    if env_type == "synthetic":
        return SyntheticEnv(EnvConfig(
            num_items=int(cfg.get("num_items", 1000)),
            num_candidates=int(cfg.get("num_candidates", 50)),
            slate_size=int(cfg.get("slate_size", 10)),
            user_dim=int(cfg.get("user_dim", 32)),
            item_dim=int(cfg.get("item_dim", 32)),
            position_bias_decay=float(cfg.get("position_bias_decay", 0.5)),
        ))

    if env_type == "kuairec":
        return KuaiRecEnv(
            processed_dir=cfg.processed_dir,
            slate_size=int(cfg.get("slate_size", 6)),
            num_candidates=int(cfg.get("num_candidates", 50)),
            feature_dim=int(cfg.get("feature_dim", 32)),
            feature_source=str(cfg.get("feature_source", "native")),
            seed=int(cfg.get("seed", 0)),
        )

    if env_type == "finn_no_slate":
        return FinnNoSlateEnv(
            processed_dir=cfg.processed_dir,
            slate_size=int(cfg.get("slate_size", 5)),
            feature_dim=int(cfg.get("feature_dim", 16)),
            feature_source=str(cfg.get("feature_source", "hashed")),
            seed=int(cfg.get("seed", 0)),
        )

    if env_type == "rl4rs":
        return RL4RSEnv(
            processed_dir=cfg.processed_dir,
            slate_size=int(cfg.get("slate_size", 6)),
            feature_dim=int(cfg.get("feature_dim", 32)),
            feature_source=str(cfg.get("feature_source", "native")),
            seed=int(cfg.get("seed", 0)),
        )

    if env_type == "logged":
        parquet_path = Path(str(cfg.processed_dir)) / str(cfg.filename)
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Processed data not found: {parquet_path}. "
                "Run scripts/prepare_data.py first."
            )
        df = pd.read_parquet(parquet_path)
        return LoggedInteractionEnv(
            df,
            slate_size=int(cfg.get("slate_size", 10)),
            num_candidates=int(cfg.get("num_candidates", 50)),
            feature_dim=int(cfg.get("feature_dim", 16)),
            rating_threshold=float(cfg.get("rating_threshold", 4.0)),
            seed=int(cfg.get("seed", 0)),
        )

    raise ValueError(
        f"Unknown env type {env_type!r}. "
        "Known: synthetic, kuairec, finn_no_slate, rl4rs, logged"
    )
