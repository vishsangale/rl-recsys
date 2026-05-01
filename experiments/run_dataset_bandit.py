"""Compare random and LinUCB agents on sampled logged-interaction datasets."""
from __future__ import annotations

import argparse
import importlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from rl_recsys.agents import LinUCBAgent, RandomAgent
from rl_recsys.config import AgentConfig, EnvConfig
from rl_recsys.data.registry import get_dataset_info, get_pipeline
from rl_recsys.environments.logged import LoggedInteractionEnv
from rl_recsys.evaluation.bandit import evaluate_bandit_agent


PIPELINE_MODULES = [
    "rl_recsys.data.pipelines.movielens",
    "rl_recsys.data.pipelines.lastfm",
    "rl_recsys.data.pipelines.rl4rs",
    "rl_recsys.data.pipelines.book_crossing",
    "rl_recsys.data.pipelines.kuairec",
    "rl_recsys.data.pipelines.finn_no_slate",
    "rl_recsys.data.pipelines.open_bandit",
    "rl_recsys.data.pipelines.gowalla",
    "rl_recsys.data.pipelines.steam",
    "rl_recsys.data.pipelines.amazon",
]
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    _register_builtin_pipelines()
    args = _parse_args()
    rows: list[dict[str, float | int | str]] = []

    for dataset in args.datasets:
        info = get_dataset_info(dataset)
        if info.schema not in {"interactions", "sessions"}:
            raise ValueError(
                f"{dataset!r} has schema {info.schema!r}; this benchmark supports "
                "interaction/session tables, not native slate logs yet"
            )
        pipeline = get_pipeline(dataset)
        if args.download:
            pipeline.download()
        path = _processed_path(dataset, pipeline.processed_dir)
        if args.process or not path.exists():
            pipeline.process()
        if not path.exists():
            raise FileNotFoundError(f"processed dataset not found: {path}")

        df = pd.read_parquet(path)
        if args.max_rows is not None and len(df) > args.max_rows:
            df = df.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)

        env_cfg = EnvConfig(
            num_candidates=args.num_candidates,
            slate_size=args.slate_size,
            user_dim=args.feature_dim,
            item_dim=args.feature_dim,
        )
        for agent_name in ("random", "linucb"):
            env = LoggedInteractionEnv(
                df,
                slate_size=args.slate_size,
                num_candidates=args.num_candidates,
                feature_dim=args.feature_dim,
                rating_threshold=args.rating_threshold,
                seed=args.seed,
            )
            agent_cfg = AgentConfig(name=agent_name, alpha=args.alpha)
            if agent_name == "random":
                agent = RandomAgent(slate_size=env_cfg.slate_size, seed=args.seed)
            else:
                agent = LinUCBAgent(
                    slate_size=env_cfg.slate_size,
                    user_dim=env_cfg.user_dim,
                    item_dim=env_cfg.item_dim,
                    alpha=agent_cfg.alpha,
                )
            result = evaluate_bandit_agent(
                env,
                agent,
                agent_name=agent_name,
                episodes=args.episodes,
                seed=args.seed,
            )
            row = result.as_dict()
            row.update(
                {
                    "dataset": dataset,
                    "rows": int(len(df)),
                    "num_candidates": args.num_candidates,
                    "slate_size": args.slate_size,
                    "feature_dim": args.feature_dim,
                    "rating_threshold": args.rating_threshold,
                }
            )
            rows.append(row)

    summary = pd.DataFrame(rows)
    output_dir = args.output_dir / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "summary.csv"
    summary.to_csv(output_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved summary to {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["movielens-100k"],
        help="Dataset keys to benchmark.",
    )
    parser.add_argument("--download", action="store_true", help="Download raw files.")
    parser.add_argument("--process", action="store_true", help="Process raw files.")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--num-candidates", type=int, default=50)
    parser.add_argument("--slate-size", type=int, default=10)
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--rating-threshold", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT / "results" / "rl-recsys" / "dataset_bandit",
    )
    return parser.parse_args()


def _register_builtin_pipelines() -> None:
    for module in PIPELINE_MODULES:
        importlib.import_module(module)


def _processed_path(dataset: str, processed_dir: Path) -> Path:
    if dataset.startswith("movielens-"):
        variant = dataset.removeprefix("movielens-")
        return processed_dir / f"ratings_{variant}.parquet"

    preferred = [
        processed_dir / "interactions.parquet",
        processed_dir / "ratings.parquet",
        processed_dir / "sessions.parquet",
    ]
    for path in preferred:
        if path.exists():
            return path
    parquet_files = sorted(processed_dir.glob("*.parquet"))
    if len(parquet_files) == 1:
        return parquet_files[0]
    return preferred[0]


if __name__ == "__main__":
    main()
