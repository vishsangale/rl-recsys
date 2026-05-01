"""Evaluate target recommendation policies with Open Bandit OPE estimators."""
from __future__ import annotations

import argparse
import importlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from rl_recsys.agents import LinUCBAgent, RandomAgent
from rl_recsys.config import AgentConfig, EnvConfig
from rl_recsys.data.registry import get_dataset_info, get_pipeline
from rl_recsys.environments.open_bandit import OpenBanditEventSampler
from rl_recsys.evaluation.ope import evaluate_ope_agent


PIPELINE_MODULES = ["rl_recsys.data.pipelines.open_bandit"]
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = "random"
DEFAULT_CAMPAIGN = "all"


def main() -> None:
    _register_builtin_pipelines()
    args = _parse_args()
    info = get_dataset_info(args.dataset)
    if args.dataset != "open-bandit" or info.schema != "interactions":
        raise ValueError("run_ope_benchmark.py currently supports only open-bandit")

    pipeline = get_pipeline(args.dataset)
    if args.download:
        pipeline.download()
    path = pipeline.processed_dir / "interactions.parquet"
    if args.process or not path.exists():
        pipeline.process()
    if not path.exists():
        raise FileNotFoundError(f"processed dataset not found: {path}")

    df = _load_open_bandit_interactions(path, policy=args.policy, campaign=args.campaign)
    if args.max_rows is not None and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)

    env_cfg = EnvConfig(
        num_candidates=args.num_candidates,
        slate_size=1,
        user_dim=args.feature_dim,
        item_dim=args.feature_dim,
    )
    rows: list[dict[str, float | int | str]] = []
    for agent_name in ("random", "linucb"):
        event_source = OpenBanditEventSampler(
            df,
            num_candidates=args.num_candidates,
            feature_dim=args.feature_dim,
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
        result = evaluate_ope_agent(
            event_source,
            agent,
            agent_name=agent_name,
            episodes=args.episodes,
            seed=args.seed,
        )
        row = result.as_dict()
        row.update(
            {
                "dataset": args.dataset,
                "rows": int(len(df)),
                "policy_filter": args.policy,
                "campaign_filter": args.campaign,
                "policies": ",".join(sorted(map(str, df["policy"].unique()))),
                "campaigns": ",".join(sorted(map(str, df["campaign"].unique()))),
                "num_candidates": args.num_candidates,
                "feature_dim": args.feature_dim,
                "seed": args.seed,
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
    parser.add_argument("--dataset", default="open-bandit")
    parser.add_argument("--download", action="store_true", help="Download raw files.")
    parser.add_argument("--process", action="store_true", help="Process raw files.")
    parser.add_argument(
        "--policy",
        choices=["random", "bts", "any"],
        default=DEFAULT_POLICY,
        help="Open Bandit behavior policy slice. Use 'any' for all policies.",
    )
    parser.add_argument(
        "--campaign",
        choices=["all", "men", "women", "any"],
        default=DEFAULT_CAMPAIGN,
        help="Open Bandit campaign slice. Use 'any' for all campaigns.",
    )
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--num-candidates", type=int, default=50)
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT / "results" / "rl-recsys" / "ope_benchmark",
    )
    return parser.parse_args()


def _register_builtin_pipelines() -> None:
    for module in PIPELINE_MODULES:
        importlib.import_module(module)


def _load_open_bandit_interactions(
    path: Path,
    *,
    policy: str,
    campaign: str,
) -> pd.DataFrame:
    schema_names = set(pq.read_schema(path).names)
    has_split_metadata = {"policy", "campaign"}.issubset(schema_names)

    if not has_split_metadata:
        if policy not in {"any", DEFAULT_POLICY} or campaign not in {"any", DEFAULT_CAMPAIGN}:
            raise ValueError(
                "processed Open Bandit file has no policy/campaign columns; "
                "rerun with --process before filtering non-default splits"
            )
        df = pd.read_parquet(path)
        df["policy"] = DEFAULT_POLICY
        df["campaign"] = DEFAULT_CAMPAIGN
        return df

    filters: list[tuple[str, str, str]] = []
    if policy != "any":
        filters.append(("policy", "==", policy))
    if campaign != "any":
        filters.append(("campaign", "==", campaign))

    df = pd.read_parquet(path, filters=filters or None)
    if df.empty:
        raise ValueError(
            f"no Open Bandit rows matched policy={policy!r}, campaign={campaign!r}"
        )
    return df.reset_index(drop=True)


if __name__ == "__main__":
    main()
