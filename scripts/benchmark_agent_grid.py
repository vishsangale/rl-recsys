"""Run the (agent x seed x pretrained) ablation grid on RL4RS-B."""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from rl_recsys.agents.factory import AGENT_REGISTRY
from rl_recsys.data.loaders.rl4rs_trajectory_ope import RL4RSTrajectoryOPESource
from rl_recsys.evaluation.behavior_policy import (
    fit_behavior_policy_with_calibration,
)
from rl_recsys.training.agent_grid_runner import GridRun, run_grid
from rl_recsys.training.session_split import split_session_ids


def _expand_agents(arg: str) -> list[str]:
    if arg == "all":
        return [name for name in AGENT_REGISTRY if name != "oracle_click"]
    return [name.strip() for name in arg.split(",") if name.strip()]


def _expand_pretrained(arg: str) -> list[bool]:
    if arg == "both":
        return [False, True]
    if arg == "true":
        return [True]
    if arg == "false":
        return [False]
    raise ValueError(f"--pretrained-modes must be both|true|false, got {arg!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark (agent x seed x pretrained) grid on RL4RS-B.",
    )
    parser.add_argument(
        "--dataset", default="rl4rs_b",
        help="rl4rs_b is the only dataset wired today",
    )
    parser.add_argument(
        "--parquet", type=Path,
        default=Path("data/processed/rl4rs/sessions_b.parquet"),
    )
    parser.add_argument("--agents", default="all")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--pretrained-modes", default="both")
    parser.add_argument("--max-trajectories", type=int, default=5000)
    parser.add_argument("--boltzmann-T", type=float, default=1.0)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/agent_grid/2026-05-10"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--slate-size", type=int, default=9,
        help="RL4RS-B uses 9; override only for synthetic smoke tests",
    )
    parser.add_argument("--user-dim", type=int, default=10)
    parser.add_argument("--item-dim", type=int, default=10)
    args = parser.parse_args()

    if args.dataset != "rl4rs_b":
        raise NotImplementedError(
            "Only rl4rs_b is wired in sub-project #1; "
            "KuaiRec/finn/ML come later"
        )

    seeds = [int(s) for s in args.seeds.split(",")]
    agents = _expand_agents(args.agents)
    pretrained_modes = _expand_pretrained(args.pretrained_modes)

    train_ids, eval_ids = split_session_ids(args.parquet, train_fraction=0.5)

    # Auto-detect num_items from the parquet (mirrors benchmark_rl4rs_b_seq_dr).
    slate_table = pq.read_table(args.parquet, columns=["slate"])
    flat_items = slate_table["slate"].combine_chunks().flatten()
    num_items = int(pc.count_distinct(flat_items).as_py())

    behavior = fit_behavior_policy_with_calibration(
        args.parquet,
        user_dim=args.user_dim, item_dim=args.item_dim,
        slate_size=args.slate_size, num_items=num_items,
        epochs=5, batch_size=512, seed=0,
    )

    def train_factory(seed: int) -> RL4RSTrajectoryOPESource:
        return RL4RSTrajectoryOPESource(
            args.parquet, behavior, slate_size=args.slate_size,
            session_filter=train_ids,
        )

    def eval_factory(seed: int) -> RL4RSTrajectoryOPESource:
        return RL4RSTrajectoryOPESource(
            args.parquet, behavior, slate_size=args.slate_size,
            session_filter=eval_ids,
        )

    sample = train_factory(seeds[0])
    env_kwargs = dict(
        slate_size=args.slate_size,
        user_dim=args.user_dim,
        item_dim=args.item_dim,
        num_candidates=len(sample._candidate_ids),
    )

    runs = [
        GridRun(agent_name=a, seed=s, pretrained=p)
        for a in agents for s in seeds for p in pretrained_modes
    ]
    written = run_grid(
        runs,
        train_source_factory=train_factory,
        eval_source_factory=eval_factory,
        env_kwargs=env_kwargs,
        output_dir=args.output_dir,
        max_trajectories=args.max_trajectories,
        boltzmann_T=args.boltzmann_T,
        resume=args.resume,
        behavior_policy=behavior,
    )
    print(f"wrote {len(written)} run artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
