from __future__ import annotations

import argparse

from rl_recsys.data.pipelines.rl4rs import RL4RSPipeline
from rl_recsys.data.pipelines.movielens import MovieLensPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare datasets for rl-recsys.")
    parser.add_argument("--dataset", type=str, choices=["rl4rs", "movielens"], default="rl4rs")
    parser.add_argument("--download", action="store_true", help="Download raw files.")
    parser.add_argument("--process", action="store_true", help="Process raw files.")

    args = parser.parse_args()

    if args.dataset == "rl4rs":
        pipeline = RL4RSPipeline()
    elif args.dataset == "movielens":
        pipeline = MovieLensPipeline()
    else:
        print(f"Dataset {args.dataset} not yet implemented.")
        return

    if args.download:
        pipeline.download()
    if args.process:
        pipeline.process()


if __name__ == "__main__":
    main()
