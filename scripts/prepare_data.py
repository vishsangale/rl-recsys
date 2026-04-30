# scripts/prepare_data.py
from __future__ import annotations

import argparse

# Import pipeline modules to trigger self-registration before list_datasets() is called
import rl_recsys.data.pipelines.movielens  # noqa: F401
import rl_recsys.data.pipelines.lastfm  # noqa: F401
import rl_recsys.data.pipelines.rl4rs  # noqa: F401
import rl_recsys.data.pipelines.book_crossing  # noqa: F401

from rl_recsys.data.registry import get_pipeline, list_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare datasets for rl-recsys.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list_datasets(),
        metavar="DATASET",
        help=f"Dataset key. Available: {list_datasets()}",
    )
    parser.add_argument("--raw-dir", type=str, default=None,
                        help="Override default raw data directory")
    parser.add_argument("--processed-dir", type=str, default=None,
                        help="Override default processed data directory")
    parser.add_argument("--download", action="store_true",
                        help="Download raw files")
    parser.add_argument("--process", action="store_true",
                        help="Process raw files into Parquet")

    args = parser.parse_args()
    pipeline = get_pipeline(
        args.dataset,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
    )

    if args.download:
        pipeline.download()
    if args.process:
        pipeline.process()


if __name__ == "__main__":
    main()
