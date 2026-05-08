from __future__ import annotations

import tarfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from rl_recsys.data.pipelines.base import BasePipeline
from rl_recsys.data.schema import validate_parquet_schema


class RL4RSPipeline(BasePipeline):
    DATASET_URL = "https://zenodo.org/record/6622390/files/rl4rs-dataset.tar.gz"

    def __init__(
        self,
        raw_dir: str | Path = "data/raw/rl4rs",
        processed_dir: str | Path = "data/processed/rl4rs",
    ) -> None:
        super().__init__(raw_dir, processed_dir)

    def download(self) -> None:
        archive_path = self.raw_dir / "rl4rs-dataset.tar.gz"
        if not archive_path.exists():
            print(f"Downloading RL4RS dataset from {self.DATASET_URL}...")
            response = requests.get(self.DATASET_URL, stream=True)
            total_size = int(response.headers.get("content-length", 0))
            with open(archive_path, "wb") as f, tqdm(
                total=total_size, unit="iB", unit_scale=True
            ) as pbar:
                for data in response.iter_content(1024):
                    size = f.write(data)
                    pbar.update(size)
        else:
            print(f"Archive found at {archive_path}, skipping download.")
        print(f"Extracting to {self.raw_dir}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=self.raw_dir)
        print("Done.")

    def process(self) -> None:
        rl_file = self.raw_dir / "rl4rs-dataset" / "rl4rs_dataset_a_rl.csv"
        if not rl_file.exists():
            raise FileNotFoundError(
                f"Not found: {rl_file}. Run --download first."
            )
        df = pd.read_csv(rl_file, sep="@")

        # Parse comma-separated columns into lists / arrays
        df["slate"] = df["exposed_items"].apply(lambda s: [int(x) for x in str(s).split(",")])
        df["clicks"] = df["user_feedback"].apply(lambda s: [int(x) for x in str(s).split(",")])
        df["user_state"] = df["user_protrait"].apply(lambda s: [float(x) for x in str(s).split(",")])
        # item_feature: rows are semicolon-separated per-item vectors (comma-separated floats)
        df["item_features"] = df["item_feature"].apply(
            lambda s: [[float(v) for v in vec.split(",")] for vec in str(s).split(";")]
        )
        df["step"] = df.groupby("session_id").cumcount()

        out_df = df[["session_id", "step", "user_state", "slate", "item_features", "clicks"]]
        out = self.processed_dir / "sessions.parquet"
        out_df.to_parquet(out, index=False)
        validate_parquet_schema(out, "rl_sessions")
        print(f"Saved {len(out_df):,} rows ({out_df['session_id'].nunique():,} sessions) to {out}")

    def process_b(self) -> None:
        """Process dataset B (multi-step) into sessions_b.parquet.

        Groups raw rows by (session_id, sequence_id), parses CSV columns,
        and attaches a fixed candidate universe (all unique items in the
        slate column) to every row so loaders have a stable candidate set.
        """
        rl_file = self.raw_dir / "rl4rs-dataset" / "rl4rs_dataset_b_rl.csv"
        if not rl_file.exists():
            raise FileNotFoundError(
                f"Not found: {rl_file}. Run --download first."
            )
        df = pd.read_csv(rl_file, sep="@")
        df["slate"] = df["exposed_items"].apply(
            lambda s: [int(x) for x in str(s).split(",")]
        )
        df["user_feedback"] = df["user_feedback"].apply(
            lambda s: [int(x) for x in str(s).split(",")]
        )
        df["user_state"] = df["user_protrait"].apply(
            lambda s: [float(x) for x in str(s).split(",")]
        )
        df["item_features"] = df["item_feature"].apply(
            lambda s: [
                [float(v) for v in vec.split(",")]
                for vec in str(s).split(";")
            ]
        )

        universe_set: set[int] = set()
        for slate in df["slate"]:
            universe_set.update(slate)
        universe = sorted(universe_set)
        # Build a parallel feature lookup: pick each item's first observed feature vector.
        feature_for: dict[int, list[float]] = {}
        for slate, item_feats in zip(df["slate"], df["item_features"]):
            for item_id, feat in zip(slate, item_feats):
                if item_id not in feature_for:
                    feature_for[item_id] = list(feat)
        candidate_features = [feature_for[i] for i in universe]
        df["candidate_ids"] = [list(universe)] * len(df)
        df["candidate_features"] = [list(candidate_features)] * len(df)

        df = df.sort_values(["session_id", "sequence_id"], kind="stable")
        out_df = df[
            [
                "session_id", "sequence_id", "user_state", "slate",
                "item_features", "user_feedback",
                "candidate_ids", "candidate_features",
            ]
        ].reset_index(drop=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        out = self.processed_dir / "sessions_b.parquet"
        out_df.to_parquet(out, index=False)
        validate_parquet_schema(out, "rl_sessions_b")
        print(
            f"Saved {len(out_df):,} rows "
            f"({out_df['session_id'].nunique():,} sessions) to {out}"
        )


from rl_recsys.data.registry import register  # noqa: E402

register(
    "rl4rs",
    RL4RSPipeline,
    schema="rl_sessions",
    tags=["RL/Slate"],
    raw_dir="data/raw/rl4rs",
    processed_dir="data/processed/rl4rs",
)

register(
    "rl4rs_b",
    RL4RSPipeline,
    schema="rl_sessions_b",
    tags=["RL/Slate", "Multi-step"],
    raw_dir="data/raw/rl4rs",
    processed_dir="data/processed/rl4rs",
)
