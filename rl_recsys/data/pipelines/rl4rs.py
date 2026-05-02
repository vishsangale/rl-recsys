from __future__ import annotations

import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from rl_recsys.data.pipelines.base import BasePipeline


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
        df = pd.read_csv(rl_file)
        cols = _detect_columns(df)

        df["step"] = df.groupby(cols["session_id"]).cumcount()
        df["user_state"] = df[cols["user_feat"]].values.tolist()
        df["slate"] = df[cols["item_id"]].values.tolist()
        n_items = len(cols["item_id"])
        arr = np.stack(
            [df[cols["item_feat"][i]].to_numpy() for i in range(n_items)], axis=1
        )
        df["item_features"] = arr.tolist()
        df["clicks"] = df[cols["click"]].values.tolist()

        out_df = df[
            [cols["session_id"], "step", "user_state", "slate", "item_features", "clicks"]
        ].rename(columns={cols["session_id"]: "session_id"})
        out = self.processed_dir / "sessions.parquet"
        out_df.to_parquet(out, index=False)
        print(f"Saved {len(out_df):,} rows ({out_df['session_id'].nunique():,} sessions) to {out}")


def _detect_columns(df: pd.DataFrame) -> dict:
    user_feat = sorted(
        [c for c in df.columns if re.match(r"^user_feat_\d+$", c)],
        key=lambda x: int(x.split("_")[-1]),
    )
    item_id = sorted(
        [c for c in df.columns if re.match(r"^item_id_\d+$", c)],
        key=lambda x: int(x.split("_")[-1]),
    )
    n_items = len(item_id)
    item_feat = [
        sorted(
            [c for c in df.columns if re.match(rf"^item_{i}_feat_\d+$", c)],
            key=lambda x: int(x.split("_")[-1]),
        )
        for i in range(n_items)
    ]
    click = sorted(
        [c for c in df.columns if re.match(r"^click_\d+$", c)],
        key=lambda x: int(x.split("_")[-1]),
    )
    session_id = "session_id"
    if session_id not in df.columns:
        raise ValueError(
            f"'session_id' column not found in CSV. Available columns: {list(df.columns)}"
        )
    if not user_feat:
        raise ValueError("No user_feat_N columns found in CSV.")
    if not item_id:
        raise ValueError("No item_id_N columns found in CSV.")
    if not click:
        raise ValueError("No click_N columns found in CSV.")
    return {
        "session_id": session_id,
        "user_feat": user_feat,
        "item_id": item_id,
        "item_feat": item_feat,
        "click": click,
    }


from rl_recsys.data.registry import register  # noqa: E402

register(
    "rl4rs",
    RL4RSPipeline,
    schema="slates",
    tags=["RL/Slate"],
    raw_dir="data/raw/rl4rs",
    processed_dir="data/processed/rl4rs",
)
