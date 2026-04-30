from __future__ import annotations

import tarfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from rl_recsys.data.pipelines.base import BasePipeline


class RL4RSPipeline(BasePipeline):
    """Pipeline for downloading and processing the RL4RS dataset."""

    DATASET_URL = "https://zenodo.org/record/6622390/files/rl4rs-dataset.tar.gz"

    def __init__(self, raw_dir: str | Path = "data/raw/rl4rs",
                 processed_dir: str | Path = "data/processed/rl4rs") -> None:
        super().__init__(raw_dir, processed_dir)

    def download(self) -> None:
        """Download and extract the RL4RS dataset."""
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

        # Extract
        print(f"Extracting to {self.raw_dir}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=self.raw_dir)
        print("Done.")

    def process(self) -> None:
        """Process RL4RS slate data into standard Parquet files."""
        # Find the _sl.csv and _rl.csv files in the extracted dir
        # Based on search: rl4rs-dataset/rl4rs_dataset_a_sl.csv
        slate_dir = self.raw_dir / "rl4rs-dataset"
        sl_file = slate_dir / "rl4rs_dataset_a_sl.csv"
        rl_file = slate_dir / "rl4rs_dataset_a_rl.csv"

        if not sl_file.exists():
            raise FileNotFoundError(f"Could not find {sl_file}. Check extraction.")

        print(f"Processing {sl_file}...")
        df_sl = pd.read_csv(sl_file)
        # Placeholder for real preprocessing (e.g. normalization, encoding)
        df_sl.to_parquet(self.processed_dir / "slate_train.parquet", index=False)

        print(f"Processing {rl_file}...")
        df_rl = pd.read_csv(rl_file)
        df_rl.to_parquet(self.processed_dir / "slate_eval.parquet", index=False)
        print(f"Processed data saved to {self.processed_dir}")


from rl_recsys.data.registry import register  # noqa: E402

register(
    "rl4rs",
    RL4RSPipeline,
    schema="slates",
    tags=["RL/Slate"],
    raw_dir="data/raw/rl4rs",
    processed_dir="data/processed/rl4rs",
)
