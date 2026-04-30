from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from rl_recsys.data.pipelines.base import BasePipeline


class MovieLensPipeline(BasePipeline):
    """Pipeline for downloading and processing the MovieLens-100k dataset."""

    DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

    def __init__(self, raw_dir: str | Path = "data/raw/movielens",
                 processed_dir: str | Path = "data/processed/movielens",
                 variant: str = "100k") -> None:
        super().__init__(raw_dir, processed_dir)
        self.variant = variant

    def download(self) -> None:
        """Download and extract the MovieLens-100k dataset."""
        archive_path = self.raw_dir / "ml-100k.zip"

        if not archive_path.exists():
            print(f"Downloading MovieLens-100k from {self.DATASET_URL}...")
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
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        print("Done.")

    def process(self) -> None:
        """Process MovieLens ratings into standard Parquet format."""
        data_file = self.raw_dir / "ml-100k" / "u.data"

        if not data_file.exists():
            raise FileNotFoundError(f"Could not find {data_file}. Check extraction.")

        print(f"Processing {data_file}...")
        # MovieLens 100k uses tab-separated u.data: user_id | item_id | rating | timestamp
        columns = ["user_id", "item_id", "rating", "timestamp"]
        df = pd.read_csv(data_file, sep="	", names=columns)

        # Basic processing: convert to 0-indexed IDs
        df["user_id"] = df["user_id"] - 1
        df["item_id"] = df["item_id"] - 1

        output_path = self.processed_dir / "ratings.parquet"
        df.to_parquet(output_path, index=False)
        print(f"Processed data saved to {output_path}")
