from __future__ import annotations

import tarfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from rl_recsys.data.pipelines.base import BasePipeline


class LastFMPipeline(BasePipeline):
    """Pipeline for downloading and processing the Last.fm-1K dataset."""

    DATASET_URL = "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz"

    def __init__(self, raw_dir: str | Path = "data/raw/lastfm",
                 processed_dir: str | Path = "data/processed/lastfm") -> None:
        super().__init__(raw_dir, processed_dir)

    def download(self) -> None:
        """Download and extract the Last.fm-1K dataset."""
        archive_path = self.raw_dir / "lastfm-dataset-1K.tar.gz"

        if not archive_path.exists():
            print(f"Downloading Last.fm-1K from {self.DATASET_URL}...")
            response = requests.get(self.DATASET_URL, stream=True, timeout=60)
            response.raise_for_status()

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
        """Process Last.fm TSV logs into standard Parquet format."""
        # Expected file: lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv
        data_file = self.raw_dir / "lastfm-dataset-1K" / "userid-timestamp-artid-artname-traid-traname.tsv"

        if not data_file.exists():
            raise FileNotFoundError(f"Could not find {data_file}. Check extraction.")

        print(f"Processing {data_file.name}...")
        
        # TSV format: user | timestamp | artist-id | artist-name | track-id | track-name
        columns = ["user_id", "timestamp", "artist_id", "artist_name", "track_id", "track_name"]
        
        # Note: This dataset is large (~600MB TSV, 19M rows). 
        # We'll use chunking or just load if memory permits. 
        # For the sake of verification, let's load first 1M rows or all if possible.
        df = pd.read_csv(data_file, sep="	", names=columns, on_bad_lines='skip')

        # Basic processing: factorize IDs
        df["user_idx"] = pd.factorize(df["user_id"])[0]
        df["track_idx"] = pd.factorize(df["track_id"])[0]

        output_path = self.processed_dir / "interactions.parquet"
        df.to_parquet(output_path, index=False)
        print(f"Processed data saved to {output_path}")
