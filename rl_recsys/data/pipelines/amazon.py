from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from rl_recsys.data.download import download_file
from rl_recsys.data.pipelines.base import BasePipeline
from rl_recsys.data.schema import validate_parquet_schema

_BASE_URL = (
    "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/{category}_5.json.gz"
)

_REGISTERED_CATEGORIES: dict[str, str] = {
    "amazon-books": "Books",
    "amazon-movies": "Movies_and_TV",
    "amazon-electronics": "Electronics",
    "amazon-video-games": "Video_Games",
}


class AmazonPipeline(BasePipeline):
    """Amazon Reviews 2018 5-core pipeline parameterized by category."""

    def __init__(
        self,
        category: str = "Books",
        raw_dir: str | Path = "data/raw/amazon",
        processed_dir: str | Path = "data/processed/amazon",
    ) -> None:
        self.category = category
        super().__init__(raw_dir, processed_dir)

    def download(self) -> None:
        url = _BASE_URL.format(category=self.category)
        dest = self.raw_dir / f"{self.category}_5.json.gz"
        download_file(url, dest)

    def process(self) -> None:
        gz_file = self.raw_dir / f"{self.category}_5.json.gz"
        if not gz_file.exists():
            raise FileNotFoundError(f"Not found: {gz_file}. Run --download first.")

        records = []
        with gzip.open(gz_file, "rb") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        df = pd.DataFrame(records)
        df = df.dropna(subset=["reviewerID", "asin", "overall"])
        df["user_id"] = pd.factorize(df["reviewerID"])[0]
        df["item_id"] = pd.factorize(df["asin"])[0]
        df["rating"] = df["overall"].astype(float)
        if "unixReviewTime" in df.columns:
            df["timestamp"] = (
                pd.to_numeric(df["unixReviewTime"], errors="coerce")
                .fillna(0)
                .astype("int64")
            )
        else:
            df["timestamp"] = 0

        out = self.processed_dir / "interactions.parquet"
        df[["user_id", "item_id", "rating", "timestamp"]].to_parquet(out, index=False)
        validate_parquet_schema(out, "interactions")
        print(f"Saved {len(df):,} rows to {out} (category={self.category})")


from rl_recsys.data.registry import register  # noqa: E402

for _key, _category in _REGISTERED_CATEGORIES.items():
    register(
        _key,
        AmazonPipeline,
        schema="interactions",
        tags=["CF"],
        category=_category,
        raw_dir=f"data/raw/amazon/{_category}",
        processed_dir=f"data/processed/amazon/{_category}",
    )
