from __future__ import annotations

import ast
import gzip
from pathlib import Path

import pandas as pd

from rl_recsys.data.download import download_file
from rl_recsys.data.pipelines.base import BasePipeline
from rl_recsys.data.schema import validate_parquet_schema

_URL = "https://cseweb.ucsd.edu/~jmcauley/datasets/steam/steam_reviews.json.gz"


class SteamPipeline(BasePipeline):
    """Steam reviews pipeline.

    Source lines use Python literal syntax rather than JSON. Hours played is
    used as the implicit rating.
    """

    def __init__(
        self,
        raw_dir: str | Path = "data/raw/steam",
        processed_dir: str | Path = "data/processed/steam",
    ) -> None:
        super().__init__(raw_dir, processed_dir)

    def download(self) -> None:
        dest = self.raw_dir / "steam_reviews.json.gz"
        download_file(_URL, dest)

    def process(self) -> None:
        gz_file = self.raw_dir / "steam_reviews.json.gz"
        if not gz_file.exists():
            raise FileNotFoundError(f"Not found: {gz_file}. Run --download first.")

        records = []
        with gzip.open(gz_file, "rb") as f:
            for line in f:
                try:
                    records.append(ast.literal_eval(line.decode("utf-8")))
                except Exception:
                    continue

        df = pd.DataFrame(records)
        df = df.dropna(subset=["user_id", "product_id"])
        df["user_id"] = pd.factorize(df["user_id"])[0]
        df["item_id"] = pd.factorize(df["product_id"])[0]
        if "hours" in df.columns:
            df["rating"] = pd.to_numeric(df["hours"], errors="coerce").fillna(0.0)
        else:
            df["rating"] = 0.0

        if "date" in df.columns:
            parsed_dates = pd.to_datetime(df["date"], format="%b %d, %Y", errors="coerce")
            timestamps = parsed_dates.astype("int64") // 10**9
            df["timestamp"] = timestamps.where(parsed_dates.notna(), 0).astype("int64")
        else:
            df["timestamp"] = 0

        out = self.processed_dir / "interactions.parquet"
        df[["user_id", "item_id", "rating", "timestamp"]].to_parquet(out, index=False)
        validate_parquet_schema(out, "interactions")
        print(f"Saved {len(df):,} rows to {out}")


from rl_recsys.data.registry import register  # noqa: E402

register(
    "steam",
    SteamPipeline,
    schema="interactions",
    tags=["CF"],
    raw_dir="data/raw/steam",
    processed_dir="data/processed/steam",
)
