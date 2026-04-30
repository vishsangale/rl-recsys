from __future__ import annotations

from pathlib import Path

import pandas as pd

from rl_recsys.data.download import download_file
from rl_recsys.data.pipelines.base import BasePipeline
from rl_recsys.data.schema import validate_parquet_schema

_URL = "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"


class GowallaPipeline(BasePipeline):
    """Gowalla location check-in pipeline.

    Source: https://snap.stanford.edu/data/loc-gowalla.html
    session_id is the row index; each check-in is treated independently.
    """

    def __init__(
        self,
        raw_dir: str | Path = "data/raw/gowalla",
        processed_dir: str | Path = "data/processed/gowalla",
    ) -> None:
        super().__init__(raw_dir, processed_dir)

    def download(self) -> None:
        dest = self.raw_dir / "loc-gowalla_totalCheckins.txt.gz"
        download_file(_URL, dest)

    def process(self) -> None:
        gz_file = self.raw_dir / "loc-gowalla_totalCheckins.txt.gz"
        if not gz_file.exists():
            raise FileNotFoundError(f"Not found: {gz_file}. Run --download first.")

        df = pd.read_csv(
            gz_file,
            sep="\t",
            header=None,
            names=["user_id", "checkin_time", "latitude", "longitude", "location_id"],
        )
        df["item_id"] = pd.factorize(df["location_id"])[0]
        df["timestamp"] = (
            pd.to_datetime(df["checkin_time"], utc=True).astype("int64") // 10**9
        )
        df["session_id"] = df.index.astype("int64")

        out = self.processed_dir / "sessions.parquet"
        df[["session_id", "user_id", "item_id", "timestamp"]].to_parquet(out, index=False)
        validate_parquet_schema(out, "sessions")
        print(f"Saved {len(df):,} rows to {out}")


from rl_recsys.data.registry import register  # noqa: E402

register(
    "gowalla",
    GowallaPipeline,
    schema="sessions",
    tags=["Session"],
    raw_dir="data/raw/gowalla",
    processed_dir="data/processed/gowalla",
)
