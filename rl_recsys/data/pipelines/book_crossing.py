from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from rl_recsys.data.download import download_file
from rl_recsys.data.pipelines.base import BasePipeline
from rl_recsys.data.schema import validate_parquet_schema

_URL = "http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip"


class BookCrossingPipeline(BasePipeline):
    """Book-Crossing dataset: 1.1M ratings, 278K users, 271K books.

    No timestamp in source; timestamp column is filled with 0.
    """

    def __init__(
        self,
        raw_dir: str | Path = "data/raw/book_crossing",
        processed_dir: str | Path = "data/processed/book_crossing",
    ) -> None:
        super().__init__(raw_dir, processed_dir)

    def download(self) -> None:
        archive = self.raw_dir / "BX-CSV-Dump.zip"
        download_file(_URL, archive)
        print(f"Extracting to {self.raw_dir}...")
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(self.raw_dir)

    def process(self) -> None:
        ratings_file = self.raw_dir / "BX-CSV-Dump" / "BX-Book-Ratings.csv"
        if not ratings_file.exists():
            raise FileNotFoundError(
                f"Not found: {ratings_file}. Run --download first."
            )
        df = pd.read_csv(ratings_file, sep=";", encoding="latin-1", on_bad_lines="skip")
        df.columns = ["user_id_raw", "isbn", "rating"]
        df["user_id"] = pd.factorize(df["user_id_raw"])[0]
        df["item_id"] = pd.factorize(df["isbn"])[0]
        df["timestamp"] = 0
        out = self.processed_dir / "ratings.parquet"
        df[["user_id", "item_id", "rating", "timestamp"]].to_parquet(out, index=False)
        validate_parquet_schema(out, "interactions")
        print(f"Saved to {out}")


from rl_recsys.data.registry import register  # noqa: E402

register(
    "book-crossing",
    BookCrossingPipeline,
    schema="interactions",
    tags=["CF", "books"],
    raw_dir="data/raw/book_crossing",
    processed_dir="data/processed/book_crossing",
)
