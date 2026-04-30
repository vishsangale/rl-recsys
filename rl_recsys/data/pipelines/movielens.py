from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from rl_recsys.data.download import download_file
from rl_recsys.data.pipelines.base import BasePipeline
from rl_recsys.data.schema import validate_parquet_schema


class MovieLensPipeline(BasePipeline):
    """GroupLens MovieLens pipeline supporting variants: 100k, 1m, 10m, 20m, 25m."""

    _VARIANTS: dict[str, tuple[str, str]] = {
        "100k": ("ml-100k.zip", "https://files.grouplens.org/datasets/movielens/ml-100k.zip"),
        "1m":   ("ml-1m.zip",   "https://files.grouplens.org/datasets/movielens/ml-1m.zip"),
        "10m":  ("ml-10M100K.zip", "https://files.grouplens.org/datasets/movielens/ml-10M100K.zip"),
        "20m":  ("ml-20m.zip",  "https://files.grouplens.org/datasets/movielens/ml-20m.zip"),
        "25m":  ("ml-25m.zip",  "https://files.grouplens.org/datasets/movielens/ml-25m.zip"),
    }

    def __init__(
        self,
        variant: str = "100k",
        raw_dir: str | Path = "data/raw/movielens",
        processed_dir: str | Path = "data/processed/movielens",
    ) -> None:
        if variant not in self._VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}. Choose from {list(self._VARIANTS)}")
        self._variant = variant
        self._archive_name, self._url = self._VARIANTS[variant]
        super().__init__(raw_dir, processed_dir)

    def download(self) -> None:
        archive = self.raw_dir / self._archive_name
        download_file(self._url, archive)
        print(f"Extracting to {self.raw_dir}...")
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(self.raw_dir)

    def process(self) -> None:
        df = self._load_ratings()
        out = self.processed_dir / f"ratings_{self._variant}.parquet"
        df.to_parquet(out, index=False)
        validate_parquet_schema(out, "interactions")
        print(f"Saved to {out}")

    def _load_ratings(self) -> pd.DataFrame:
        if self._variant == "100k":
            path = self.raw_dir / "ml-100k" / "u.data"
            df = pd.read_csv(path, sep="\t",
                             names=["user_id", "item_id", "rating", "timestamp"])
            df["user_id"] -= 1
            df["item_id"] -= 1
            return df

        if self._variant == "1m":
            path = self.raw_dir / "ml-1m" / "ratings.dat"
            df = pd.read_csv(path, sep="::", engine="python",
                             names=["user_id", "item_id", "rating", "timestamp"])
            df["user_id"] -= 1
            df["item_id"] -= 1
            return df

        if self._variant == "10m":
            path = self.raw_dir / "ml-10M100K" / "ratings.dat"
            df = pd.read_csv(path, sep="::", engine="python",
                             names=["user_id", "item_id", "rating", "timestamp"])
            df["user_id"] -= 1
            df["item_id"] -= 1
            return df

        # 20m and 25m: ratings.csv with header userId,movieId,rating,timestamp
        folder = "ml-20m" if self._variant == "20m" else "ml-25m"
        path = self.raw_dir / folder / "ratings.csv"
        df = pd.read_csv(path).rename(columns={"userId": "user_id", "movieId": "item_id"})
        df["user_id"] -= 1
        df["item_id"] -= 1
        return df[["user_id", "item_id", "rating", "timestamp"]]


# Self-register all variants on import
from rl_recsys.data.registry import register  # noqa: E402

for _v in MovieLensPipeline._VARIANTS:
    register(
        f"movielens-{_v}",
        MovieLensPipeline,
        schema="interactions",
        tags=["CF", "movies"],
        variant=_v,
        raw_dir="data/raw/movielens",
        processed_dir="data/processed/movielens",
    )
