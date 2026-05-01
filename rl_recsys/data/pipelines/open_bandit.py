from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from rl_recsys.data.download import download_file
from rl_recsys.data.pipelines.base import BasePipeline
from rl_recsys.data.schema import validate_parquet_schema

_URL = "https://research.zozo.com/data_release/open_bandit_dataset.zip"


class OpenBanditPipeline(BasePipeline):
    """Open Bandit Dataset logged feedback pipeline.

    Processes campaign=all, policy=random. The dataset is anonymous, so user_id
    is set to 0 and click is used as the binary rating.
    """

    def __init__(
        self,
        raw_dir: str | Path = "data/raw/open_bandit",
        processed_dir: str | Path = "data/processed/open_bandit",
    ) -> None:
        super().__init__(raw_dir, processed_dir)

    def download(self) -> None:
        archive = self.raw_dir / "open_bandit_dataset.zip"
        download_file(_URL, archive)
        print(f"Extracting to {self.raw_dir}...")
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(self.raw_dir)

    def process(self) -> None:
        csv_file = self._find_random_all_csv()

        df = pd.read_csv(csv_file)
        df["user_id"] = 0
        df = df.rename(columns={"click": "rating"})
        keep = ["user_id", "item_id", "rating", "timestamp", "propensity_score"]

        out = self.processed_dir / "interactions.parquet"
        df[[c for c in keep if c in df.columns]].to_parquet(out, index=False)
        validate_parquet_schema(out, "interactions")
        print(f"Saved {len(df):,} rows to {out}")

    def _find_random_all_csv(self) -> Path:
        candidates = [
            self.raw_dir / "open_bandit_dataset" / "random" / "all" / "all.csv",
            self.raw_dir / "open_bandit_dataset" / "all" / "random" / "all.csv",
        ]
        for path in candidates:
            if path.exists():
                return path

        matches = sorted(self.raw_dir.glob("**/random/all/all.csv"))
        if matches:
            return matches[0]
        raise FileNotFoundError(
            f"Open Bandit random/all/all.csv not found under {self.raw_dir}. "
            "Run --download first."
        )


from rl_recsys.data.registry import register  # noqa: E402

register(
    "open-bandit",
    OpenBanditPipeline,
    schema="interactions",
    tags=["OPE"],
    raw_dir="data/raw/open_bandit",
    processed_dir="data/processed/open_bandit",
)
