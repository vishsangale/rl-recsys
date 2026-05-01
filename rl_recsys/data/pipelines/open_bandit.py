from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from rl_recsys.data.download import download_file
from rl_recsys.data.pipelines.base import BasePipeline
from rl_recsys.data.schema import validate_parquet_schema

_URL = "https://research.zozo.com/data_release/open_bandit_dataset.zip"
POLICIES = ("random", "bts")
CAMPAIGNS = ("all", "men", "women")
_OUTPUT_COLUMNS = [
    "user_id",
    "item_id",
    "rating",
    "timestamp",
    "propensity_score",
    "position",
    "policy",
    "campaign",
]


@dataclass(frozen=True)
class OpenBanditSplit:
    policy: str
    campaign: str
    path: Path


class OpenBanditPipeline(BasePipeline):
    """Open Bandit Dataset logged feedback pipeline.

    Processes every discovered policy/campaign split into one interactions file.
    The dataset is anonymous, so user_id is set to 0 and click is used as the
    binary rating. Each row keeps its source policy and campaign for filtering.
    """

    def __init__(
        self,
        raw_dir: str | Path = "data/raw/open_bandit",
        processed_dir: str | Path = "data/processed/open_bandit",
        chunksize: int = 250_000,
    ) -> None:
        super().__init__(raw_dir, processed_dir)
        if chunksize < 1:
            raise ValueError("chunksize must be positive")
        self.chunksize = chunksize

    def download(self) -> None:
        archive = self.raw_dir / "open_bandit_dataset.zip"
        download_file(_URL, archive)
        print(f"Extracting to {self.raw_dir}...")
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(self.raw_dir)

    def process(self) -> None:
        splits = self._find_split_csvs()
        out = self.processed_dir / "interactions.parquet"
        if out.exists():
            out.unlink()

        writer: pq.ParquetWriter | None = None
        total_rows = 0
        try:
            for split in splits:
                for chunk in pd.read_csv(split.path, chunksize=self.chunksize):
                    processed = self._normalize_chunk(chunk, split)
                    table = pa.Table.from_pandas(processed, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            out,
                            table.schema,
                            compression="snappy",
                        )
                    writer.write_table(table)
                    total_rows += len(processed)
                print(
                    "Processed "
                    f"{split.policy}/{split.campaign} from {split.path.name}"
                )
        finally:
            if writer is not None:
                writer.close()

        if writer is None:
            raise FileNotFoundError(
                f"No Open Bandit event CSVs found under {self.raw_dir}. "
                "Run --download first."
            )

        validate_parquet_schema(out, "interactions")
        print(f"Saved {total_rows:,} rows from {len(splits)} splits to {out}")

    def _normalize_chunk(
        self,
        chunk: pd.DataFrame,
        split: OpenBanditSplit,
    ) -> pd.DataFrame:
        missing = {"timestamp", "item_id", "click", "propensity_score"} - set(
            chunk.columns
        )
        if missing:
            raise ValueError(f"{split.path} missing required columns: {sorted(missing)}")

        df = chunk.rename(columns={"click": "rating"})
        df["user_id"] = 0
        df["policy"] = split.policy
        df["campaign"] = split.campaign
        if "position" not in df.columns:
            df["position"] = pd.NA
        return df[_OUTPUT_COLUMNS]

    def _find_split_csvs(self) -> list[OpenBanditSplit]:
        splits: list[OpenBanditSplit] = []
        for policy in POLICIES:
            for campaign in CAMPAIGNS:
                path = self._find_split_csv(policy, campaign)
                if path is not None:
                    splits.append(OpenBanditSplit(policy, campaign, path))
        if splits:
            return splits
        raise FileNotFoundError(
            f"No Open Bandit event CSVs found under {self.raw_dir}. "
            "Run --download first."
        )

    def _find_split_csv(self, policy: str, campaign: str) -> Path | None:
        dataset_root = self.raw_dir / "open_bandit_dataset"
        candidates = [
            dataset_root / policy / campaign / f"{campaign}.csv",
            dataset_root / campaign / policy / f"{campaign}.csv",
        ]
        for path in candidates:
            if path.exists():
                return path

        matches = sorted(self.raw_dir.glob(f"**/{policy}/{campaign}/{campaign}.csv"))
        return matches[0] if matches else None


from rl_recsys.data.registry import register  # noqa: E402

register(
    "open-bandit",
    OpenBanditPipeline,
    schema="interactions",
    tags=["OPE"],
    raw_dir="data/raw/open_bandit",
    processed_dir="data/processed/open_bandit",
)
