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
    item_context_path: Path | None


class OpenBanditPipeline(BasePipeline):
    """Open Bandit Dataset logged feedback pipeline.

    Processes every discovered policy/campaign split into one interactions file.
    The dataset is anonymous, so user_id is set to 0 and click is used as the
    binary rating. Each row keeps its source policy and campaign for filtering,
    plus native user, item, and user-item context columns for contextual bandits.
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
        context_columns = self._collect_context_columns(splits)
        out = self.processed_dir / "interactions.parquet"
        if out.exists():
            out.unlink()

        writer: pq.ParquetWriter | None = None
        total_rows = 0
        try:
            for split in splits:
                item_context = self._load_item_context(split.item_context_path)
                for chunk in pd.read_csv(split.path, chunksize=self.chunksize):
                    processed = self._normalize_chunk(
                        chunk,
                        split,
                        item_context,
                        context_columns,
                    )
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
        item_context: pd.DataFrame | None,
        context_columns: list[str],
    ) -> pd.DataFrame:
        missing = {"timestamp", "item_id", "click", "propensity_score"} - set(
            chunk.columns
        )
        if missing:
            raise ValueError(f"{split.path} missing required columns: {sorted(missing)}")

        df = chunk.drop(columns=_unnamed_columns(chunk), errors="ignore").rename(
            columns={"click": "rating"}
        )
        df = df.rename(columns=_native_column_renames(df.columns))
        if item_context is not None:
            df = df.merge(item_context, on="item_id", how="left", validate="many_to_one")
        df["user_id"] = 0
        df["policy"] = split.policy
        df["campaign"] = split.campaign
        if "position" not in df.columns:
            df["position"] = pd.NA
        for column in context_columns:
            if column not in df.columns:
                df[column] = pd.NA
            if column.startswith("user_item_affinity_"):
                df[column] = pd.to_numeric(df[column], errors="coerce").astype(
                    "float32"
                )
        return df[_OUTPUT_COLUMNS + context_columns]

    def _find_split_csvs(self) -> list[OpenBanditSplit]:
        splits: list[OpenBanditSplit] = []
        for policy in POLICIES:
            for campaign in CAMPAIGNS:
                path = self._find_split_csv(policy, campaign)
                if path is not None:
                    splits.append(
                        OpenBanditSplit(
                            policy=policy,
                            campaign=campaign,
                            path=path,
                            item_context_path=self._find_item_context_csv(
                                policy, campaign
                            ),
                        )
                    )
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

    def _find_item_context_csv(self, policy: str, campaign: str) -> Path | None:
        dataset_root = self.raw_dir / "open_bandit_dataset"
        candidates = [
            dataset_root / policy / campaign / "item_context.csv",
            dataset_root / campaign / policy / "item_context.csv",
        ]
        for path in candidates:
            if path.exists():
                return path

        matches = sorted(self.raw_dir.glob(f"**/{policy}/{campaign}/item_context.csv"))
        return matches[0] if matches else None

    def _load_item_context(self, path: Path | None) -> pd.DataFrame | None:
        if path is None:
            return None
        item_context = pd.read_csv(path)
        item_context = item_context.drop(
            columns=_unnamed_columns(item_context), errors="ignore"
        )
        if "item_id" not in item_context.columns:
            raise ValueError(f"{path} missing required column: item_id")
        item_context = item_context.rename(
            columns=_native_column_renames(item_context.columns)
        )
        return item_context[["item_id"] + _context_columns(item_context.columns)]

    def _collect_context_columns(self, splits: list[OpenBanditSplit]) -> list[str]:
        columns: set[str] = set()
        for split in splits:
            event_columns = pd.read_csv(split.path, nrows=0).columns
            columns.update(_context_columns(_normalized_columns(event_columns)))
            if split.item_context_path is not None:
                item_columns = pd.read_csv(split.item_context_path, nrows=0).columns
                columns.update(_context_columns(_normalized_columns(item_columns)))
        return sorted(columns, key=_column_sort_key)


def _unnamed_columns(df_or_columns: pd.DataFrame | list[str] | pd.Index) -> list[str]:
    columns = (
        df_or_columns.columns
        if isinstance(df_or_columns, pd.DataFrame)
        else df_or_columns
    )
    return [str(col) for col in columns if str(col).startswith("Unnamed:")]


def _native_column_renames(columns: list[str] | pd.Index) -> dict[str, str]:
    return {
        str(column): str(column).replace("user-item_affinity_", "user_item_affinity_")
        for column in columns
        if str(column).startswith("user-item_affinity_")
    }


def _normalized_columns(columns: list[str] | pd.Index) -> pd.Index:
    renames = _native_column_renames(columns)
    return pd.Index([renames.get(str(column), str(column)) for column in columns])


def _context_columns(columns: list[str] | pd.Index) -> list[str]:
    prefixes = ("user_feature_", "item_feature_", "user_item_affinity_")
    return sorted(
        [str(column) for column in columns if str(column).startswith(prefixes)],
        key=_column_sort_key,
    )


def _column_sort_key(column: str) -> tuple[str, int | str]:
    prefix, _, suffix = column.rpartition("_")
    if suffix.isdigit():
        return prefix, int(suffix)
    return prefix, suffix


from rl_recsys.data.registry import register  # noqa: E402

register(
    "open-bandit",
    OpenBanditPipeline,
    schema="interactions",
    tags=["OPE"],
    raw_dir="data/raw/open_bandit",
    processed_dir="data/processed/open_bandit",
)
