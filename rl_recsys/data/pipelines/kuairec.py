from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from rl_recsys.data.download import download_file
from rl_recsys.data.pipelines.base import BasePipeline
from rl_recsys.data.schema import validate_parquet_schema

_URL = "https://nas.chongminggao.top:4430/datasets/KuaiRec.zip"


class KuaiRecPipeline(BasePipeline):
    """KuaiRec sparse interaction pipeline.

    Processes big_matrix.csv; watch_ratio is used as the implicit rating.
    Source: https://github.com/chongminggao/KuaiRec
    """

    def __init__(
        self,
        raw_dir: str | Path = "data/raw/kuairec",
        processed_dir: str | Path = "data/processed/kuairec",
    ) -> None:
        super().__init__(raw_dir, processed_dir)

    def download(self) -> None:
        archive = self.raw_dir / "KuaiRec.zip"
        download_file(_URL, archive, verify=False)
        print(f"Extracting to {self.raw_dir}...")
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(self.raw_dir)

    def process(self) -> None:
        matrix_file = self.raw_dir / "KuaiRec 2.0" / "data" / "big_matrix.csv"
        if not matrix_file.exists():
            raise FileNotFoundError(f"Not found: {matrix_file}. Run --download first.")

        df = pd.read_csv(matrix_file)
        df = df.rename(
            columns={
                "video_id": "item_id",
                "watch_ratio": "rating",
                "time": "timestamp",
            }
        )

        out = self.processed_dir / "interactions.parquet"
        df[["user_id", "item_id", "rating", "timestamp"]].to_parquet(out, index=False)
        validate_parquet_schema(out, "interactions")
        print(f"Saved {len(df):,} rows to {out}")


from rl_recsys.data.registry import register  # noqa: E402

register(
    "kuairec",
    KuaiRecPipeline,
    schema="interactions",
    tags=["RL/Session"],
    raw_dir="data/raw/kuairec",
    processed_dir="data/processed/kuairec",
)
