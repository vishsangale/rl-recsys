from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from rl_recsys.data.download import download_file
from rl_recsys.data.pipelines.base import BasePipeline
from rl_recsys.data.schema import validate_parquet_schema

_URL = "https://zenodo.org/record/4884099/files/data.tar.gz"


class FinnNoSlatePipeline(BasePipeline):
    """FINN.no slate impressions pipeline.

    The Zenodo archive contains train.npz and test.npz with pre-encoded integer
    arrays. Each slate has 25 candidate item IDs; click is the chosen index.
    """

    def __init__(
        self,
        raw_dir: str | Path = "data/raw/finn_no_slate",
        processed_dir: str | Path = "data/processed/finn_no_slate",
    ) -> None:
        super().__init__(raw_dir, processed_dir)

    def download(self) -> None:
        archive = self.raw_dir / "data.tar.gz"
        download_file(_URL, archive)
        print(f"Extracting to {self.raw_dir}...")
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(self.raw_dir)

    def process(self) -> None:
        parts: list[pd.DataFrame] = []
        offset = 0
        for split in ("train.npz", "test.npz"):
            path = self._find_npz(split)
            with np.load(path) as data:
                n = len(data["userId"])
                parts.append(
                    pd.DataFrame(
                        {
                            "request_id": np.arange(offset, offset + n, dtype=np.int64),
                            "user_id": data["userId"].astype(np.int64),
                            "slate": [list(map(int, row)) for row in data["slate"]],
                            "clicks": data["click"].astype(np.int64),
                            "timestamp": data["timestamps"].astype(np.int64),
                        }
                    )
                )
            offset += n

        df = pd.concat(parts, ignore_index=True)
        out = self.processed_dir / "slates.parquet"
        df.to_parquet(out, index=False)
        validate_parquet_schema(out, "slates")
        print(f"Saved {len(df):,} rows to {out}")

    def _find_npz(self, name: str) -> Path:
        candidates = list(self.raw_dir.glob(f"**/{name}"))
        if not candidates:
            raise FileNotFoundError(
                f"Not found: {name} under {self.raw_dir}. Run --download first."
            )
        return candidates[0]


from rl_recsys.data.registry import register  # noqa: E402

register(
    "finn-no-slate",
    FinnNoSlatePipeline,
    schema="slates",
    tags=["RL/Slate"],
    raw_dir="data/raw/finn_no_slate",
    processed_dir="data/processed/finn_no_slate",
)
