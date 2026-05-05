from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from rl_recsys.data.pipelines.base import BasePipeline
from rl_recsys.data.schema import validate_parquet_schema

_BASE_URL = (
    "https://media.githubusercontent.com/media/"
    "finn-no/recsys_slates_dataset/main/data"
)
_FILES = ["data.npz", "ind2val.json", "itemattr.npz"]


class FinnNoSlatePipeline(BasePipeline):
    """FINN.no slate impressions pipeline.

    Downloads from the finn-no/recsys_slates_dataset GitHub LFS mirror.
    data.npz contains arrays shaped [N_users, 20, ...]:
      slate      — [N_users, 20, slate_size] item IDs per impression
      click      — [N_users, 20] clicked item ID (0=PAD, 1=noClick, >1=real click)
      click_idx  — [N_users, 20] position of the click within the slate
      userId     — [N_users]
    """

    def __init__(
        self,
        raw_dir: str | Path = "data/raw/finn_no_slate",
        processed_dir: str | Path = "data/processed/finn-no-slate",
    ) -> None:
        super().__init__(raw_dir, processed_dir)

    def download(self) -> None:
        for fname in _FILES:
            dest = self.raw_dir / fname
            if dest.exists():
                print(f"{fname} already exists, skipping.")
                continue
            url = f"{_BASE_URL}/{fname}"
            print(f"Downloading {fname} from {url} ...")
            r = requests.get(url, stream=True)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total, unit="iB", unit_scale=True, desc=fname
            ) as bar:
                for chunk in r.iter_content(64 * 1024):
                    f.write(chunk)
                    bar.update(len(chunk))
        print("Download complete.")

    def process(self, chunk_size: int = 50_000) -> None:
        path = self.raw_dir / "data.npz"
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}. Run --download first.")

        print(f"Loading {path} (memory-mapped) ...")
        # mmap_mode='r' avoids loading the full 9 GB into RAM at once.
        d = np.load(path, mmap_mode="r")
        user_ids = d["userId"]       # [N_users]
        slates = d["slate"]          # [N_users, 20, slate_size]
        clicks = d["click"]          # [N_users, 20] — item ID; 0=PAD, 1=noClick
        click_idxs = d["click_idx"]  # [N_users, 20] — position within slate

        N_users, T, S = slates.shape
        print(f"  {N_users:,} users × {T} interactions × {S}-item slates")

        out = self.processed_dir / "slates.parquet"
        parts: list[Path] = []
        total_rows = 0
        req_id = 0

        for start in tqdm(range(0, N_users, chunk_size), desc="processing chunks"):
            end = min(start + chunk_size, N_users)
            uu = np.repeat(user_ids[start:end], T)
            cc = clicks[start:end].ravel()
            ci = click_idxs[start:end].ravel().astype(np.int64)
            sl = np.array(slates[start:end]).reshape(-1, S)  # materialise chunk

            # Keep only actual clicks (0=PAD, 1=noClick, >1=real item clicked).
            mask = cc > 1
            if not mask.any():
                continue

            uu_f = uu[mask].astype(np.int64)
            ci_f = ci[mask]
            sl_f = sl[mask]
            n = int(mask.sum())

            chunk_df = pd.DataFrame(
                {
                    "request_id": np.arange(req_id, req_id + n, dtype=np.int64),
                    "user_id": uu_f,
                    "clicks": ci_f,
                    "slate": [row.tolist() for row in sl_f],
                }
            )
            part_path = self.processed_dir / f"_part_{start}.parquet"
            chunk_df.to_parquet(part_path, index=False)
            parts.append(part_path)
            total_rows += n
            req_id += n

        d.close()

        print(f"Merging {len(parts)} chunks → {out} ...")
        pd.concat(
            [pd.read_parquet(p) for p in parts], ignore_index=True
        ).to_parquet(out, index=False)
        for p in parts:
            p.unlink()

        validate_parquet_schema(out, "slates")
        print(f"Saved {total_rows:,} rows to {out}")


from rl_recsys.data.registry import register  # noqa: E402

register(
    "finn-no-slate",
    FinnNoSlatePipeline,
    schema="slates",
    tags=["RL/Slate"],
    raw_dir="data/raw/finn_no_slate",
    processed_dir="data/processed/finn_no_slate",
)
