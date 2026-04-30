# Dataset Pipeline Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 6 new dataset pipelines — KuaiRec, FINN.no Slate, Open Bandit Dataset, Gowalla, Steam, Amazon Reviews 2018 — each self-registering in the existing registry-driven infrastructure.

**Architecture:** Each pipeline subclasses `BasePipeline`, self-registers at module level via `register()`, emits standardized Parquet validated by `validate_parquet_schema()`, and is wired into `scripts/prepare_data.py` with a single import line. Tests follow the established `test_pipeline_*.py` pattern: instantiation, schema output (with a synthetic fixture), and registry presence.

**Tech Stack:** Python, pandas, numpy (FINN.no), pyarrow, requests, tqdm, ast (Steam), tarfile (FINN.no), zipfile (KuaiRec, OBD), gzip (Gowalla, Steam, Amazon), pytest

---

## Existing Infrastructure (read before starting)

- `rl_recsys/data/pipelines/base.py` — `BasePipeline(raw_dir, processed_dir)` with abstract `download()` and `process()`
- `rl_recsys/data/download.py` — `download_file(url, dest, *, chunk_size, expected_md5)` — shared resumable download utility
- `rl_recsys/data/schema.py` — `validate_parquet_schema(path, schema_type)` — validates required columns exist
- `rl_recsys/data/registry.py` — `register(name, cls, schema, tags, **default_kwargs)` and `get_pipeline(name, ...)`
- `scripts/prepare_data.py` — imports all pipeline modules at top to trigger registration, then dispatches via registry
- `tests/test_pipeline_book_crossing.py` — reference implementation for test pattern

**Required columns per schema type** (from `schema.py`):
- `interactions`: `user_id`, `item_id`, `timestamp`
- `sessions`: `session_id`, `user_id`, `item_id`, `timestamp`
- `slates`: `request_id`, `user_id`

## File Structure

**Create:**
- `rl_recsys/data/pipelines/kuairec.py`
- `rl_recsys/data/pipelines/finn_no_slate.py`
- `rl_recsys/data/pipelines/open_bandit.py`
- `rl_recsys/data/pipelines/gowalla.py`
- `rl_recsys/data/pipelines/steam.py`
- `rl_recsys/data/pipelines/amazon.py`
- `tests/test_pipeline_kuairec.py`
- `tests/test_pipeline_finn_no_slate.py`
- `tests/test_pipeline_open_bandit.py`
- `tests/test_pipeline_gowalla.py`
- `tests/test_pipeline_steam.py`
- `tests/test_pipeline_amazon.py`

**Modify:**
- `rl_recsys/data/download.py` — add `verify: bool = True` parameter (KuaiRec host uses self-signed TLS cert)
- `scripts/prepare_data.py` — add 6 import lines

---

## Task 1: KuaiRec Pipeline

KuaiRec is a fully-observed interaction dataset from Kuaishou. The `big_matrix.csv` has 12M sparse interactions; `watch_ratio` (play / video duration) is the implicit rating. The host at `nas.chongminggao.top` has historically used a self-signed TLS certificate, so `download_file` needs a `verify` parameter.

**Files:**
- Create: `rl_recsys/data/pipelines/kuairec.py`
- Modify: `rl_recsys/data/download.py`
- Modify: `scripts/prepare_data.py`
- Test: `tests/test_pipeline_kuairec.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline_kuairec.py`:

```python
import pandas as pd
import pytest
from rl_recsys.data.pipelines.kuairec import KuaiRecPipeline


def test_instantiation_sets_dirs(tmp_path):
    p = KuaiRecPipeline(raw_dir=str(tmp_path / "raw"),
                        processed_dir=str(tmp_path / "proc"))
    assert p.raw_dir.name == "raw"
    assert p.processed_dir.name == "proc"


def test_process_produces_correct_schema(tmp_path):
    raw_dir = tmp_path / "raw"
    extracted = raw_dir / "KuaiRec 2.0" / "data"
    extracted.mkdir(parents=True)
    (extracted / "big_matrix.csv").write_text(
        "user_id,video_id,play_duration,video_duration,time,date,watch_ratio\n"
        "0,100,30.0,60.0,1609459200,2021-01-01,0.5\n"
        "1,101,45.0,90.0,1609459260,2021-01-01,0.5\n"
    )
    proc_dir = tmp_path / "proc"
    p = KuaiRecPipeline(raw_dir=str(raw_dir), processed_dir=str(proc_dir))
    p.process()

    out = proc_dir / "interactions.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert set(df.columns) >= {"user_id", "item_id", "rating", "timestamp"}
    assert df["user_id"].dtype.kind in ("i", "u")
    assert df["item_id"].dtype.kind in ("i", "u")
    assert df["rating"].ge(0.0).all()


def test_kuairec_is_registered():
    import rl_recsys.data.pipelines.kuairec  # noqa: F401
    from rl_recsys.data.registry import _REGISTRY
    assert "kuairec" in _REGISTRY
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pipeline_kuairec.py -v
```

Expected: `ModuleNotFoundError: No module named 'rl_recsys.data.pipelines.kuairec'`

- [ ] **Step 3: Add `verify` parameter to `download_file`**

Modify `rl_recsys/data/download.py`. Replace the function signature and the `requests.get` call:

```python
def download_file(
    url: str,
    dest: Path,
    *,
    chunk_size: int = 8192,
    expected_md5: str | None = None,
    verify: bool = True,
) -> None:
    if dest.exists():
        if expected_md5 is None or _md5(dest) == expected_md5:
            print(f"Already downloaded: {dest.name}")
            return
        print(f"Checksum mismatch for {dest.name}, re-downloading...")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    try:
        response = requests.get(url, stream=True, timeout=60, verify=verify)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with open(tmp, "wb") as f, tqdm(total=total, unit="iB", unit_scale=True, desc=dest.name) as pbar:
            for chunk in response.iter_content(chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    if expected_md5 is not None and _md5(tmp) != expected_md5:
        tmp.unlink()
        raise ValueError(f"Checksum mismatch for {dest.name} after download")

    tmp.rename(dest)
```

- [ ] **Step 4: Run the existing download tests to confirm no regressions**

```bash
pytest tests/test_download.py -v
```

Expected: All pass (the new `verify` param has a default, so existing call sites are unaffected).

- [ ] **Step 5: Create `rl_recsys/data/pipelines/kuairec.py`**

```python
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from rl_recsys.data.download import download_file
from rl_recsys.data.pipelines.base import BasePipeline
from rl_recsys.data.schema import validate_parquet_schema

_URL = "https://nas.chongminggao.top:4430/datasets/KuaiRec.zip"


class KuaiRecPipeline(BasePipeline):
    """KuaiRec: 12M sparse interactions, 7,176 users, 10,728 items.

    Source: https://github.com/chongminggao/KuaiRec
    Processes big_matrix.csv; watch_ratio (play/video duration) is the implicit rating.
    """

    def __init__(
        self,
        raw_dir: str | Path = "data/raw/kuairec",
        processed_dir: str | Path = "data/processed/kuairec",
    ) -> None:
        super().__init__(raw_dir, processed_dir)

    def download(self) -> None:
        archive = self.raw_dir / "KuaiRec.zip"
        # verify=False: host uses a self-signed TLS certificate
        download_file(_URL, archive, verify=False)
        print(f"Extracting to {self.raw_dir}...")
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(self.raw_dir)

    def process(self) -> None:
        matrix_file = self.raw_dir / "KuaiRec 2.0" / "data" / "big_matrix.csv"
        if not matrix_file.exists():
            raise FileNotFoundError(
                f"Not found: {matrix_file}. Run --download first."
            )
        df = pd.read_csv(matrix_file)
        df = df.rename(columns={"video_id": "item_id", "watch_ratio": "rating", "time": "timestamp"})
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
```

- [ ] **Step 6: Run KuaiRec tests to verify they pass**

```bash
pytest tests/test_pipeline_kuairec.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 7: Add import to `scripts/prepare_data.py`**

In `scripts/prepare_data.py`, add after the existing 4 pipeline imports:

```python
import rl_recsys.data.pipelines.kuairec  # noqa: F401
```

- [ ] **Step 8: Run full test suite**

```bash
pytest -v
```

Expected: All existing tests plus 3 new ones pass.

- [ ] **Step 9: Commit**

```bash
git add rl_recsys/data/download.py rl_recsys/data/pipelines/kuairec.py tests/test_pipeline_kuairec.py scripts/prepare_data.py
git commit -m "feat: add KuaiRec pipeline with watch_ratio as implicit rating"
```

---

## Task 2: FINN.no Slate Pipeline

FINN.no Slate has 37M real slate impressions. The Zenodo archive is a `tar.gz` containing `train.npz` and `test.npz` — pre-encoded NumPy integer arrays (not CSV). Each slate has 25 candidate item IDs; `click` is the index of the chosen item.

**Files:**
- Create: `rl_recsys/data/pipelines/finn_no_slate.py`
- Modify: `scripts/prepare_data.py`
- Test: `tests/test_pipeline_finn_no_slate.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline_finn_no_slate.py`:

```python
import numpy as np
import pandas as pd
import pytest
from rl_recsys.data.pipelines.finn_no_slate import FinnNoSlatePipeline


def _write_fake_npz(path, n=3):
    np.savez(
        path,
        userId=np.arange(n, dtype=np.int64),
        click=np.zeros(n, dtype=np.int64),
        slate=np.arange(n * 25, dtype=np.int64).reshape(n, 25),
        timestamps=np.array([1600000000 + i * 1000 for i in range(n)], dtype=np.int64),
        interaction_type=np.zeros(n, dtype=np.int64),
    )


def test_instantiation_sets_dirs(tmp_path):
    p = FinnNoSlatePipeline(raw_dir=str(tmp_path / "raw"),
                            processed_dir=str(tmp_path / "proc"))
    assert p.raw_dir.name == "raw"
    assert p.processed_dir.name == "proc"


def test_process_produces_correct_schema(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_fake_npz(raw_dir / "train.npz")
    _write_fake_npz(raw_dir / "test.npz")
    proc_dir = tmp_path / "proc"
    p = FinnNoSlatePipeline(raw_dir=str(raw_dir), processed_dir=str(proc_dir))
    p.process()

    out = proc_dir / "slates.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert set(df.columns) >= {"request_id", "user_id", "slate", "clicks", "timestamp"}
    assert len(df) == 6  # 3 train + 3 test
    assert df["user_id"].dtype.kind in ("i", "u")
    assert isinstance(df["slate"].iloc[0], list)
    assert len(df["slate"].iloc[0]) == 25


def test_finn_no_slate_is_registered():
    import rl_recsys.data.pipelines.finn_no_slate  # noqa: F401
    from rl_recsys.data.registry import _REGISTRY
    assert "finn-no-slate" in _REGISTRY
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pipeline_finn_no_slate.py -v
```

Expected: `ModuleNotFoundError: No module named 'rl_recsys.data.pipelines.finn_no_slate'`

- [ ] **Step 3: Create `rl_recsys/data/pipelines/finn_no_slate.py`**

```python
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
    """FINN.no Slate: 37M slate impressions, 2.3M users, ~1.3M items.

    Source: https://github.com/finn-no/recsys-slates-dataset (Zenodo 4884099)
    Format: tar.gz containing train.npz + test.npz (pre-encoded integer arrays).
    Each slate has 25 candidate item IDs; click is the index of the chosen item.
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

    def _find_npz(self, name: str) -> Path:
        candidates = list(self.raw_dir.glob(f"**/{name}"))
        if not candidates:
            raise FileNotFoundError(
                f"Not found: {name} under {self.raw_dir}. Run --download first."
            )
        return candidates[0]

    def process(self) -> None:
        parts = []
        offset = 0
        for split in ("train.npz", "test.npz"):
            path = self._find_npz(split)
            data = np.load(path)
            n = len(data["userId"])
            parts.append(pd.DataFrame({
                "request_id": np.arange(offset, offset + n, dtype=np.int64),
                "user_id": data["userId"].astype(np.int64),
                "slate": [list(map(int, row)) for row in data["slate"]],
                "clicks": data["click"].astype(np.int64),
                "timestamp": data["timestamps"].astype(np.int64),
            }))
            offset += n

        df = pd.concat(parts, ignore_index=True)
        out = self.processed_dir / "slates.parquet"
        df.to_parquet(out, index=False)
        validate_parquet_schema(out, "slates")
        print(f"Saved {len(df):,} rows to {out}")


from rl_recsys.data.registry import register  # noqa: E402

register(
    "finn-no-slate",
    FinnNoSlatePipeline,
    schema="slates",
    tags=["RL/Slate"],
    raw_dir="data/raw/finn_no_slate",
    processed_dir="data/processed/finn_no_slate",
)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pipeline_finn_no_slate.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Add import to `scripts/prepare_data.py`**

```python
import rl_recsys.data.pipelines.finn_no_slate  # noqa: F401
```

- [ ] **Step 6: Run full test suite**

```bash
pytest -v
```

Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add rl_recsys/data/pipelines/finn_no_slate.py tests/test_pipeline_finn_no_slate.py scripts/prepare_data.py
git commit -m "feat: add FINN.no Slate pipeline for slate-level RL training"
```

---

## Task 3: Open Bandit Dataset Pipeline

The Open Bandit Dataset (OBD) contains 26M logged bandit feedback events across 3 campaigns and 2 policies. There is no persistent user ID — it's anonymous bandit data — so `user_id` is set to 0 and `click` (0/1) becomes the rating. The `propensity_score` column is preserved for off-policy evaluation.

**Files:**
- Create: `rl_recsys/data/pipelines/open_bandit.py`
- Modify: `scripts/prepare_data.py`
- Test: `tests/test_pipeline_open_bandit.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline_open_bandit.py`:

```python
import pandas as pd
import pytest
from rl_recsys.data.pipelines.open_bandit import OpenBanditPipeline


def test_instantiation_sets_dirs(tmp_path):
    p = OpenBanditPipeline(raw_dir=str(tmp_path / "raw"),
                           processed_dir=str(tmp_path / "proc"))
    assert p.raw_dir.name == "raw"
    assert p.processed_dir.name == "proc"


def test_process_produces_correct_schema(tmp_path):
    raw_dir = tmp_path / "raw"
    campaign_dir = raw_dir / "open_bandit_dataset" / "all" / "random"
    campaign_dir.mkdir(parents=True)
    (campaign_dir / "all.csv").write_text(
        "timestamp,item_id,position,click,propensity_score,user_feature_0\n"
        "1609459200,42,0,1,0.33,0.5\n"
        "1609459260,43,1,0,0.33,0.7\n"
    )
    proc_dir = tmp_path / "proc"
    p = OpenBanditPipeline(raw_dir=str(raw_dir), processed_dir=str(proc_dir))
    p.process()

    out = proc_dir / "interactions.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert set(df.columns) >= {"user_id", "item_id", "rating", "timestamp"}
    assert (df["rating"].isin([0, 1])).all()
    assert (df["user_id"] == 0).all()
    assert "propensity_score" in df.columns


def test_open_bandit_is_registered():
    import rl_recsys.data.pipelines.open_bandit  # noqa: F401
    from rl_recsys.data.registry import _REGISTRY
    assert "open-bandit" in _REGISTRY
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pipeline_open_bandit.py -v
```

Expected: `ModuleNotFoundError: No module named 'rl_recsys.data.pipelines.open_bandit'`

- [ ] **Step 3: Create `rl_recsys/data/pipelines/open_bandit.py`**

```python
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from rl_recsys.data.download import download_file
from rl_recsys.data.pipelines.base import BasePipeline
from rl_recsys.data.schema import validate_parquet_schema

_URL = "https://research.zozo.com/data_release/open_bandit_dataset.zip"


class OpenBanditPipeline(BasePipeline):
    """Open Bandit Dataset: 26M logged bandit feedback, 3 campaigns, 2 policies.

    Source: https://github.com/st-tech/zr-obp
    Processes campaign='all', policy='random'.
    No persistent user_id exists (anonymous bandit context); user_id is set to 0.
    propensity_score is preserved for off-policy evaluation.
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
        csv_file = (
            self.raw_dir / "open_bandit_dataset" / "all" / "random" / "all.csv"
        )
        if not csv_file.exists():
            raise FileNotFoundError(
                f"Not found: {csv_file}. Run --download first."
            )
        df = pd.read_csv(csv_file)
        df["user_id"] = 0
        df = df.rename(columns={"click": "rating"})
        keep = ["user_id", "item_id", "rating", "timestamp", "propensity_score"]
        out = self.processed_dir / "interactions.parquet"
        df[[c for c in keep if c in df.columns]].to_parquet(out, index=False)
        validate_parquet_schema(out, "interactions")
        print(f"Saved {len(df):,} rows to {out}")


from rl_recsys.data.registry import register  # noqa: E402

register(
    "open-bandit",
    OpenBanditPipeline,
    schema="interactions",
    tags=["OPE"],
    raw_dir="data/raw/open_bandit",
    processed_dir="data/processed/open_bandit",
)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pipeline_open_bandit.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Add import to `scripts/prepare_data.py`**

```python
import rl_recsys.data.pipelines.open_bandit  # noqa: F401
```

- [ ] **Step 6: Run full test suite**

```bash
pytest -v
```

Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add rl_recsys/data/pipelines/open_bandit.py tests/test_pipeline_open_bandit.py scripts/prepare_data.py
git commit -m "feat: add Open Bandit Dataset pipeline for off-policy evaluation"
```

---

## Task 4: Gowalla Pipeline

Gowalla is a location check-in dataset from SNAP Stanford. The raw file is a single gzip-compressed TSV with no header; pandas reads it directly. Timestamps are ISO 8601 UTC strings converted to Unix seconds. Location IDs are factorized to sequential integers. `session_id` is the row index (no session-grouping semantics).

**Files:**
- Create: `rl_recsys/data/pipelines/gowalla.py`
- Modify: `scripts/prepare_data.py`
- Test: `tests/test_pipeline_gowalla.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline_gowalla.py`:

```python
import gzip
import pandas as pd
import pytest
from rl_recsys.data.pipelines.gowalla import GowallaPipeline


def test_instantiation_sets_dirs(tmp_path):
    p = GowallaPipeline(raw_dir=str(tmp_path / "raw"),
                        processed_dir=str(tmp_path / "proc"))
    assert p.raw_dir.name == "raw"
    assert p.processed_dir.name == "proc"


def test_process_produces_correct_schema(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    content = (
        "196514\t2010-10-19T23:55:27Z\t30.2359091\t-97.7951395\t145064\n"
        "196514\t2010-10-18T22:17:43Z\t30.2691029\t-97.7493953\t1275991\n"
        "196515\t2010-10-19T20:20:17Z\t30.2749767\t-97.7403954\t145064\n"
    )
    gz_path = raw_dir / "loc-gowalla_totalCheckins.txt.gz"
    gz_path.write_bytes(gzip.compress(content.encode()))
    proc_dir = tmp_path / "proc"
    p = GowallaPipeline(raw_dir=str(raw_dir), processed_dir=str(proc_dir))
    p.process()

    out = proc_dir / "sessions.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert set(df.columns) >= {"session_id", "user_id", "item_id", "timestamp"}
    assert len(df) == 3
    assert df["user_id"].dtype.kind in ("i", "u")
    assert df["item_id"].dtype.kind in ("i", "u")
    assert df["timestamp"].dtype.kind in ("i", "u")
    # location_id 145064 appears twice, should get same item_id
    assert df[df["user_id"] == 196514]["item_id"].nunique() == 2


def test_gowalla_is_registered():
    import rl_recsys.data.pipelines.gowalla  # noqa: F401
    from rl_recsys.data.registry import _REGISTRY
    assert "gowalla" in _REGISTRY
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pipeline_gowalla.py -v
```

Expected: `ModuleNotFoundError: No module named 'rl_recsys.data.pipelines.gowalla'`

- [ ] **Step 3: Create `rl_recsys/data/pipelines/gowalla.py`**

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

from rl_recsys.data.download import download_file
from rl_recsys.data.pipelines.base import BasePipeline
from rl_recsys.data.schema import validate_parquet_schema

_URL = "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"


class GowallaPipeline(BasePipeline):
    """Gowalla check-in dataset: 6.4M check-ins, 196K users.

    Source: https://snap.stanford.edu/data/loc-gowalla.html
    session_id = row index (no session-grouping; each check-in is independent).
    Timestamps converted from ISO 8601 UTC to Unix seconds.
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
            raise FileNotFoundError(
                f"Not found: {gz_file}. Run --download first."
            )
        df = pd.read_csv(
            gz_file,
            sep="\t",
            header=None,
            names=["user_id", "checkin_time", "latitude", "longitude", "location_id"],
        )
        df["item_id"] = pd.factorize(df["location_id"])[0]
        df["timestamp"] = (
            pd.to_datetime(df["checkin_time"]).astype("int64") // 10**9
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pipeline_gowalla.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Add import to `scripts/prepare_data.py`**

```python
import rl_recsys.data.pipelines.gowalla  # noqa: F401
```

- [ ] **Step 6: Run full test suite**

```bash
pytest -v
```

Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add rl_recsys/data/pipelines/gowalla.py tests/test_pipeline_gowalla.py scripts/prepare_data.py
git commit -m "feat: add Gowalla check-in pipeline (SNAP Stanford, 6.4M events)"
```

---

## Task 5: Steam Pipeline

Steam reviews use Python-eval syntax (single quotes, bare `True`/`False`) — **not valid JSON**. Each line must be parsed with `ast.literal_eval()`. Hours played is the implicit rating.

**Files:**
- Create: `rl_recsys/data/pipelines/steam.py`
- Modify: `scripts/prepare_data.py`
- Test: `tests/test_pipeline_steam.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline_steam.py`:

```python
import gzip
import pandas as pd
import pytest
from rl_recsys.data.pipelines.steam import SteamPipeline


def test_instantiation_sets_dirs(tmp_path):
    p = SteamPipeline(raw_dir=str(tmp_path / "raw"),
                      processed_dir=str(tmp_path / "proc"))
    assert p.raw_dir.name == "raw"
    assert p.processed_dir.name == "proc"


def test_process_produces_correct_schema(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    # Steam files use Python eval syntax (single quotes, True/False) — not JSON
    lines = [
        "{'username': 'user1', 'user_id': 'u1', 'product_id': 'g1', 'hours': 10.5, 'date': 'Oct 1, 2011'}",
        "{'username': 'user2', 'user_id': 'u2', 'product_id': 'g2', 'hours': 0.0, 'date': 'Jan 5, 2013'}",
        "bad line that should be skipped",
    ]
    content = "\n".join(lines).encode()
    (raw_dir / "steam_reviews.json.gz").write_bytes(gzip.compress(content))
    proc_dir = tmp_path / "proc"
    p = SteamPipeline(raw_dir=str(raw_dir), processed_dir=str(proc_dir))
    p.process()

    out = proc_dir / "interactions.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert set(df.columns) >= {"user_id", "item_id", "rating", "timestamp"}
    assert len(df) == 2  # bad line skipped
    assert df["user_id"].dtype.kind in ("i", "u")
    assert df["item_id"].dtype.kind in ("i", "u")
    assert df["rating"].dtype.kind == "f"
    assert df["rating"].iloc[0] == pytest.approx(10.5)


def test_steam_is_registered():
    import rl_recsys.data.pipelines.steam  # noqa: F401
    from rl_recsys.data.registry import _REGISTRY
    assert "steam" in _REGISTRY
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pipeline_steam.py -v
```

Expected: `ModuleNotFoundError: No module named 'rl_recsys.data.pipelines.steam'`

- [ ] **Step 3: Create `rl_recsys/data/pipelines/steam.py`**

```python
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
    """Steam reviews: 7.8M reviews, 2.5M users.

    Source: https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data
    Files use Python-eval syntax (single quotes, True/False) — not JSON.
    hours (playtime) is the implicit rating.
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
            raise FileNotFoundError(
                f"Not found: {gz_file}. Run --download first."
            )
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
            ts = pd.to_datetime(df["date"], format="%b %d, %Y", errors="coerce")
            df["timestamp"] = (ts.astype("int64") // 10**9).where(ts.notna(), 0)
        else:
            df["timestamp"] = 0
        df["timestamp"] = df["timestamp"].astype("int64")

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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pipeline_steam.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Add import to `scripts/prepare_data.py`**

```python
import rl_recsys.data.pipelines.steam  # noqa: F401
```

- [ ] **Step 6: Run full test suite**

```bash
pytest -v
```

Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add rl_recsys/data/pipelines/steam.py tests/test_pipeline_steam.py scripts/prepare_data.py
git commit -m "feat: add Steam reviews pipeline (7.8M reviews, implicit rating = hours)"
```

---

## Task 6: Amazon Reviews 2018 Pipeline (Parameterized)

Amazon Reviews 2018 has 29 categories, each a separate download. One `AmazonPipeline` class parameterized by category slug handles all of them. Four categories are pre-registered at module import; others can be instantiated directly with `AmazonPipeline(category="...")`. Unlike Steam, these files are proper JSON Lines.

**Files:**
- Create: `rl_recsys/data/pipelines/amazon.py`
- Modify: `scripts/prepare_data.py`
- Test: `tests/test_pipeline_amazon.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline_amazon.py`:

```python
import gzip
import json
import pandas as pd
import pytest
from rl_recsys.data.pipelines.amazon import AmazonPipeline


def _write_fake_reviews(path, n=3):
    lines = [
        json.dumps({
            "overall": float(i % 5 + 1),
            "reviewerID": f"R{i}",
            "asin": f"B00{i:03d}",
            "unixReviewTime": 1600000000 + i * 1000,
        })
        for i in range(n)
    ]
    path.write_bytes(gzip.compress("\n".join(lines).encode()))


def test_instantiation_default_category(tmp_path):
    p = AmazonPipeline(raw_dir=str(tmp_path / "raw"),
                       processed_dir=str(tmp_path / "proc"))
    assert p.category == "Books"


def test_instantiation_custom_category(tmp_path):
    p = AmazonPipeline(category="Electronics",
                       raw_dir=str(tmp_path / "raw"),
                       processed_dir=str(tmp_path / "proc"))
    assert p.category == "Electronics"


def test_process_produces_correct_schema(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_fake_reviews(raw_dir / "Books_5.json.gz")
    proc_dir = tmp_path / "proc"
    p = AmazonPipeline(category="Books",
                       raw_dir=str(raw_dir), processed_dir=str(proc_dir))
    p.process()

    out = proc_dir / "interactions.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert set(df.columns) >= {"user_id", "item_id", "rating", "timestamp"}
    assert df["user_id"].dtype.kind in ("i", "u")
    assert df["item_id"].dtype.kind in ("i", "u")
    assert df["rating"].between(1.0, 5.0).all()
    assert (df["timestamp"] > 0).all()


def test_amazon_categories_registered():
    import rl_recsys.data.pipelines.amazon  # noqa: F401
    from rl_recsys.data.registry import _REGISTRY
    for key in ("amazon-books", "amazon-movies", "amazon-electronics", "amazon-video-games"):
        assert key in _REGISTRY
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pipeline_amazon.py -v
```

Expected: `ModuleNotFoundError: No module named 'rl_recsys.data.pipelines.amazon'`

- [ ] **Step 3: Create `rl_recsys/data/pipelines/amazon.py`**

```python
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from rl_recsys.data.download import download_file
from rl_recsys.data.pipelines.base import BasePipeline
from rl_recsys.data.schema import validate_parquet_schema

_BASE_URL = (
    "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/{category}_5.json.gz"
)

_REGISTERED_CATEGORIES: dict[str, str] = {
    "amazon-books": "Books",
    "amazon-movies": "Movies_and_TV",
    "amazon-electronics": "Electronics",
    "amazon-video-games": "Video_Games",
}


class AmazonPipeline(BasePipeline):
    """Amazon Reviews 2018 (5-core), parameterized by category.

    Source: https://nijianmo.github.io/amazon/index.html
    Each line is proper JSON; unixReviewTime is used as timestamp.
    overall (1-5) is the explicit rating.
    """

    def __init__(
        self,
        category: str = "Books",
        raw_dir: str | Path = "data/raw/amazon",
        processed_dir: str | Path = "data/processed/amazon",
    ) -> None:
        self.category = category
        super().__init__(raw_dir, processed_dir)

    def download(self) -> None:
        url = _BASE_URL.format(category=self.category)
        dest = self.raw_dir / f"{self.category}_5.json.gz"
        download_file(url, dest)

    def process(self) -> None:
        gz_file = self.raw_dir / f"{self.category}_5.json.gz"
        if not gz_file.exists():
            raise FileNotFoundError(
                f"Not found: {gz_file}. Run --download first."
            )
        records = []
        with gzip.open(gz_file, "rb") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        df = pd.DataFrame(records)
        df = df.dropna(subset=["reviewerID", "asin", "overall"])
        df["user_id"] = pd.factorize(df["reviewerID"])[0]
        df["item_id"] = pd.factorize(df["asin"])[0]
        df["rating"] = df["overall"].astype(float)
        if "unixReviewTime" in df.columns:
            df["timestamp"] = pd.to_numeric(
                df["unixReviewTime"], errors="coerce"
            ).fillna(0).astype("int64")
        else:
            df["timestamp"] = 0

        out = self.processed_dir / "interactions.parquet"
        df[["user_id", "item_id", "rating", "timestamp"]].to_parquet(out, index=False)
        validate_parquet_schema(out, "interactions")
        print(f"Saved {len(df):,} rows to {out} (category={self.category})")


from rl_recsys.data.registry import register  # noqa: E402

for _key, _cat in _REGISTERED_CATEGORIES.items():
    register(
        _key,
        AmazonPipeline,
        schema="interactions",
        tags=["CF"],
        category=_cat,
        raw_dir=f"data/raw/amazon/{_cat}",
        processed_dir=f"data/processed/amazon/{_cat}",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pipeline_amazon.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Add import to `scripts/prepare_data.py`**

```python
import rl_recsys.data.pipelines.amazon  # noqa: F401
```

- [ ] **Step 6: Run full test suite**

```bash
pytest -v
```

Expected: All tests pass (30+ tests total across all pipeline test files).

- [ ] **Step 7: Commit**

```bash
git add rl_recsys/data/pipelines/amazon.py tests/test_pipeline_amazon.py scripts/prepare_data.py
git commit -m "feat: add Amazon Reviews 2018 pipeline (Books, Movies, Electronics, Video Games)"
```

---

## Post-completion verification

After all 6 tasks are done, verify the registry lists all expected datasets:

```bash
python -c "
import rl_recsys.data.pipelines.movielens
import rl_recsys.data.pipelines.lastfm
import rl_recsys.data.pipelines.rl4rs
import rl_recsys.data.pipelines.book_crossing
import rl_recsys.data.pipelines.kuairec
import rl_recsys.data.pipelines.finn_no_slate
import rl_recsys.data.pipelines.open_bandit
import rl_recsys.data.pipelines.gowalla
import rl_recsys.data.pipelines.steam
import rl_recsys.data.pipelines.amazon
from rl_recsys.data.registry import list_datasets
print(list_datasets())
"
```

Expected output (17 datasets):
```
['amazon-books', 'amazon-electronics', 'amazon-movies', 'amazon-video-games',
 'book-crossing', 'finn-no-slate', 'gowalla', 'kuairec', 'lastfm-1k',
 'movielens-100k', 'movielens-10m', 'movielens-1m', 'movielens-20m', 'movielens-25m',
 'open-bandit', 'rl4rs', 'steam']
```
