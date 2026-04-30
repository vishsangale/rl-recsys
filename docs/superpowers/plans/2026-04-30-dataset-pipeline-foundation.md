# Dataset Pipeline Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the dataset pipeline infrastructure to be registry-driven with robust downloads and schema validation, parameterize MovieLens for all five variants, and add Book-Crossing as the first new standalone pipeline.

**Architecture:** A central registry (`rl_recsys/data/registry.py`) maps dataset keys to pipeline classes; pipeline modules self-register on import via a `register()` call at module level. A shared download utility (`rl_recsys/data/download.py`) handles resumable downloads with tmp-file safety and optional MD5 verification. A schema module (`rl_recsys/data/schema.py`) validates required columns per data type. `prepare_data.py` imports pipeline modules to trigger registration then reads available datasets from the registry.

**Tech Stack:** Python 3.9+, pandas, pyarrow, requests, tqdm, pytest

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `rl_recsys/data/download.py` | Shared download utility: tmp-file writes, MD5, skip-if-exists |
| Create | `rl_recsys/data/schema.py` | Required-column validation per schema type |
| Create | `rl_recsys/data/registry.py` | `register()`, `get_pipeline()`, `list_datasets()` |
| Modify | `rl_recsys/data/pipelines/movielens.py` | Parameterize for 100k/1m/10m/20m/25m; self-register |
| Modify | `rl_recsys/data/pipelines/lastfm.py` | Self-register in registry |
| Modify | `rl_recsys/data/pipelines/rl4rs.py` | Self-register in registry |
| Create | `rl_recsys/data/pipelines/book_crossing.py` | Book-Crossing pipeline; self-register |
| Modify | `scripts/prepare_data.py` | Registry-driven dispatch; drop hard-coded choices |
| Create | `tests/test_download.py` | Unit tests for download utility |
| Create | `tests/test_schema.py` | Unit tests for schema validation |
| Create | `tests/test_registry.py` | Unit tests for registry |
| Create | `tests/test_pipeline_movielens.py` | Unit tests for parameterized MovieLens pipeline |
| Create | `tests/test_pipeline_book_crossing.py` | Unit tests for Book-Crossing pipeline |

---

## Task 1: Shared download utility

**Files:**
- Create: `rl_recsys/data/download.py`
- Create: `tests/test_download.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_download.py
from pathlib import Path
import pytest
from rl_recsys.data.download import _md5, download_file


def test_md5_returns_32_char_hex(tmp_path):
    f = tmp_path / "file.txt"
    f.write_bytes(b"hello")
    result = _md5(f)
    assert isinstance(result, str) and len(result) == 32


def test_download_file_skips_existing_without_network(tmp_path, monkeypatch):
    dest = tmp_path / "existing.txt"
    dest.write_bytes(b"data")
    called = []
    monkeypatch.setattr("requests.get", lambda *a, **kw: called.append(True))
    download_file("http://unused.invalid", dest)
    assert called == []


def test_download_file_raises_on_bad_checksum_after_download(tmp_path, monkeypatch):
    dest = tmp_path / "file.txt"

    class FakeResp:
        headers = {"content-length": "5"}
        def raise_for_status(self): pass
        def iter_content(self, size): yield b"hello"

    monkeypatch.setattr("requests.get", lambda *a, **kw: FakeResp())
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_file("http://unused.invalid", dest,
                      expected_md5="deadbeef00000000deadbeef00000000")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_download.py -v
```
Expected: `ModuleNotFoundError: No module named 'rl_recsys.data.download'`

- [ ] **Step 3: Implement download.py**

```python
# rl_recsys/data/download.py
from __future__ import annotations

import hashlib
from pathlib import Path

import requests
from tqdm import tqdm


def download_file(
    url: str,
    dest: Path,
    *,
    chunk_size: int = 8192,
    expected_md5: str | None = None,
) -> None:
    if dest.exists():
        if expected_md5 is None or _md5(dest) == expected_md5:
            print(f"Already downloaded: {dest.name}")
            return
        print(f"Checksum mismatch for {dest.name}, re-downloading...")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(tmp, "wb") as f, tqdm(total=total, unit="iB", unit_scale=True, desc=dest.name) as pbar:
        for chunk in response.iter_content(chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))

    if expected_md5 and _md5(tmp) != expected_md5:
        tmp.unlink()
        raise ValueError(f"Checksum mismatch for {dest.name} after download")

    tmp.rename(dest)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_download.py -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add rl_recsys/data/download.py tests/test_download.py
git commit -m "feat: add shared download utility with tmp-file writes and md5 verification"
```

---

## Task 2: Schema validation

**Files:**
- Create: `rl_recsys/data/schema.py`
- Create: `tests/test_schema.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_schema.py
import pandas as pd
import pytest
from rl_recsys.data.schema import validate_parquet_schema


def test_validate_passes_for_correct_interactions(tmp_path):
    path = tmp_path / "ok.parquet"
    pd.DataFrame({"user_id": [0], "item_id": [1], "timestamp": [0]}).to_parquet(path)
    validate_parquet_schema(path, "interactions")  # must not raise


def test_validate_fails_for_missing_column(tmp_path):
    path = tmp_path / "bad.parquet"
    pd.DataFrame({"user_id": [0], "item_id": [1]}).to_parquet(path)
    with pytest.raises(ValueError, match="missing columns"):
        validate_parquet_schema(path, "interactions")


def test_validate_fails_for_unknown_schema_type(tmp_path):
    path = tmp_path / "any.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(path)
    with pytest.raises(ValueError, match="Unknown schema type"):
        validate_parquet_schema(path, "nonexistent")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_schema.py -v
```
Expected: `ModuleNotFoundError: No module named 'rl_recsys.data.schema'`

- [ ] **Step 3: Implement schema.py**

```python
# rl_recsys/data/schema.py
from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

REQUIRED_COLUMNS: dict[str, set[str]] = {
    "interactions": {"user_id", "item_id", "timestamp"},
    "sessions": {"session_id", "user_id", "item_id", "timestamp"},
    "slates": {"request_id", "user_id"},
    "social_edges": {"user_a", "user_b"},
    "items": {"item_id"},
}


def validate_parquet_schema(path: Path, schema_type: str) -> None:
    if schema_type not in REQUIRED_COLUMNS:
        raise ValueError(
            f"Unknown schema type {schema_type!r}. Known: {sorted(REQUIRED_COLUMNS)}"
        )
    schema = pq.read_schema(path)
    missing = REQUIRED_COLUMNS[schema_type] - set(schema.names)
    if missing:
        raise ValueError(
            f"{path.name} missing columns for schema {schema_type!r}: {missing}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_schema.py -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add rl_recsys/data/schema.py tests/test_schema.py
git commit -m "feat: add parquet schema validation with required-column enforcement per data type"
```

---

## Task 3: Dataset registry

**Files:**
- Create: `rl_recsys/data/registry.py`
- Create: `tests/test_registry.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_registry.py
import pytest
from rl_recsys.data import registry as reg
from rl_recsys.data.pipelines.movielens import MovieLensPipeline


def test_register_and_get_pipeline(tmp_path):
    reg.register(
        "_test_reg_a",
        MovieLensPipeline,
        schema="interactions",
        tags=["CF"],
        variant="100k",
        raw_dir=str(tmp_path / "raw"),
        processed_dir=str(tmp_path / "proc"),
    )
    pipeline = reg.get_pipeline("_test_reg_a")
    assert isinstance(pipeline, MovieLensPipeline)


def test_list_datasets_is_sorted(tmp_path):
    reg.register("_test_z", MovieLensPipeline, schema="interactions", tags=[],
                 variant="100k", raw_dir=str(tmp_path), processed_dir=str(tmp_path))
    reg.register("_test_a", MovieLensPipeline, schema="interactions", tags=[],
                 variant="1m", raw_dir=str(tmp_path), processed_dir=str(tmp_path))
    datasets = reg.list_datasets()
    assert datasets == sorted(datasets)


def test_get_pipeline_unknown_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        reg.get_pipeline("_nonexistent_xyz_abc")


def test_get_pipeline_overrides_raw_dir(tmp_path):
    reg.register(
        "_test_override",
        MovieLensPipeline,
        schema="interactions",
        tags=["CF"],
        variant="100k",
        raw_dir="data/raw/movielens",
        processed_dir="data/processed/movielens",
    )
    pipeline = reg.get_pipeline(
        "_test_override",
        raw_dir=str(tmp_path / "custom_raw"),
        processed_dir=str(tmp_path / "custom_proc"),
    )
    assert str(pipeline.raw_dir) == str(tmp_path / "custom_raw")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_registry.py -v
```
Expected: `ModuleNotFoundError: No module named 'rl_recsys.data.registry'`

- [ ] **Step 3: Implement registry.py**

```python
# rl_recsys/data/registry.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from rl_recsys.data.pipelines.base import BasePipeline


@dataclass
class DatasetInfo:
    name: str
    pipeline_cls: Type[BasePipeline]
    schema: str
    tags: list[str]
    default_kwargs: dict = field(default_factory=dict)


_REGISTRY: dict[str, DatasetInfo] = {}


def register(
    name: str,
    pipeline_cls: Type[BasePipeline],
    schema: str,
    tags: list[str],
    **default_kwargs,
) -> None:
    _REGISTRY[name] = DatasetInfo(
        name=name,
        pipeline_cls=pipeline_cls,
        schema=schema,
        tags=tags,
        default_kwargs=default_kwargs,
    )


def get_pipeline(
    name: str,
    raw_dir: str | None = None,
    processed_dir: str | None = None,
    **kwargs,
) -> BasePipeline:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}. Available: {sorted(_REGISTRY)}")
    info = _REGISTRY[name]
    merged = {**info.default_kwargs, **kwargs}
    if raw_dir is not None:
        merged["raw_dir"] = raw_dir
    if processed_dir is not None:
        merged["processed_dir"] = processed_dir
    return info.pipeline_cls(**merged)


def list_datasets() -> list[str]:
    return sorted(_REGISTRY)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_registry.py -v
```
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add rl_recsys/data/registry.py tests/test_registry.py
git commit -m "feat: add dataset registry with register, get_pipeline, list_datasets"
```

---

## Task 4: Refactor MovieLens as parameterized family

**Files:**
- Modify: `rl_recsys/data/pipelines/movielens.py`
- Create: `tests/test_pipeline_movielens.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline_movielens.py
import pytest
from rl_recsys.data.pipelines.movielens import MovieLensPipeline


def test_default_variant_is_100k(tmp_path):
    p = MovieLensPipeline(raw_dir=str(tmp_path), processed_dir=str(tmp_path))
    assert p._variant == "100k"
    assert "ml-100k" in p._url


def test_variant_1m_sets_correct_url(tmp_path):
    p = MovieLensPipeline(variant="1m", raw_dir=str(tmp_path), processed_dir=str(tmp_path))
    assert "ml-1m" in p._url


def test_unknown_variant_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown variant"):
        MovieLensPipeline(variant="999m", raw_dir=str(tmp_path), processed_dir=str(tmp_path))


def test_all_five_variants_registered():
    import rl_recsys.data.pipelines.movielens  # noqa: F401 — triggers self-registration
    from rl_recsys.data.registry import _REGISTRY
    for key in ["movielens-100k", "movielens-1m", "movielens-10m", "movielens-20m", "movielens-25m"]:
        assert key in _REGISTRY, f"{key} not in registry"


def test_process_100k_produces_correct_schema(tmp_path):
    import pandas as pd
    raw_dir = tmp_path / "raw" / "ml-100k"
    raw_dir.mkdir(parents=True)
    # Synthesize the u.data file that MovieLens-100k uses
    (raw_dir / "u.data").write_text("1\t1\t5\t881250949\n2\t1\t3\t891717742\n")

    p = MovieLensPipeline(
        variant="100k",
        raw_dir=str(tmp_path / "raw"),
        processed_dir=str(tmp_path / "proc"),
    )
    p.process()

    out = tmp_path / "proc" / "ratings_100k.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert set(df.columns) >= {"user_id", "item_id", "rating", "timestamp"}
    assert df["user_id"].tolist() == [0, 1]  # 1-indexed → 0-indexed
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pipeline_movielens.py -v
```
Expected: failures on missing `_variant`, `_url`, and registry entries

- [ ] **Step 3: Rewrite movielens.py**

```python
# rl_recsys/data/pipelines/movielens.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pipeline_movielens.py -v
```
Expected: 5 PASSED

- [ ] **Step 5: Run full suite to check no regressions**

```bash
pytest tests/ -v
```
Expected: all previously passing tests still pass

- [ ] **Step 6: Commit**

```bash
git add rl_recsys/data/pipelines/movielens.py tests/test_pipeline_movielens.py
git commit -m "feat: parameterize MovieLens pipeline for all five variants with self-registration"
```

---

## Task 5: Add Book-Crossing pipeline

**Files:**
- Create: `rl_recsys/data/pipelines/book_crossing.py`
- Create: `tests/test_pipeline_book_crossing.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline_book_crossing.py
import pandas as pd
import pytest
from rl_recsys.data.pipelines.book_crossing import BookCrossingPipeline


def test_instantiation_sets_dirs(tmp_path):
    p = BookCrossingPipeline(raw_dir=str(tmp_path / "raw"),
                             processed_dir=str(tmp_path / "proc"))
    assert p.raw_dir.name == "raw"
    assert p.processed_dir.name == "proc"


def test_process_produces_correct_schema(tmp_path):
    raw_dir = tmp_path / "raw"
    extracted = raw_dir / "BX-CSV-Dump"
    extracted.mkdir(parents=True)
    (extracted / "BX-Book-Ratings.csv").write_text(
        '"User-ID";"ISBN";"Book-Rating"\n'
        '276725;"034545104X";0\n'
        '276726;"0155061224";5\n'
    )
    proc_dir = tmp_path / "proc"
    p = BookCrossingPipeline(raw_dir=str(raw_dir), processed_dir=str(proc_dir))
    p.process()

    out = proc_dir / "ratings.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert set(df.columns) >= {"user_id", "item_id", "rating", "timestamp"}
    assert df["user_id"].dtype.kind in ("i", "u")
    assert df["item_id"].dtype.kind in ("i", "u")
    assert (df["timestamp"] == 0).all()


def test_book_crossing_is_registered():
    import rl_recsys.data.pipelines.book_crossing  # noqa: F401
    from rl_recsys.data.registry import _REGISTRY
    assert "book-crossing" in _REGISTRY
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pipeline_book_crossing.py -v
```
Expected: `ModuleNotFoundError: No module named 'rl_recsys.data.pipelines.book_crossing'`

- [ ] **Step 3: Implement book_crossing.py**

```python
# rl_recsys/data/pipelines/book_crossing.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pipeline_book_crossing.py -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add rl_recsys/data/pipelines/book_crossing.py tests/test_pipeline_book_crossing.py
git commit -m "feat: add Book-Crossing pipeline with schema validation and registry self-registration"
```

---

## Task 6: Add self-registration to existing lastfm and rl4rs pipelines

**Files:**
- Modify: `rl_recsys/data/pipelines/lastfm.py`
- Modify: `rl_recsys/data/pipelines/rl4rs.py`

- [ ] **Step 1: Add registration to lastfm.py**

Append to the bottom of `rl_recsys/data/pipelines/lastfm.py`:

```python
from rl_recsys.data.registry import register  # noqa: E402

register(
    "lastfm-1k",
    LastFMPipeline,
    schema="sessions",
    tags=["Session", "music"],
    raw_dir="data/raw/lastfm",
    processed_dir="data/processed/lastfm",
)
```

- [ ] **Step 2: Add registration to rl4rs.py**

Append to the bottom of `rl_recsys/data/pipelines/rl4rs.py`:

```python
from rl_recsys.data.registry import register  # noqa: E402

register(
    "rl4rs",
    RL4RSPipeline,
    schema="slates",
    tags=["RL/Slate"],
    raw_dir="data/raw/rl4rs",
    processed_dir="data/processed/rl4rs",
)
```

- [ ] **Step 3: Verify both datasets appear in the registry**

```bash
python - <<'EOF'
import rl_recsys.data.pipelines.lastfm
import rl_recsys.data.pipelines.rl4rs
from rl_recsys.data.registry import list_datasets
print(list_datasets())
EOF
```
Expected output includes: `'lastfm-1k'` and `'rl4rs'`

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ -v
```
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add rl_recsys/data/pipelines/lastfm.py rl_recsys/data/pipelines/rl4rs.py
git commit -m "feat: add registry self-registration to lastfm and rl4rs pipelines"
```

---

## Task 7: Refactor prepare_data.py to registry-driven dispatch

**Files:**
- Modify: `scripts/prepare_data.py`

- [ ] **Step 1: Rewrite prepare_data.py**

```python
# scripts/prepare_data.py
from __future__ import annotations

import argparse

# Import pipeline modules to trigger self-registration before list_datasets() is called
import rl_recsys.data.pipelines.movielens  # noqa: F401
import rl_recsys.data.pipelines.lastfm  # noqa: F401
import rl_recsys.data.pipelines.rl4rs  # noqa: F401
import rl_recsys.data.pipelines.book_crossing  # noqa: F401

from rl_recsys.data.registry import get_pipeline, list_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare datasets for rl-recsys.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list_datasets(),
        metavar="DATASET",
        help=f"Dataset key. Available: {list_datasets()}",
    )
    parser.add_argument("--raw-dir", type=str, default=None,
                        help="Override default raw data directory")
    parser.add_argument("--processed-dir", type=str, default=None,
                        help="Override default processed data directory")
    parser.add_argument("--download", action="store_true",
                        help="Download raw files")
    parser.add_argument("--process", action="store_true",
                        help="Process raw files into Parquet")

    args = parser.parse_args()
    pipeline = get_pipeline(
        args.dataset,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
    )

    if args.download:
        pipeline.download()
    if args.process:
        pipeline.process()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify help lists all registered datasets**

```bash
python scripts/prepare_data.py --help
```
Expected: help text with `DATASET` choices listing all 8 registered keys:
`book-crossing, lastfm-1k, movielens-100k, movielens-10m, movielens-1m, movielens-20m, movielens-25m, rl4rs`

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v
```
Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add scripts/prepare_data.py
git commit -m "refactor: make prepare_data.py registry-driven with --raw-dir and --processed-dir overrides"
```

---

## Adding Future Datasets (Template)

For every new dataset added after this plan, follow this pattern:

1. Create `rl_recsys/data/pipelines/<name>.py` implementing `BasePipeline` with:
   - `download_file()` from `rl_recsys.data.download`
   - `validate_parquet_schema()` from `rl_recsys.data.schema` at end of `process()`
   - `register(...)` call at module level
2. Add `import rl_recsys.data.pipelines.<name>  # noqa: F401` to `scripts/prepare_data.py`
3. Add a test file `tests/test_pipeline_<name>.py` with a synthesized-data test for `process()` output schema and a registration test
4. For datasets >1M rows, use pandas `chunksize` or PyArrow batch reading in `process()`
