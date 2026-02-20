from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BasePipeline(ABC):
    """Base class for data pipelines (download, process, load)."""

    def __init__(self, raw_dir: str | Path, processed_dir: str | Path) -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def download(self) -> None:
        """Download the raw dataset files."""
        ...

    @abstractmethod
    def process(self) -> None:
        """Clean and transform raw data into a usable format (e.g., Parquet)."""
        ...
