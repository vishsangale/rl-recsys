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
