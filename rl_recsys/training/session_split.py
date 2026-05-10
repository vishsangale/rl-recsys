from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq


def _bucket(sid: int, seed: int) -> int:
    """Return a deterministic integer in [0, 1000) for (seed, sid)."""
    h = hashlib.blake2b(
        f"{seed}:{sid}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(h, "big") % 1000


def split_session_ids(
    parquet_path: str | Path,
    *,
    train_fraction: float = 0.5,
    seed: int = 42,
) -> tuple[set[int], set[int]]:
    """Deterministic session-level partition of a sessions_b.parquet.

    Reads only the session_id column. For each unique session_id,
    bucket = blake2b(seed:sid) % 1000; if bucket < int(1000 * train_fraction)
    the session is in the train half, else eval.

    Returns (train_ids, eval_ids). Raises ValueError if train_fraction is
    not in (0, 1) or if either side is empty.

    Granularity: bucket modulus is 1000, so train_fraction is effectively
    quantized to 0.001 increments.
    """
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(
            f"train_fraction must be in (0, 1); got {train_fraction!r}"
        )

    table = pq.read_table(parquet_path, columns=["session_id"])
    unique_ids = pc.unique(table["session_id"]).to_pylist()
    threshold = int(1000 * train_fraction)

    train: set[int] = set()
    evl: set[int] = set()
    for sid in unique_ids:
        if _bucket(int(sid), seed) < threshold:
            train.add(int(sid))
        else:
            evl.add(int(sid))

    if not train or not evl:
        raise ValueError(
            f"split produced an empty side (train={len(train)}, eval={len(evl)}); "
            "try a different train_fraction or seed"
        )
    return train, evl
