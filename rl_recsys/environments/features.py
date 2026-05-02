from __future__ import annotations

import hashlib

import numpy as np


def hashed_vector(prefix: str, entity_id: int, dim: int) -> np.ndarray:
    digest = hashlib.blake2b(
        f"{prefix}:{entity_id}".encode("utf-8"), digest_size=8
    )
    seed = int.from_bytes(digest.digest(), byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec
