from __future__ import annotations

import numpy as np


def ndcg_at_k(clicks: np.ndarray, k: int | None = None) -> float:
    """Normalized Discounted Cumulative Gain.

    Args:
        clicks: binary relevance array in slate order.
        k: truncation depth (default: full slate).
    """
    if k is None:
        k = len(clicks)
    clicks = clicks[:k]
    if clicks.sum() == 0:
        return 0.0
    positions = np.arange(1, len(clicks) + 1, dtype=np.float32)
    dcg = float((clicks / np.log2(positions + 1)).sum())
    # ideal: all clicks at top positions
    ideal = np.sort(clicks)[::-1]
    idcg = float((ideal / np.log2(positions + 1)).sum())
    return dcg / idcg if idcg > 0 else 0.0


def mrr(clicks: np.ndarray) -> float:
    """Mean Reciprocal Rank — reciprocal rank of the first click."""
    indices = np.where(clicks > 0)[0]
    if len(indices) == 0:
        return 0.0
    return 1.0 / (indices[0] + 1)


def ctr(clicks: np.ndarray) -> float:
    """Click-through rate — fraction of positions clicked."""
    if len(clicks) == 0:
        return 0.0
    return float(clicks.mean())
