from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BehaviorPolicy(nn.Module):
    """Per-position softmax classifier estimating μ(item | context, position).

    Forward pass scores every (candidate, position) pair given a context.
    slate_propensity returns Π_k softmax(scores)[slate[k]] across positions.
    """

    def __init__(
        self,
        *,
        user_dim: int,
        item_dim: int,
        slate_size: int,
        num_items: int,
        hidden_dim: int = 64,
        seed: int = 0,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self._user_dim = user_dim
        self._item_dim = item_dim
        self._slate_size = slate_size
        self._num_items = num_items
        self._mlp = nn.Sequential(
            nn.Linear(user_dim + item_dim + slate_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).double()

    def _score_position(
        self,
        user_feat: torch.Tensor,
        candidate_feats: torch.Tensor,
        position: int,
    ) -> torch.Tensor:
        """Returns logits of shape (num_candidates,) for the given position."""
        n = candidate_feats.shape[0]
        position_onehot = torch.zeros(self._slate_size, dtype=torch.float64)
        position_onehot[position] = 1.0
        user_tile = user_feat.unsqueeze(0).expand(n, -1)
        position_tile = position_onehot.unsqueeze(0).expand(n, -1)
        x = torch.cat([user_tile, candidate_feats, position_tile], dim=1)
        return self._mlp(x).squeeze(-1)

    def slate_propensity(
        self,
        user_features: np.ndarray,
        candidate_features: np.ndarray,
        slate: np.ndarray,
    ) -> float:
        """π_b(slate | context) = Π_k softmax(score(·, k))[slate[k]]."""
        user = torch.as_tensor(user_features, dtype=torch.float64)
        cand = torch.as_tensor(candidate_features, dtype=torch.float64)
        slate_t = torch.as_tensor(np.asarray(slate, dtype=np.int64))
        log_prob_total = 0.0
        with torch.no_grad():
            for k in range(int(slate_t.shape[0])):
                logits = self._score_position(user, cand, k)
                log_probs = torch.log_softmax(logits, dim=-1)
                log_prob_total += float(log_probs[int(slate_t[k])].item())
        result = float(np.exp(log_prob_total))
        if result <= 0.0:
            raise ValueError("zero propensity in logged slate")
        return result


def _build_universe_from_df(df) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Build sorted candidate universe from slate + item_features columns.

    Returns:
        universe_ids: sorted unique item ids as int64 array
        universe_features: feature array in the same order
        id_to_idx: item_id -> position in universe
    """
    import pyarrow.compute as pc
    import pyarrow as pa

    # Collect unique ids using pyarrow for speed on large DataFrames.
    # Fall back to pure Python if the column is not a pa.ChunkedArray.
    slates = df["slate"].tolist()
    flat = [int(x) for s in slates for x in s]
    universe_ids = np.sort(np.unique(np.array(flat, dtype=np.int64)))

    feature_for: dict[int, list[float]] = {}
    for slate, item_feats in zip(df["slate"], df["item_features"]):
        for item_id, feat in zip(slate, item_feats):
            if int(item_id) not in feature_for:
                feature_for[int(item_id)] = list(feat)
        if len(feature_for) == len(universe_ids):
            break

    universe_features = np.array(
        [feature_for[int(i)] for i in universe_ids], dtype=np.float64
    )
    id_to_idx = {int(cid): k for k, cid in enumerate(universe_ids)}
    return universe_ids, universe_features, id_to_idx


def fit_behavior_policy(
    parquet_path: Path,
    *,
    user_dim: int,
    item_dim: int,
    slate_size: int,
    num_items: int,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-2,
    seed: int = 0,
    hidden_dim: int = 64,
) -> BehaviorPolicy:
    """Fit a per-position softmax classifier on logged slate placements.

    Each row of `parquet_path` is expanded into `slate_size` training tuples:
    (user_state, candidate_features, position k, target = candidate-index of slate[k]).

    The candidate universe (sorted unique item ids + feature vectors) is derived
    from the parquet's slate and item_features columns — no candidate_ids /
    candidate_features columns required in the parquet.
    """
    import pandas as pd  # local import to keep module-level deps minimal

    df = pd.read_parquet(parquet_path)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Build candidate universe once from slate+item_features.
    universe_ids, universe_features, id_to_idx = _build_universe_from_df(df)
    cand_t_shared = torch.as_tensor(universe_features, dtype=torch.float64)

    # Build training tensors. Every row contributes slate_size examples.
    users: list[np.ndarray] = []
    positions: list[int] = []
    targets: list[int] = []
    for _, row in df.iterrows():
        user = np.asarray(row["user_state"], dtype=np.float64)
        slate = list(row["slate"])
        for k in range(min(slate_size, len(slate))):
            target_item_id = int(slate[k])
            target_idx = id_to_idx.get(target_item_id)
            if target_idx is None:
                continue  # logged item not in candidate universe — skip
            users.append(user)
            positions.append(k)
            targets.append(target_idx)

    if not users:
        raise ValueError("no training tuples derivable from parquet")

    user_t = torch.as_tensor(np.stack(users), dtype=torch.float64)
    pos_t = torch.as_tensor(np.array(positions), dtype=torch.long)
    target_t = torch.as_tensor(np.array(targets), dtype=torch.long)

    model = BehaviorPolicy(
        user_dim=user_dim, item_dim=item_dim, slate_size=slate_size,
        num_items=num_items, hidden_dim=hidden_dim, seed=seed,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n = user_t.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = order[start : start + batch_size]
            losses: list[torch.Tensor] = []
            for i in batch_idx:
                logits = model._score_position(
                    user_t[i], cand_t_shared, int(pos_t[i].item())
                )
                log_probs = torch.log_softmax(logits, dim=-1)
                losses.append(-log_probs[int(target_t[i].item())])
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def held_out_nll(
    model: BehaviorPolicy,
    df,
    *,
    universe_ids: np.ndarray | None = None,
    universe_features: np.ndarray | None = None,
) -> float:
    """Average negative log-likelihood over (row, slate-position) tuples.

    `df` rows must include `user_state`, `slate`, and either:
      - `item_features` (to derive the universe internally), OR
      - explicit `universe_ids` and `universe_features` keyword arguments.

    Returns mean -log p(slate[k] | context, k) across all positions.
    """
    if universe_ids is None or universe_features is None:
        universe_ids, universe_features, id_to_idx = _build_universe_from_df(df)
    else:
        id_to_idx = {int(cid): k for k, cid in enumerate(universe_ids)}

    cand = torch.as_tensor(universe_features, dtype=torch.float64)

    losses: list[float] = []
    with torch.no_grad():
        for _, row in df.iterrows():
            user = torch.as_tensor(
                np.array(list(row["user_state"]), dtype=np.float64),
                dtype=torch.float64,
            )
            slate = list(row["slate"])
            for k in range(len(slate)):
                target_item_id = int(slate[k])
                target_idx = id_to_idx.get(target_item_id)
                if target_idx is None:
                    continue
                logits = model._score_position(user, cand, k)
                log_probs = torch.log_softmax(logits, dim=-1)
                losses.append(-float(log_probs[target_idx].item()))
    if not losses:
        raise ValueError("no held-out tuples; cannot compute NLL")
    return float(np.mean(losses))


def fit_behavior_policy_with_calibration(
    parquet_path: Path,
    *,
    user_dim: int,
    item_dim: int,
    slate_size: int,
    num_items: int,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-2,
    seed: int = 0,
    hidden_dim: int = 64,
    nll_threshold: float | None = None,
    held_out_fraction: float = 0.1,
) -> BehaviorPolicy:
    """Fit + held-out NLL gate.

    Splits parquet 90/10 (deterministic via seed), fits on the 90, evaluates
    NLL on the 10, and raises if NLL > nll_threshold.
    Default threshold = 2 * log(num_items)  (twice uniform NLL).
    """
    import pandas as pd

    if nll_threshold is None:
        nll_threshold = 2.0 * float(np.log(num_items))

    df = pd.read_parquet(parquet_path)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    n_held = max(1, int(len(df) * held_out_fraction))
    held_idx = perm[:n_held]
    train_idx = perm[n_held:]

    # Build the universe once from the full df so train + held-out share the
    # same candidate ordering.
    universe_ids, universe_features, id_to_idx = _build_universe_from_df(df)

    train_path = parquet_path.with_name(parquet_path.stem + "_train.parquet")
    df.iloc[train_idx].to_parquet(train_path, index=False)
    try:
        model = fit_behavior_policy(
            train_path, user_dim=user_dim, item_dim=item_dim,
            slate_size=slate_size, num_items=num_items,
            epochs=epochs, batch_size=batch_size,
            learning_rate=learning_rate, seed=seed, hidden_dim=hidden_dim,
        )
    finally:
        train_path.unlink(missing_ok=True)

    held_df = df.iloc[held_idx]
    nll = held_out_nll(
        model, held_df,
        universe_ids=universe_ids, universe_features=universe_features,
    )
    if nll > nll_threshold:
        raise ValueError(
            f"behavior policy NLL exceeds threshold: {nll:.4f} > {nll_threshold:.4f}"
        )
    logger.info("Behavior policy held-out NLL = %.4f (threshold %.4f)", nll, nll_threshold)
    return model
