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
        device=None,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self._user_dim = user_dim
        self._item_dim = item_dim
        self._slate_size = slate_size
        self._num_items = num_items
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._mlp = nn.Sequential(
            nn.Linear(user_dim + item_dim + slate_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).double().to(self._device)

    def _score_position(
        self,
        user_feat: torch.Tensor,
        candidate_feats: torch.Tensor,
        position: int,
    ) -> torch.Tensor:
        """Returns logits of shape (num_candidates,) for the given position."""
        user_feat = user_feat.to(self._device)
        candidate_feats = candidate_feats.to(self._device)
        n = candidate_feats.shape[0]
        position_onehot = torch.zeros(self._slate_size, dtype=torch.float64, device=self._device)
        position_onehot[position] = 1.0
        user_tile = user_feat.unsqueeze(0).expand(n, -1)
        position_tile = position_onehot.unsqueeze(0).expand(n, -1)
        x = torch.cat([user_tile, candidate_feats, position_tile], dim=1)
        return self._mlp(x).squeeze(-1)

    def _score_batch(
        self,
        users: torch.Tensor,        # (B, user_dim)
        cands: torch.Tensor,        # (B, num_candidates, item_dim)
        positions: torch.Tensor,    # (B,), long
    ) -> torch.Tensor:
        """Batched forward. Returns logits of shape (B, num_candidates)."""
        users = users.to(self._device)
        cands = cands.to(self._device)
        positions = positions.to(self._device)
        b, n, _ = cands.shape
        user_tile = users.unsqueeze(1).expand(b, n, -1)
        pos_onehot = torch.zeros(b, self._slate_size, dtype=torch.float64, device=self._device)
        pos_onehot.scatter_(1, positions.unsqueeze(1), 1.0)
        pos_tile = pos_onehot.unsqueeze(1).expand(b, n, -1)
        x = torch.cat([user_tile, cands, pos_tile], dim=2)
        return self._mlp(x).squeeze(-1)

    def slate_propensity(
        self,
        user_features: np.ndarray,
        candidate_features: np.ndarray,
        slate: np.ndarray,
    ) -> float:
        """π_b(slate | context) = Π_k softmax(score(·, k))[slate[k]]."""
        user = torch.as_tensor(user_features, dtype=torch.float64, device=self._device)
        cand = torch.as_tensor(candidate_features, dtype=torch.float64, device=self._device)
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
    device=None,
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

    model = BehaviorPolicy(
        user_dim=user_dim, item_dim=item_dim, slate_size=slate_size,
        num_items=num_items, hidden_dim=hidden_dim, seed=seed, device=device,
    )
    logger.info("BehaviorPolicy training on device=%s", model._device)

    # Build candidate universe once from slate+item_features.
    universe_ids, universe_features, id_to_idx = _build_universe_from_df(df)
    cand_t_shared = torch.as_tensor(universe_features, dtype=torch.float64, device=model._device)

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

    user_t = torch.as_tensor(np.stack(users), dtype=torch.float64, device=model._device)
    pos_t = torch.as_tensor(np.array(positions), dtype=torch.long, device=model._device)
    target_t = torch.as_tensor(np.array(targets), dtype=torch.long, device=model._device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n = user_t.shape[0]
    for epoch in range(epochs):
        order = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            batch_idx = order[start : start + batch_size]
            bi_t = torch.as_tensor(batch_idx, dtype=torch.long, device=model._device)
            b = bi_t.shape[0]
            # Expand shared candidate universe as a zero-copy view per sample.
            cands_b = cand_t_shared.unsqueeze(0).expand(b, -1, -1)
            logits = model._score_batch(
                user_t.index_select(0, bi_t),
                cands_b,
                pos_t.index_select(0, bi_t),
            )  # (b, num_candidates)
            log_probs = torch.log_softmax(logits, dim=-1)
            targets = target_t.index_select(0, bi_t).unsqueeze(1)
            losses = -log_probs.gather(1, targets).squeeze(1)
            loss = losses.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        logger.info(
            "epoch %d/%d  mean_loss=%.4f", epoch + 1, epochs, epoch_loss / max(1, n_batches)
        )

    return model


def held_out_nll(
    model: BehaviorPolicy,
    df,
    *,
    universe_ids: np.ndarray | None = None,
    universe_features: np.ndarray | None = None,
    chunk_size: int = 4096,
) -> float:
    """Average negative log-likelihood over (row, slate-position) tuples (vectorized).

    `df` rows must include `user_state`, `slate`, and either:
      - `item_features` (to derive the universe internally), OR
      - explicit `universe_ids` and `universe_features` keyword arguments.

    Returns mean -log p(slate[k] | context, k) across all positions.
    Processes tuples in chunks of `chunk_size` to bound peak memory.
    """
    if universe_ids is None or universe_features is None:
        universe_ids, universe_features, id_to_idx = _build_universe_from_df(df)
    else:
        id_to_idx = {int(cid): k for k, cid in enumerate(universe_ids)}

    cand_t_shared = torch.as_tensor(universe_features, dtype=torch.float64, device=model._device)

    # Build all (user, position, target) tuples up front.
    users: list[np.ndarray] = []
    positions: list[int] = []
    targets: list[int] = []
    for _, row in df.iterrows():
        user = np.asarray(row["user_state"], dtype=np.float64)
        slate = list(row["slate"])
        for k in range(len(slate)):
            target_item_id = int(slate[k])
            target_idx = id_to_idx.get(target_item_id)
            if target_idx is None:
                continue
            users.append(user)
            positions.append(k)
            targets.append(target_idx)

    if not users:
        raise ValueError("no held-out tuples; cannot compute NLL")

    user_t = torch.as_tensor(np.stack(users), dtype=torch.float64, device=model._device)
    pos_t = torch.as_tensor(np.array(positions), dtype=torch.long, device=model._device)
    target_t = torch.as_tensor(np.array(targets), dtype=torch.long, device=model._device)

    n = user_t.shape[0]
    total_nll = 0.0
    with torch.no_grad():
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            b = end - start
            cands_b = cand_t_shared.unsqueeze(0).expand(b, -1, -1)
            logits = model._score_batch(
                user_t[start:end],
                cands_b,
                pos_t[start:end],
            )  # (b, num_candidates)
            log_probs = torch.log_softmax(logits, dim=-1)
            nlls = -log_probs.gather(1, target_t[start:end].unsqueeze(1)).squeeze(1)
            total_nll += float(nlls.sum().item())
    return total_nll / n


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
    device=None,
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
            device=device,
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
