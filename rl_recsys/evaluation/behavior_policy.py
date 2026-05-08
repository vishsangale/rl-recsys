from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


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
    """
    import pandas as pd  # local import to keep module-level deps minimal

    df = pd.read_parquet(parquet_path)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Build training tensors. Every row contributes slate_size examples.
    users: list[np.ndarray] = []
    cands: list[np.ndarray] = []
    positions: list[int] = []
    targets: list[int] = []
    for _, row in df.iterrows():
        user = np.asarray(row["user_state"], dtype=np.float64)
        cand = np.array(list(row["candidate_features"]), dtype=np.float64)
        cand_ids = list(row["candidate_ids"])
        slate = list(row["slate"])
        for k in range(min(slate_size, len(slate))):
            target_item_id = int(slate[k])
            try:
                target_idx = cand_ids.index(target_item_id)
            except ValueError:
                continue  # logged item not in candidate universe — skip
            users.append(user)
            cands.append(cand)
            positions.append(k)
            targets.append(target_idx)

    if not users:
        raise ValueError("no training tuples derivable from parquet")

    user_t = torch.as_tensor(np.stack(users), dtype=torch.float64)
    cand_t = torch.as_tensor(np.stack(cands), dtype=torch.float64)
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
                    user_t[i], cand_t[i], int(pos_t[i].item())
                )
                log_probs = torch.log_softmax(logits, dim=-1)
                losses.append(-log_probs[int(target_t[i].item())])
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
