from __future__ import annotations

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
