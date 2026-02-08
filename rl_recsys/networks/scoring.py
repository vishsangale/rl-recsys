from __future__ import annotations

import torch
import torch.nn as nn

from rl_recsys.networks.mlp import build_mlp


class ItemScorer(nn.Module):
    """Scores each candidate item given user + item features.

    Input:  user_features (batch, user_dim), item_features (batch, num_candidates, item_dim)
    Output: scores (batch, num_candidates)
    """

    def __init__(
        self,
        user_dim: int,
        item_dim: int,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
        self.net = build_mlp(user_dim + item_dim, hidden_dims, output_dim=1)

    def forward(
        self, user_features: torch.Tensor, item_features: torch.Tensor
    ) -> torch.Tensor:
        # user_features: (B, user_dim) -> (B, 1, user_dim) -> (B, N, user_dim)
        B, N, _ = item_features.shape
        user_exp = user_features.unsqueeze(1).expand(B, N, -1)
        combined = torch.cat([user_exp, item_features], dim=-1)  # (B, N, user+item)
        scores = self.net(combined).squeeze(-1)  # (B, N)
        return scores
