from __future__ import annotations

import torch.nn as nn


def build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    activation: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    """Build a simple feedforward MLP."""
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)
