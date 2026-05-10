# rl_recsys/agents/sasrec.py
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecObs


class _SASRecEncoder(nn.Module):
    def __init__(
        self,
        num_candidates: int,
        hidden_dim: int,
        n_heads: int,
        n_blocks: int,
        max_history_len: int,
    ) -> None:
        super().__init__()
        # +1 for pad/sentinel token at index `num_candidates`.
        self._pad_idx = num_candidates
        self.item_emb = nn.Embedding(
            num_candidates + 1, hidden_dim, padding_idx=self._pad_idx,
        )
        self.pos_emb = nn.Embedding(max_history_len + 1, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=4 * hidden_dim,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)

    def forward(
        self,
        item_ids: torch.Tensor,        # (B, T, slate_size) long
        click_mask: torch.Tensor,      # (B, T, slate_size) float (0/1)
        positions: torch.Tensor,       # (B, T) long
    ) -> torch.Tensor:
        # Token embedding: mean-pool slate items, scaled by (1 + click).
        emb = self.item_emb(item_ids) * (1.0 + click_mask).unsqueeze(-1)
        token = emb.mean(dim=2)                 # (B, T, hidden)
        token = token + self.pos_emb(positions) # (B, T, hidden)
        out = self.encoder(token)               # (B, T, hidden)
        return out[:, -1, :]                    # last position pooled state


class SASRecAgent(Agent):
    """Self-attention sequential ranker: encoder over (slate, clicks)
    history; per-candidate score = h_session . W_out @ item_embedding."""

    def __init__(
        self,
        slate_size: int,
        num_candidates: int,
        item_dim: int,
        *,
        hidden_dim: int = 64,
        n_heads: int = 2,
        n_blocks: int = 2,
        max_history_len: int = 20,
        epochs: int = 10,
        device: str = "cuda",
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA required for SASRecAgent; pass device='cpu' to run on CPU"
            )
        self._slate_size = int(slate_size)
        self._num_candidates = int(num_candidates)
        self._item_dim = int(item_dim)
        self._max_history_len = int(max_history_len)
        self._epochs = int(epochs)
        self._hidden_dim = int(hidden_dim)
        self._device = torch.device(device)
        self._encoder = _SASRecEncoder(
            num_candidates=num_candidates,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_blocks=n_blocks,
            max_history_len=max_history_len,
        ).to(self._device)
        self._out = nn.Linear(hidden_dim, hidden_dim, bias=True).to(self._device)

    def _build_history_tensors(self, history):
        T = min(len(history), self._max_history_len)
        if T == 0:
            ids = torch.full(
                (1, 1, self._slate_size), self._encoder._pad_idx,
                dtype=torch.long, device=self._device,
            )
            clicks = torch.zeros((1, 1, self._slate_size), device=self._device)
            pos = torch.zeros((1, 1), dtype=torch.long, device=self._device)
            return ids, clicks, pos
        recent = history[-T:]
        ids = torch.tensor(
            np.stack([h.slate for h in recent]),
            dtype=torch.long, device=self._device,
        ).unsqueeze(0)
        clicks = torch.tensor(
            np.stack([h.clicks for h in recent]),
            dtype=torch.float32, device=self._device,
        ).unsqueeze(0)
        pos = torch.arange(
            1, T + 1, dtype=torch.long, device=self._device,
        ).unsqueeze(0)
        return ids, clicks, pos

    def score_items(self, obs: RecObs) -> np.ndarray:
        self._encoder.eval()
        with torch.no_grad():
            ids, clicks, pos = self._build_history_tensors(obs.history)
            h = self._encoder(ids, clicks, pos)
            h = self._out(h)
            all_items = torch.arange(
                self._num_candidates, dtype=torch.long, device=self._device,
            )
            item_emb = self._encoder.item_emb(all_items)
            scores = (item_emb @ h.squeeze(0)).cpu().numpy().astype(np.float64)
        return scores

    def select_slate(self, obs: RecObs) -> np.ndarray:
        return np.argsort(self.score_items(obs))[-self._slate_size:][::-1].astype(
            np.int64
        )

    def train_offline(self, source, *, seed: int = 0) -> dict[str, float]:
        torch.manual_seed(seed)
        self._encoder.train()
        opt = torch.optim.Adam(
            list(self._encoder.parameters()) + list(self._out.parameters()),
            lr=1e-3,
        )
        last_loss = float("nan")
        for epoch in range(self._epochs):
            for traj in source.iter_trajectories(seed=seed + epoch):
                # At each step (except the first, which has no history),
                # predict the items the user actually clicked at this step
                # given the prior session history.
                for step in traj:
                    if len(step.obs.history) == 0:
                        continue
                    ids, clicks, pos = self._build_history_tensors(step.obs.history)
                    pooled = self._out(self._encoder(ids, clicks, pos))
                    all_items = torch.arange(
                        self._num_candidates, dtype=torch.long, device=self._device,
                    )
                    item_emb = self._encoder.item_emb(all_items)
                    logits = item_emb @ pooled.squeeze(0)
                    logp = torch.log_softmax(logits, dim=-1)
                    pos_idx = torch.tensor(
                        step.logged_action[step.logged_clicks > 0],
                        dtype=torch.long, device=self._device,
                    )
                    if pos_idx.numel() == 0:
                        continue
                    loss = -logp[pos_idx].mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    last_loss = float(loss.item())
        return {"loss": last_loss, "epochs": float(self._epochs)}

    def update(self, obs, slate, reward, clicks, next_obs) -> dict[str, float]:
        return {}
