# rl_recsys/agents/decision_transformer.py
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecObs


class DecisionTransformerAgent(Agent):
    """Decision Transformer — sequence model over (R-to-go, state, action).
    Conditions on a target return at inference; decodes top-k by cosine
    similarity between the predicted action embedding and item embeddings."""

    def __init__(
        self,
        slate_size: int,
        num_candidates: int,
        user_dim: int,
        item_dim: int,
        *,
        hidden_dim: int = 64,
        n_blocks: int = 3,
        context_window: int = 20,
        target_return: float = 10.0,
        gamma: float = 0.95,
        epochs: int = 10,
        device: str = "cuda",
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA required for DecisionTransformerAgent; "
                "pass device='cpu' to run on CPU"
            )
        self._slate_size = int(slate_size)
        self._num_candidates = int(num_candidates)
        self._user_dim = int(user_dim)
        self._context_window = int(context_window)
        self._target_return = float(target_return)
        self._gamma = float(gamma)
        self._epochs = int(epochs)
        self._hidden_dim = int(hidden_dim)
        self._device = torch.device(device)

        self.r_proj = nn.Linear(1, hidden_dim).to(self._device)
        self.s_proj = nn.Linear(user_dim, hidden_dim).to(self._device)
        self.item_emb = nn.Embedding(num_candidates, hidden_dim).to(self._device)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=2, dim_feedforward=4 * hidden_dim,
            batch_first=True, activation="gelu",
        )
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks).to(
            self._device,
        )
        self.head = nn.Linear(hidden_dim, hidden_dim).to(self._device)

    def _action_emb(self, slate: np.ndarray) -> torch.Tensor:
        ids = torch.tensor(slate, dtype=torch.long, device=self._device)
        return self.item_emb(ids).mean(dim=0)  # (hidden,)

    def _state_emb(self, obs: RecObs) -> torch.Tensor:
        return self.s_proj(
            torch.tensor(obs.user_features, dtype=torch.float32,
                         device=self._device),
        )

    def _build_sequence(self, history, future_return: float) -> torch.Tensor:
        # Build the prefix [R_0, s_0, a_0, ..., R_t-1, s_t-1, a_t-1, R_t].
        # We always end on the (R_t) token so the next predicted thing is s_t,
        # then a_t. Caller appends s_t externally.
        tokens: list[torch.Tensor] = []
        for h in history[-self._context_window:]:
            r = self.r_proj(
                torch.tensor([0.0], dtype=torch.float32, device=self._device),
            )
            s = self.s_proj(
                torch.tensor(np.zeros(self._user_dim), dtype=torch.float32,
                             device=self._device),
            )
            a = self._action_emb(h.slate)
            tokens.extend([r, s, a])
        # Final R (target return).
        r_final = self.r_proj(
            torch.tensor([future_return], dtype=torch.float32, device=self._device),
        )
        tokens.append(r_final)
        return torch.stack(tokens, dim=0).unsqueeze(0)  # (1, L, hidden)

    def score_items(self, obs: RecObs) -> np.ndarray:
        self.tr.eval()
        with torch.no_grad():
            seq = self._build_sequence(obs.history, self._target_return)
            # Append state token; predict action embedding from final position.
            s = self._state_emb(obs).unsqueeze(0).unsqueeze(0)
            seq = torch.cat([seq, s], dim=1)
            out = self.tr(seq)
            pred = self.head(out[:, -1, :]).squeeze(0)
            all_items = torch.arange(
                self._num_candidates, dtype=torch.long, device=self._device,
            )
            item_emb = self.item_emb(all_items)
            scores = torch.nn.functional.cosine_similarity(
                pred.unsqueeze(0), item_emb, dim=1,
            )
        return scores.cpu().numpy().astype(np.float64)

    def select_slate(self, obs: RecObs) -> np.ndarray:
        return np.argsort(self.score_items(obs))[-self._slate_size:][::-1].astype(
            np.int64
        )

    def train_offline(self, source, *, seed: int = 0) -> dict[str, float]:
        torch.manual_seed(seed)
        self.tr.train()
        params = (
            list(self.r_proj.parameters()) + list(self.s_proj.parameters()) +
            list(self.item_emb.parameters()) + list(self.tr.parameters()) +
            list(self.head.parameters())
        )
        opt = torch.optim.Adam(params, lr=1e-3)
        last_loss = float("nan")
        for epoch in range(self._epochs):
            for traj in source.iter_trajectories(seed=seed + epoch):
                # Compute return-to-go.
                rewards = np.array(
                    [s.logged_reward for s in traj], dtype=np.float64,
                )
                rtg = np.zeros_like(rewards)
                running = 0.0
                for t in range(len(rewards) - 1, -1, -1):
                    running = rewards[t] + self._gamma * running
                    rtg[t] = running
                for t, step in enumerate(traj):
                    seq = self._build_sequence(step.obs.history, float(rtg[t]))
                    s = self._state_emb(step.obs).unsqueeze(0).unsqueeze(0)
                    seq = torch.cat([seq, s], dim=1)
                    out = self.tr(seq)
                    pred = self.head(out[:, -1, :]).squeeze(0)
                    target = self._action_emb(step.logged_action)
                    loss = nn.functional.mse_loss(pred, target)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    last_loss = float(loss.item())
        return {"loss": last_loss, "epochs": float(self._epochs)}

    def update(self, obs, slate, reward, clicks, next_obs) -> dict[str, float]:
        return {}
