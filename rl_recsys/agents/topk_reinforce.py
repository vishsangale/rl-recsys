# rl_recsys/agents/topk_reinforce.py
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from rl_recsys.agents.base import Agent
from rl_recsys.agents.sasrec import _SASRecEncoder
from rl_recsys.environments.base import RecObs


class TopKReinforceAgent(Agent):
    """Chen et al. 2019 top-K off-policy correction over a SASRec-style
    encoder. Importance-weighted policy gradient using the loader's
    precomputed propensities."""

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
        clip_c: float = 10.0,
        device: str = "cuda",
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA required for TopKReinforceAgent; "
                "pass device='cpu' to run on CPU"
            )
        self._slate_size = int(slate_size)
        self._num_candidates = int(num_candidates)
        self._max_history_len = int(max_history_len)
        self._epochs = int(epochs)
        self._clip_c = float(clip_c)
        self._device = torch.device(device)
        self._encoder = _SASRecEncoder(
            num_candidates=num_candidates,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_blocks=n_blocks,
            max_history_len=max_history_len,
        ).to(self._device)
        self._heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_candidates) for _ in range(slate_size)
        ]).to(self._device)

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

    def _per_position_logits(self, obs: RecObs) -> torch.Tensor:
        ids, clicks, pos = self._build_history_tensors(obs.history)
        h = self._encoder(ids, clicks, pos).squeeze(0)
        return torch.stack([head(h) for head in self._heads])  # (slate_size, n_cands)

    def score_items(self, obs: RecObs) -> np.ndarray:
        self._encoder.eval()
        with torch.no_grad():
            logits = self._per_position_logits(obs)
        return logits.sum(dim=0).cpu().numpy().astype(np.float64)

    def select_slate(self, obs: RecObs) -> np.ndarray:
        return np.argsort(self.score_items(obs))[-self._slate_size:][::-1].astype(
            np.int64
        )

    def train_offline(self, source, *, seed: int = 0) -> dict[str, float]:
        torch.manual_seed(seed)
        self._encoder.train()
        opt = torch.optim.Adam(
            list(self._encoder.parameters()) + list(self._heads.parameters()),
            lr=1e-3,
        )
        last_loss = float("nan")
        for epoch in range(self._epochs):
            for traj in source.iter_trajectories(seed=seed + epoch):
                for step in traj:
                    logits = self._per_position_logits(step.obs)  # (K, N)
                    log_probs = torch.log_softmax(logits, dim=-1)  # (K, N)
                    slate_t = torch.tensor(
                        step.logged_action,
                        dtype=torch.long, device=self._device,
                    )
                    per_pos_logp = log_probs[
                        torch.arange(self._slate_size, device=self._device),
                        slate_t,
                    ]  # (K,)
                    log_pi = per_pos_logp.sum()
                    pi = torch.exp(log_pi)
                    rho = torch.clamp(
                        pi / float(step.propensity), min=0.0, max=self._clip_c,
                    )
                    K = self._slate_size
                    lambda_k = float(K) * (1.0 - pi.item()) ** (K - 1)
                    lambda_k = max(1.0, min(lambda_k, float(K)))
                    reward = float(step.logged_reward)
                    loss = -(rho * lambda_k * log_pi * reward)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    last_loss = float(loss.item())
        return {"loss": last_loss, "epochs": float(self._epochs)}

    def update(self, obs, slate, reward, clicks, next_obs) -> dict[str, float]:
        return {}
