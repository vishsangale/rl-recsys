from __future__ import annotations

import numpy as np
import torch
from torch import nn

from rl_recsys.agents.base import Agent
from rl_recsys.environments.base import RecObs


class _NeuralLinearMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralLinearAgent(Agent):
    """MLP feature extractor + LinUCB head in embedding space (Riquelme 2018)."""

    def __init__(
        self,
        slate_size: int,
        user_dim: int,
        item_dim: int,
        *,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        mlp_epochs: int = 5,
        alpha: float = 1.0,
        device: str = "cuda",
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA required for NeuralLinearAgent; pass device='cpu' to run on CPU"
            )
        self._slate_size = int(slate_size)
        self._user_dim = int(user_dim)
        self._item_dim = int(item_dim)
        self._embedding_dim = int(embedding_dim)
        self._mlp_epochs = int(mlp_epochs)
        self._alpha = float(alpha)
        self._device = torch.device(device)
        self._mlp = _NeuralLinearMLP(
            user_dim + item_dim, hidden_dim, embedding_dim
        ).to(self._device)
        self._a_matrix = np.eye(embedding_dim, dtype=np.float64)
        self._b_vector = np.zeros(embedding_dim, dtype=np.float64)

    def train_offline(self, source, *, seed: int = 0) -> dict[str, float]:
        torch.manual_seed(seed)
        # Materialize the regression dataset (context_features, click) one
        # tuple per (step, slate position).
        rows_x: list[np.ndarray] = []
        rows_y: list[float] = []
        for traj in source.iter_trajectories(seed=seed):
            for step in traj:
                u = step.obs.user_features
                items = step.obs.candidate_features[step.logged_action]
                for item, click in zip(items, step.logged_clicks):
                    rows_x.append(np.concatenate([u, item]))
                    rows_y.append(float(click))
        if not rows_x:
            return {"mlp_loss": float("nan"), "items_seen": 0.0}
        x = torch.tensor(np.stack(rows_x), dtype=torch.float32, device=self._device)
        y = torch.tensor(rows_y, dtype=torch.float32, device=self._device)

        opt = torch.optim.Adam(self._mlp.parameters(), lr=1e-3)
        last_loss = float("nan")
        for _ in range(self._mlp_epochs):
            # Linear head on top of the MLP for click regression. The head
            # is discarded; we only keep the embedding.
            head = nn.Linear(self._embedding_dim, 1).to(self._device)
            head_opt = torch.optim.Adam(head.parameters(), lr=1e-3)
            for batch_start in range(0, len(x), 4096):
                xb = x[batch_start : batch_start + 4096]
                yb = y[batch_start : batch_start + 4096]
                phi = self._mlp(xb)
                pred = head(phi).squeeze(-1)
                loss = nn.functional.mse_loss(pred, yb)
                opt.zero_grad()
                head_opt.zero_grad()
                loss.backward()
                opt.step()
                head_opt.step()
                last_loss = float(loss.item())

        # Recompute (A, b) in embedding space using the trained MLP.
        self._mlp.eval()
        with torch.no_grad():
            phi = self._mlp(x).cpu().numpy().astype(np.float64)
        clicks = y.cpu().numpy().astype(np.float64)
        self._a_matrix = np.eye(self._embedding_dim, dtype=np.float64)
        self._b_vector = np.zeros(self._embedding_dim, dtype=np.float64)
        for p, c in zip(phi, clicks):
            self._a_matrix += np.outer(p, p)
            self._b_vector += c * p
        return {"mlp_loss": last_loss, "items_seen": float(len(x))}

    def _embed(self, obs: RecObs) -> np.ndarray:
        u = np.broadcast_to(
            obs.user_features, (len(obs.candidate_features), self._user_dim)
        )
        x = np.concatenate([u, obs.candidate_features], axis=1)
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32, device=self._device)
            phi = self._mlp(t).cpu().numpy().astype(np.float64)
        return phi

    def score_items(self, obs: RecObs) -> np.ndarray:
        phi = self._embed(obs)
        theta = np.linalg.solve(self._a_matrix, self._b_vector)
        means = phi @ theta
        solved = np.linalg.solve(self._a_matrix, phi.T).T
        variances = np.einsum("ij,ij->i", phi, solved)
        bonuses = self._alpha * np.sqrt(np.clip(variances, 0.0, None))
        return means + bonuses

    def select_slate(self, obs: RecObs) -> np.ndarray:
        n = len(obs.candidate_features)
        if self._slate_size > n:
            raise ValueError(
                f"slate_size={self._slate_size} exceeds num_candidates={n}"
            )
        return np.argsort(self.score_items(obs))[-self._slate_size:][::-1]

    def update(self, obs, slate, reward, clicks, next_obs) -> dict[str, float]:
        # NeuralLinear is a batch-trained agent; per-step update is a no-op.
        return {}
