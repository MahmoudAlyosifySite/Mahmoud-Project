"""A tiny pixel JEPA world model, LeWorldModel in spirit and much smaller.

One loss, trained end to end with no stop-gradient, no EMA target and no
pretrained encoder: predict the next frame's embedding from the last two, plus a
Gaussianity regulariser against collapse. The world here is passive, so there is
no action input.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .sigreg import sigreg


class Encoder(nn.Module):
    def __init__(self, dim: int = 64, size: int = 48):
        super().__init__()
        if size % 8:
            raise ValueError("frame size must be a multiple of 8")
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Flatten(),
            nn.Linear(128 * (size // 8) ** 2, 512), nn.GELU(),
            nn.Linear(512, dim),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:  # (B, 1, H, W) in [0, 1]
        return self.net(frames)


class Predictor(nn.Module):
    def __init__(self, dim: int = 64, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, z: torch.Tensor, z_prev: torch.Tensor) -> torch.Tensor:
        return z + self.net(torch.cat([z, z - z_prev], dim=-1))


class TinyJEPA(nn.Module):
    def __init__(self, dim: int = 64, size: int = 48):
        super().__init__()
        self.encoder = Encoder(dim, size)
        self.predictor = Predictor(dim)

    def loss(self, triples: torch.Tensor, lam: float):
        """triples: (B, 3, H, W) uint8 frames at t-1, t, t+1. Returns (loss, prediction part, regulariser part)."""
        b = triples.shape[0]
        z = self.encoder(triples.reshape(b * 3, 1, *triples.shape[2:]).float() / 255.0).view(b, 3, -1)
        prediction = (self.predictor(z[:, 1], z[:, 0]) - z[:, 2]).pow(2).sum(-1).mean()
        regulariser = sigreg(z.transpose(0, 1))
        return prediction + lam * regulariser, prediction.detach(), regulariser.detach()

    @torch.no_grad()
    def encode(self, frames: np.ndarray, device, batch: int = 2048) -> np.ndarray:
        """frames: (T, H, W) uint8 -> latents (T, dim) float64."""
        self.eval()
        out = []
        for i in range(0, len(frames), batch):
            x = torch.from_numpy(frames[i : i + batch]).to(device).float().unsqueeze(1) / 255.0
            out.append(self.encoder(x).double().cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def prediction_errors(self, z: np.ndarray, device) -> np.ndarray:
        """(T-2,): squared error of the model's own prediction of z[2:], aligned with ResidualMLP.errors."""
        self.eval()
        zt = torch.from_numpy(z).float().to(device)
        pred = self.predictor(zt[1:-1], zt[:-2])
        return (pred - zt[2:]).pow(2).sum(-1).double().cpu().numpy()


def train(model: TinyJEPA, frames: torch.Tensor, steps: int, batch: int, lam: float, lr: float, log_every: int = 500):
    """frames: (N, T, H, W) uint8 on the training device. Returns the loss log."""
    n, t = frames.shape[:2]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    log = []
    model.train()
    for step in range(1, steps + 1):
        clip = torch.randint(0, n, (batch,), device=frames.device)
        mid = torch.randint(1, t - 1, (batch,), device=frames.device)
        triples = torch.stack([frames[clip, mid - 1], frames[clip, mid], frames[clip, mid + 1]], dim=1)
        loss, prediction, regulariser = model.loss(triples, lam)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps:
            row = {"step": step, "loss": loss.item(), "prediction": prediction.item(), "sigreg": regulariser.item()}
            log.append(row)
            print(f"  step {step}/{steps}  loss {row['loss']:.4f}  pred {row['prediction']:.4f}  sigreg {row['sigreg']:.4f}", flush=True)
    return log
