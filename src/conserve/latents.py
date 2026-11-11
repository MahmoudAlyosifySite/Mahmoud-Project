"""Latent views of a trajectory.

`frame_latent` is the idealised frame encoder: one vector per frame holding what
a perfect encoder of a single frame could know (which balls are present and
where). It holds no velocities. `Mixer` bends that latent with a random
nonlinearity and noise, to imitate an encoder whose latents are not linearly
identifiable.
"""
from __future__ import annotations

import numpy as np

from .worlds.billiards import Trajectory


def frame_latent(traj: Trajectory) -> np.ndarray:
    """(T, 3N): presence flag, x and y for every slot. Absent balls are all zeros."""
    return np.concatenate([traj.alive.astype(float), traj.pos[..., 0], traj.pos[..., 1]], axis=1)


def phase_space(z: np.ndarray, span: int = 1) -> np.ndarray:
    """(T-1, 2d): the latent at t and its change per step since t-span, for t = 1..T-1.

    A longer span averages frame-to-frame noise out of the velocity estimate, at
    the price of needing `span` event-free steps. Rows with t < span use the
    shorter baseline that exists and are marked unclean by `events.clean_mask`.
    """
    t = np.arange(1, len(z))
    back = np.maximum(t - span, 0)
    return np.concatenate([z[t], (z[t] - z[back]) / (t - back)[:, None]], axis=1)


class Mixer:
    """z -> A s + alpha * tanh(B s) + noise, with s the standardised latent.

    alpha = 0 is a rotation, which keeps every low-degree invariant low-degree.
    Larger alpha moves the latent away from linear identifiability.
    """

    def __init__(self, dim: int, alpha: float = 0.0, noise: float = 0.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.A, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
        self.B = rng.normal(size=(dim, dim)) / np.sqrt(dim)
        self.alpha = alpha
        self.noise = noise
        self.mean = np.zeros(dim)
        self.std = np.ones(dim)

    def fit(self, latents: list[np.ndarray]) -> "Mixer":
        stacked = np.concatenate(latents)
        self.mean = stacked.mean(0)
        std = stacked.std(0)
        self.std = np.where(std < 1e-9, 1.0, std)
        return self

    def __call__(self, z: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        s = (z - self.mean) / self.std
        out = s @ self.A.T + self.alpha * np.tanh(s @ self.B.T)
        if self.noise > 0:
            if rng is None:
                raise ValueError("a noisy mixer needs an rng")
            out = out + self.noise * rng.normal(size=out.shape)
        return out
