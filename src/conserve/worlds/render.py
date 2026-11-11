"""Render billiards states to small grayscale frames."""
from __future__ import annotations

import numpy as np

from .billiards import BilliardsConfig, Trajectory


def render(traj: Trajectory, cfg: BilliardsConfig, size: int = 48) -> np.ndarray:
    """Return frames of shape (T, size, size), uint8. All balls look the same; edges are one pixel soft."""
    centres = (np.arange(size) + 0.5) / size
    dx = centres[None, None, None, :] - traj.pos[:, :, 0, None, None]  # (T, N, 1, W)
    dy = centres[None, None, :, None] - traj.pos[:, :, 1, None, None]  # (T, N, H, 1)
    disc = np.clip((cfg.radius - np.sqrt(dx**2 + dy**2)) * size + 0.5, 0.0, 1.0)
    disc = disc * traj.alive[:, :, None, None]
    return (disc.max(1) * 255).astype(np.uint8)
