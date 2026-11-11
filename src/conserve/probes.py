"""Linear probes that check whether an encoder's latent carries the frame at all.

Diagnostic only, with one declared use: `probe_r2` gates Stage C (docs/hypotheses.md,
Changes, 2026-11-11). Nothing here enters the fitting of invariants or the choice
among them.

The gate's targets are functions of a single frame that do not depend on which
ball is which: the number of balls present, and a coarse occupancy map. The older
slot-indexed target (`frame_latent`) needs ball identity, which identical-looking
balls in one frame do not carry; it is kept as `probe_r2_slots` for comparison.
"""
from __future__ import annotations

import numpy as np

from .latents import frame_latent
from .worlds.billiards import BilliardsConfig, Trajectory
from .worlds.render import render

OCC_RENDER = 48  # occupancy targets are always rendered at this size
OCC_CELLS = 12  # into OCC_CELLS x OCC_CELLS blocks


def linear_probe_r2(x_train, y_train, x_test, y_test, ridge: float = 1e-3) -> float:
    """Mean R2 over target columns of a ridge regression from x to y, fitted on train, scored on test."""
    y_train = np.asarray(y_train, dtype=float).reshape(len(y_train), -1)
    y_test = np.asarray(y_test, dtype=float).reshape(len(y_test), -1)
    mean, std = x_train.mean(0), x_train.std(0)
    std = np.where(std < 1e-9, 1.0, std)
    a = np.concatenate([(x_train - mean) / std, np.ones((len(x_train), 1))], axis=1)
    b = np.concatenate([(x_test - mean) / std, np.ones((len(x_test), 1))], axis=1)
    coef = np.linalg.solve(a.T @ a + ridge * len(a) * np.eye(a.shape[1]), a.T @ y_train)
    resid = y_test - b @ coef
    var = y_test.var(0)
    keep = var > 1e-12
    return float((1.0 - resid.var(0)[keep] / var[keep]).mean())


def ball_count(traj: Trajectory) -> np.ndarray:
    """(T, 1): number of balls present in each frame."""
    return traj.alive.sum(1, keepdims=True).astype(float)


def occupancy(traj: Trajectory, cfg: BilliardsConfig) -> np.ndarray:
    """(T, OCC_CELLS**2): the frame rendered at OCC_RENDER pixels, averaged over square blocks, in [0, 1]."""
    frames = render(traj, cfg, OCC_RENDER).astype(float) / 255.0
    k = OCC_RENDER // OCC_CELLS
    t = len(frames)
    return frames.reshape(t, OCC_CELLS, k, OCC_CELLS, k).mean((2, 4)).reshape(t, -1)


def encoder_probes(z_train, clips_train, z_val, clips_val, cfg: BilliardsConfig) -> dict:
    """The gate value `probe_r2` and its parts, plus the old slot-indexed probe for comparison.

    z_*: lists of (T, d) latents, one per clip, aligned with clips_*.
    """
    zt, zv = np.concatenate(z_train), np.concatenate(z_val)

    def stack(fn, clips):
        return np.concatenate([fn(c) for c in clips])

    count = linear_probe_r2(zt, stack(ball_count, clips_train), zv, stack(ball_count, clips_val))
    occ = linear_probe_r2(
        zt, stack(lambda c: occupancy(c, cfg), clips_train), zv, stack(lambda c: occupancy(c, cfg), clips_val)
    )
    slots = linear_probe_r2(zt, stack(frame_latent, clips_train), zv, stack(frame_latent, clips_val))
    return {"probe_r2": 0.5 * (count + occ), "probe_r2_count": count, "probe_r2_occupancy": occ, "probe_r2_slots": slots}
