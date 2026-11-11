"""Per-clip scores. Higher means "this clip looks impossible"."""
from __future__ import annotations

import numpy as np

from .worlds.billiards import BilliardsConfig, Trajectory, step


def pe_scores(errors: np.ndarray, floor: float = 0.0) -> dict[str, float]:
    """Three readings of a clip's per-step prediction error.

    pe_max   the largest error
    pe_mean  the average error
    pe_rel   the largest error relative to the clip's own 90th percentile, which
             forgives a clip that is hard to predict everywhere
    """
    return {
        "pe_max": float(errors.max()),
        "pe_mean": float(errors.mean()),
        "pe_rel": float(errors.max() / (np.quantile(errors, 0.9) + floor + 1e-300)),
    }


def level_shift(
    values: np.ndarray,
    clean: np.ndarray,
    window: int = 12,
    min_clean: int = 4,
    eps: float = 0.05,
) -> dict[str, float]:
    """Largest lasting change in any invariant, in robust standard units.

    For every split point the median over the clean steps before it is compared
    with the median after it. A spike at an event does not move a median; a
    conserved quantity that takes a new value does. `eps` is the floor on the
    scale, in units of the invariant's spread across clips.

    Returns
      inv      the largest shift over all split points and invariants
      inv_rel  that shift relative to the clip's own median shift, which forgives
               a clip whose invariants wobble at every event
      inv_t    the split point of the largest shift (-1 if none could be tested)
      inv_k    the invariant that moved most
    """
    out = {"inv": 0.0, "inv_rel": 0.0, "inv_t": -1.0, "inv_k": -1.0}
    if values.shape[1] == 0:
        return out
    stats = []
    for t in range(window, len(values) - window + 1):
        before, after = clean[t - window : t], clean[t : t + window]
        if before.sum() < min_clean or after.sum() < min_clean:
            continue
        vb, va = values[t - window : t][before], values[t : t + window][after]
        mb, ma = np.median(vb, 0), np.median(va, 0)
        spread = 1.4826 * np.median(np.abs(np.concatenate([vb - mb, va - ma])), 0)
        stat = np.abs(ma - mb) / (spread + eps)
        k = int(stat.argmax())
        stats.append(float(stat[k]))
        if stat[k] > out["inv"]:
            out.update(inv=float(stat[k]), inv_t=float(t), inv_k=float(k))
    if stats:
        out["inv_rel"] = out["inv"] / (float(np.median(stats)) + eps)
    return out


def oracle_errors(traj: Trajectory, cfg: BilliardsConfig) -> np.ndarray:
    """(T-1,): error of the true simulator predicting the full state one step ahead.

    This is the best any predictor can do. It is exactly zero on lawful steps of
    a deterministic world. On rough walls it is not, because the simulator can
    only predict the mean bounce.
    """
    errors = np.zeros(len(traj.pos) - 1)
    for t in range(len(errors)):
        alive = traj.alive[t]
        pos, vel, _ = step(traj.pos[t], traj.vel[t], alive, cfg, rng=None)
        mask = alive[:, None]
        errors[t] = (
            ((pos * mask - traj.pos[t + 1]) ** 2).sum()
            + ((vel * mask - traj.vel[t + 1]) ** 2).sum()
            + float((alive != traj.alive[t + 1]).sum())
        )
    return errors
