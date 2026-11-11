"""Find the steps where the latent moves non-smoothly.

Between events (bounces, collisions, anything abrupt) the latent moves smoothly
and a finite difference is a good velocity. At an event the finite difference
straddles two regimes and is wrong. Invariants are therefore read only on clean
steps, and compared across events.
"""
from __future__ import annotations

import numpy as np


def second_diff_norm(z: np.ndarray) -> np.ndarray:
    """(T,): norm of z[t+1] - 2 z[t] + z[t-1]; zero at both ends."""
    a = np.zeros(len(z))
    if len(z) >= 3:
        a[1:-1] = np.linalg.norm(z[2:] - 2.0 * z[1:-1] + z[:-2], axis=1)
    return a


def fit_event_threshold(latents: list[np.ndarray], rel: float = 0.25) -> float:
    """An acceleration above `rel` times the typical step size counts as an event."""
    steps = np.concatenate([np.linalg.norm(z[1:] - z[:-1], axis=1) for z in latents])
    return rel * float(np.median(steps))


def clean_mask(z: np.ndarray, threshold: float, span: int = 1, adapt: float = 5.0) -> np.ndarray:
    """(T-1,) bool, aligned with `phase_space(z, span)`: True where the difference is trustworthy.

    Step t is clean when no event falls between t-span and t. The threshold rises
    with the clip's own median acceleration, so a latent that curves everywhere
    (a nonlinear encoder, a fast clip) is not flagged at every step.
    """
    a = second_diff_norm(z)
    inner = a[1:-1]
    level = max(threshold, adapt * float(np.median(inner))) if len(inner) else threshold
    csum = np.concatenate([[0], np.cumsum(a > level)])
    t = np.arange(1, len(z))
    events_in_span = csum[t + 1] - csum[np.maximum(t - span, 0)]
    return (events_in_span == 0) & (t >= span)
