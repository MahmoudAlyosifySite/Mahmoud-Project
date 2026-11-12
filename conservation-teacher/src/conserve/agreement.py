"""Do two independently trained encoders agree on what is conserved?

A real conserved quantity belongs to the world, so every good encoder of that
world should expose it. An "invariant" that only one encoder shows is more
likely an artefact of that encoder. Given the invariant values two encoders
assign to the same steps, the canonical correlations between the two sets count
what they share: values near 1 are shared invariants, values near 0 belong to
one encoder only.

Motivation: Balestriero and LeCun (arXiv 2610.39302) prove that separately
trained JEPAs agree up to one rotation under their assumptions, and a rotation
keeps a low-degree invariant low-degree. Their analysis does not cover
action-conditioned predictors, so for world models this is a hypothesis to test
(H4 in docs/hypotheses.md), not a guarantee.
"""
from __future__ import annotations

import numpy as np


def canonical_correlations(a: np.ndarray, b: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Canonical correlations between the columns of a and of b, descending, each in [0, 1].

    a: (n, ka) and b: (n, kb) are values on the same n steps. Needs no alignment
    map between the two encoders.
    """
    if len(a) != len(b):
        raise ValueError("a and b must be evaluated on the same steps")
    bases = []
    for x in (a, b):
        x = x - x.mean(0)
        u, s, _ = np.linalg.svd(x, full_matrices=False)
        bases.append(u[:, s > tol * s.max()] if s.size and s.max() > 0 else u[:, :0])
    if bases[0].shape[1] == 0 or bases[1].shape[1] == 0:
        return np.zeros(0)
    return np.clip(np.linalg.svd(bases[0].T @ bases[1], compute_uv=False), 0.0, 1.0)
