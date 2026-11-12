"""Learn what never changes.

An invariant is a function I of the phase-space latent whose value is the same
at two clean steps of one clip however far apart they are, while still differing
between clips. With I = w . phi(u) this is a generalised eigenproblem:

    minimise   E[(w . (phi(u_t2) - phi(u_t1)))^2]     over long-lag pairs in one clip
    subject to Var[w . phi(u)] = 1                    over all clips

The smallest eigenvalues ("slowness") are the invariants. Long lags that span
events are what separate a conserved quantity from one that is merely slow, or
constant only between bounces. No labels and no impossible examples are used.

This learner is not new and must not be presented as new:
- Bao (arXiv 2608.23526) finds an energy-like invariant in a frozen DreamerV3
  latent with the same kind of criterion: within-trajectory variance over total
  variance of a polynomial of the top principal components, solved as a
  generalised eigenproblem. He uses it to correct rollouts, on a pendulum.
- At lag 1 it is Slow Feature Analysis (Wiskott and Sejnowski, 2002).
- It is a search for Koopman eigenfunctions with eigenvalue one in a polynomial
  dictionary (Kaiser, Kutz and Brunton, arXiv 1811.00961).
What this repository adds is the handling of events (contacts, several objects)
and the use of the invariants as a detector. See docs/novelty.md.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .features import FeatureMap


@dataclass
class InvariantHead:
    feat: FeatureMap
    mean: np.ndarray  # (D,)
    W: np.ndarray  # (D, K), each column has unit variance on the fitting set
    slowness: np.ndarray  # (K,), 0 = perfectly conserved, about 2 = unrelated at long lag
    slowness_val: np.ndarray | None = None

    def __call__(self, u: np.ndarray) -> np.ndarray:
        return (self.feat(u) - self.mean) @ self.W

    @property
    def k(self) -> int:
        return self.W.shape[1]

    def select(self, keep: np.ndarray) -> "InvariantHead":
        val = None if self.slowness_val is None else self.slowness_val[keep]
        return replace(self, W=self.W[:, keep], slowness=self.slowness[keep], slowness_val=val)


def _pairs(clean: np.ndarray, lag_range: tuple[int, int], n: int, rng: np.random.Generator):
    idx = np.flatnonzero(clean)
    if len(idx) < 2:
        return np.zeros(0, int), np.zeros(0, int)
    t1 = rng.choice(idx, size=n)
    t2 = t1 + rng.integers(lag_range[0], lag_range[1] + 1, size=n)
    ok = t2 < len(clean)
    ok[ok] &= clean[t2[ok]]
    return t1[ok], t2[ok]


def fit_invariants(
    us: list[np.ndarray],
    cleans: list[np.ndarray],
    feat: FeatureMap,
    rng: np.random.Generator,
    lag_range: tuple[int, int] = (30, 90),
    pairs_per_clip: int = 200,
    max_k: int = 16,
    tol: float = 1e-6,
) -> InvariantHead:
    phis = [feat(u) for u in us]
    stacked = np.concatenate([p[c] for p, c in zip(phis, cleans)])
    mean = stacked.mean(0)
    cov = np.cov(stacked - mean, rowvar=False)
    diffs = []
    for p, c in zip(phis, cleans):
        t1, t2 = _pairs(c, lag_range, pairs_per_clip, rng)
        diffs.append(p[t2] - p[t1])
    d = np.concatenate(diffs)
    if len(d) == 0:
        raise ValueError("no long-lag pairs: clips are shorter than the lag range")
    cov_d = d.T @ d / len(d)
    evals, evecs = np.linalg.eigh(cov)
    keep = evals > tol * evals.max()
    whiten = evecs[:, keep] / np.sqrt(evals[keep])
    mu, v = np.linalg.eigh(whiten.T @ cov_d @ whiten)
    k = min(max_k, len(mu))
    return InvariantHead(feat, mean, whiten @ v[:, :k], mu[:k])


def long_lag_slowness(
    head: InvariantHead,
    us: list[np.ndarray],
    cleans: list[np.ndarray],
    rng: np.random.Generator,
    lag_range: tuple[int, int] = (30, 90),
    pairs_per_clip: int = 200,
) -> np.ndarray:
    """Mean squared long-lag change of every invariant on fresh lawful clips."""
    total = np.zeros(head.k)
    count = 0
    for u, c in zip(us, cleans):
        t1, t2 = _pairs(c, lag_range, pairs_per_clip, rng)
        if len(t1) == 0:
            continue
        values = head(u)
        total += ((values[t2] - values[t1]) ** 2).sum(0)
        count += len(t1)
    if count == 0:
        raise ValueError("no long-lag pairs in the validation clips")
    return total / count
