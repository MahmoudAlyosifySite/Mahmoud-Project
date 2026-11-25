"""Shared-covariance Gaussian model (SPEC 9.2).

Standardise the features, take class means, shrink the pooled residual
covariance with Ledoit-Wolf (floor ``shrinkageMin``), factor it with a
Cholesky decomposition and measure squared Mahalanobis distances by forward
substitution. The leave-one-out distances give the absolute threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

from ._numerics import mean, quantile_linear, seq_sum


@dataclass(frozen=True)
class ModelConfig:
    """Defaults of SPEC 9.2."""

    shrinkage_min: float = 0.1
    loo_quantile: float = 0.95
    abs_factor: float = 1.5

    def to_json(self) -> dict:
        return {"shrinkageMin": self.shrinkage_min, "looQuantile": self.loo_quantile, "absFactor": self.abs_factor}


DEFAULT_CONFIG = ModelConfig()

SCALE_FLOOR = 1e-6
JITTER_START = 1e-6
JITTER_RETRIES = 5


def ledoit_wolf(r: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Ledoit-Wolf statistics of the residual matrix ``R`` (n x d, assumed centred).

    Returns ``(S_emp, mu, lambda_LW)`` exactly as in SPEC 9.2 step 4 (before
    the ``shrinkageMin`` floor and the clamp).
    """
    r = np.asarray(r, dtype=np.float64)
    n, d = r.shape
    s_emp = (r.T @ r) / n
    r2 = r * r
    trace = r2.sum(axis=0) / n
    mu = float(trace.sum() / d)
    beta_ = float(np.sum(r2.T @ r2))
    delta_ = float(np.sum((r.T @ r) ** 2) / (n * n))
    beta = (beta_ / n - delta_) / (d * n)
    delta = (delta_ - 2.0 * mu * float(trace.sum()) + d * (mu * mu)) / d
    beta = min(beta, delta)
    lam = 0.0 if delta == 0.0 else beta / delta
    return s_emp, mu, lam


def cholesky(s: np.ndarray) -> np.ndarray | None:
    """Lower-triangular ``L`` with ``S = L L^T``; ``None`` on a non-positive pivot.

    Row-by-row (Cholesky-Banachiewicz) with left-to-right sums.
    """
    s = np.asarray(s, dtype=np.float64)
    d = s.shape[0]
    low = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(i + 1):
            acc = float(s[i, j])
            for k in range(j):
                acc = acc - float(low[i, k]) * float(low[j, k])
            if i == j:
                if not acc > 0.0:
                    return None
                low[i, i] = math.sqrt(acc)
            else:
                low[i, j] = acc / float(low[j, j])
    return low


def cholesky_with_jitter(s: np.ndarray, mu: float) -> tuple[np.ndarray | None, float]:
    """SPEC 9.2 step 5: try ``S``, then ``S + j I`` with ``j = 1e-6*mu, 1e-5*mu, ...``
    (5 retries, jitter multiplied by 10 each time, never accumulated).

    Returns ``(L, jitter_used)``; ``L`` is ``None`` if every attempt failed.
    """
    low = cholesky(s)
    if low is not None:
        return low, 0.0
    jitter = JITTER_START * mu
    d = s.shape[0]
    for _ in range(JITTER_RETRIES):
        low = cholesky(s + jitter * np.eye(d))
        if low is not None:
            return low, jitter
        jitter *= 10.0
    return None, jitter


def forward_sq_norm(low: np.ndarray, v: Sequence[float]) -> float:
    """``||L^{-1} v||^2`` by forward substitution (left-to-right sums)."""
    d = low.shape[0]
    y = [0.0] * d
    for i in range(d):
        acc = float(v[i])
        for k in range(i):
            acc = acc - float(low[i, k]) * y[k]
        y[i] = acc / float(low[i, i])
    return seq_sum(np.asarray(y) * np.asarray(y))


class GaussianModel:
    """SPEC 9.2. Build with :meth:`fit`.

    Attributes: ``mean``, ``scale`` (sigma after the floor), ``class_means``
    (K x d, in z units), ``class_counts``, ``shrinkage`` (lambda used),
    ``shrinkage_lw`` (lambda_LW before the floor; ``nan`` in the degenerate
    case), ``mu``, ``cov`` (S), ``chol`` (L), ``jitter``, ``abs_threshold``,
    ``loo`` (leave-one-out distances in sample order), ``degenerate``.
    """

    def __init__(self) -> None:
        self.mean = np.zeros(0)
        self.scale = np.zeros(0)
        self.class_means = np.zeros((0, 0))
        self.class_counts: list[int] = []
        self.shrinkage = 1.0
        self.shrinkage_lw = float("nan")
        self.mu = 0.0
        self.cov = np.zeros((0, 0))
        self.chol = np.zeros((0, 0))
        self.jitter = 0.0
        self.abs_threshold = 0.0
        self.loo: list[float] = []
        self.loo_index: list[int] = []
        self.degenerate = False
        self.config = DEFAULT_CONFIG

    @property
    def n_classes(self) -> int:
        return len(self.class_counts)

    @classmethod
    def fit(cls, X, labels: Sequence[int], n_classes: int, config: ModelConfig | None = None) -> "GaussianModel | None":
        """Fit on ``X`` (n x d) with class indices ``labels`` in ``0..n_classes-1``.

        Returns ``None`` if there are no samples. Classes without samples get
        a zero mean vector (they never occur through the Identifier).
        """
        cfg = config or DEFAULT_CONFIG
        x = np.asarray(X, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] == 0:
            return None
        n, d = x.shape
        lab = [int(v) for v in labels]
        if len(lab) != n:
            raise ValueError("labels and X differ in length")
        m = cls()
        m.config = cfg
        # 1. standardisation (population std, floored)
        mu_vec = np.array([mean(x[:, j]) for j in range(d)])
        sig = np.array([math.sqrt(mean((x[:, j] - mu_vec[j]) * (x[:, j] - mu_vec[j]))) for j in range(d)])
        sig = np.maximum(sig, SCALE_FLOOR)
        z = (x - mu_vec) / sig
        # 2. class means
        counts = [0] * n_classes
        for k in lab:
            counts[k] += 1
        cmeans = np.zeros((n_classes, d))
        for k in range(n_classes):
            rows = [i for i in range(n) if lab[i] == k]
            if rows:
                cmeans[k] = np.array([mean(z[rows, j]) for j in range(d)])
        # 3. residuals
        resid = z - cmeans[lab]
        # 4. Ledoit-Wolf
        k_present = sum(1 for c in counts if c > 0)
        s_emp, mu, lam_lw = ledoit_wolf(resid)
        if n - k_present < 2 or mu <= 1e-12:
            cov = np.eye(d)
            lam = 1.0
            m.degenerate = True
            m.shrinkage_lw = float("nan")
        else:
            lam = min(max(max(lam_lw, cfg.shrinkage_min), 0.0), 1.0)
            cov = (1.0 - lam) * s_emp + lam * mu * np.eye(d)
            m.shrinkage_lw = lam_lw
        # 5. Cholesky with jitter
        low, jitter = cholesky_with_jitter(cov, mu if mu > 0 else 1.0)
        if low is None:
            # Not specified by SPEC 9.2: fall back to the degenerate model.
            cov = np.eye(d)
            lam = 1.0
            low = np.eye(d)
            m.degenerate = True
        m.mean = mu_vec
        m.scale = sig
        m.class_means = cmeans
        m.class_counts = counts
        m.shrinkage = float(lam)
        m.mu = float(mu)
        m.cov = cov
        m.chol = low
        m.jitter = float(jitter)
        # 7. leave-one-out threshold
        loo = []
        loo_index = []
        for i in range(n):
            k = lab[i]
            nk = counts[k]
            if nk < 2:
                continue
            m_prime = (nk * cmeans[k] - z[i]) / (nk - 1)
            loo.append(forward_sq_norm(low, z[i] - m_prime))
            loo_index.append(i)
        m.loo = loo
        m.loo_index = loo_index
        if loo:
            m.abs_threshold = cfg.abs_factor * quantile_linear(loo, cfg.loo_quantile)
        else:
            m.abs_threshold = cfg.abs_factor * 2.0 * d
        return m

    def standardize(self, x) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.scale

    def distances(self, x) -> list[float]:
        """Squared Mahalanobis distance ``D^2_k(x)`` for every class (SPEC 9.2 step 6)."""
        z = self.standardize(x)
        return [forward_sq_norm(self.chol, z - self.class_means[k]) for k in range(self.n_classes)]
