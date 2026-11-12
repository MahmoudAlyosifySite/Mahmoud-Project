"""Degree-2 polynomial features of the phase-space latent."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def poly2(x: np.ndarray) -> np.ndarray:
    """All monomials of degree 1 and 2: (n, k) -> (n, k + k(k+1)/2)."""
    i, j = np.triu_indices(x.shape[1])
    return np.concatenate([x, x[:, i] * x[:, j]], axis=1)


@dataclass
class FeatureMap:
    mean: np.ndarray
    std: np.ndarray
    proj: np.ndarray | None  # (d, k): PCA applied after standardising, or None

    @classmethod
    def fit(cls, u: np.ndarray, pca_dim: int | None = None) -> "FeatureMap":
        mean = u.mean(0)
        std = u.std(0)
        std = np.where(std < 1e-9, 1.0, std)
        proj = None
        if pca_dim is not None and pca_dim < u.shape[1]:
            x = (u - mean) / std
            evals, evecs = np.linalg.eigh(np.cov(x, rowvar=False))
            proj = evecs[:, np.argsort(evals)[::-1][:pca_dim]]
        return cls(mean, std, proj)

    def __call__(self, u: np.ndarray) -> np.ndarray:
        x = (u - self.mean) / self.std
        if self.proj is not None:
            x = x @ self.proj
        return poly2(x)
