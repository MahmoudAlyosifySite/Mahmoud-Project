"""AUROC and a bootstrap interval."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Chance that a random positive scores above a random negative; ties count half."""
    positive, negative = np.asarray(positive, float), np.asarray(negative, float)
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("auroc needs at least one positive and one negative")
    ranks = rankdata(np.concatenate([positive, negative]))
    n_pos, n_neg = len(positive), len(negative)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_ci(positive, negative, rng: np.random.Generator, n: int = 1000, level: float = 0.95):
    positive, negative = np.asarray(positive, float), np.asarray(negative, float)
    stats = [
        auroc(rng.choice(positive, len(positive)), rng.choice(negative, len(negative)))
        for _ in range(n)
    ]
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)
