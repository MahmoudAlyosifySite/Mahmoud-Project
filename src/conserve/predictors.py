"""A learned next-latent predictor, the source of prediction-error surprise.

It predicts z[t+1] as the straight-line extrapolation z[t] + (z[t] - z[t-1])
plus a small network's correction, so free flight is exact and the network only
has to learn what happens at events.
"""
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor


class ResidualMLP:
    def __init__(self, hidden: tuple[int, ...] = (128, 128), max_iter: int = 150, seed: int = 0):
        self.net = MLPRegressor(
            hidden_layer_sizes=hidden, max_iter=max_iter, early_stopping=True,
            n_iter_no_change=10, random_state=seed,
        )
        self.x_mean = self.x_std = None
        self.y_scale = 1.0

    @staticmethod
    def _split(u: np.ndarray, z: np.ndarray):
        # u[k] describes time k+1; it predicts z[k+2].
        linear = z[1:-1] + (z[1:-1] - z[:-2])
        return u[:-1], linear, z[2:]

    def fit(self, us: list[np.ndarray], zs: list[np.ndarray]) -> "ResidualMLP":
        xs, residuals = [], []
        for u, z in zip(us, zs):
            x, linear, target = self._split(u, z)
            xs.append(x)
            residuals.append(target - linear)
        x, y = np.concatenate(xs), np.concatenate(residuals)
        self.x_mean = x.mean(0)
        std = x.std(0)
        self.x_std = np.where(std < 1e-9, 1.0, std)
        self.y_scale = float(y.std()) or 1.0
        self.net.fit((x - self.x_mean) / self.x_std, y / self.y_scale)
        return self

    def errors(self, u: np.ndarray, z: np.ndarray) -> np.ndarray:
        """(T-2,): squared error of the prediction of z[2:]."""
        x, linear, target = self._split(u, z)
        pred = linear + self.net.predict((x - self.x_mean) / self.x_std) * self.y_scale
        return ((target - pred) ** 2).sum(1)
