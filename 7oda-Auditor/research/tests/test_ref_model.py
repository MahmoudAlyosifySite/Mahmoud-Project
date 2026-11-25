"""SPEC 9.2: the shared-covariance Gaussian model."""

import math

import numpy as np
import pytest
from sklearn.covariance import ledoit_wolf_shrinkage

from auditor_ref._numerics import median, quantile_linear, seq_sum
from auditor_ref.model import GaussianModel, ModelConfig, cholesky, cholesky_with_jitter, forward_sq_norm, ledoit_wolf


def _data(seed=0, k=4, per=8, d=36, sep=2.0):
    rs = np.random.RandomState(seed)
    centers = rs.normal(0, sep, size=(k, d))
    scales = np.exp(rs.normal(0, 1.0, size=d))
    X, y = [], []
    for c in range(k):
        for _ in range(per):
            X.append((centers[c] + rs.normal(0, 1, size=d)) * scales + 5.0)
            y.append(c)
    return np.array(X), y


def _standardize(X):
    mu = X.mean(axis=0)
    sd = np.maximum(X.std(axis=0), 1e-6)
    return (X - mu) / sd, mu, sd


@pytest.mark.parametrize("n,d,seed", [(30, 36, 0), (12, 36, 1), (200, 10, 2), (5, 3, 3)])
def test_ledoit_wolf_matches_sklearn(n, d, seed):
    R = np.random.RandomState(seed).normal(size=(n, d)) * np.linspace(0.5, 2.0, d)
    s_emp, mu, lam = ledoit_wolf(R)
    assert lam == pytest.approx(ledoit_wolf_shrinkage(R, assume_centered=True), rel=1e-10, abs=1e-14)
    np.testing.assert_allclose(s_emp, R.T @ R / n, rtol=1e-12)
    assert mu == pytest.approx(np.trace(R.T @ R / n) / d, rel=1e-12)


def test_fit_standardisation_and_class_means():
    X, y = _data()
    m = GaussianModel.fit(X, y, 4)
    z, mu, sd = _standardize(X)
    np.testing.assert_allclose(m.mean, mu, rtol=1e-12)
    np.testing.assert_allclose(m.scale, sd, rtol=1e-12)
    for k in range(4):
        np.testing.assert_allclose(m.class_means[k], z[np.array(y) == k].mean(axis=0), atol=1e-12)
    assert m.class_counts == [8] * 4


def test_shrinkage_floor_and_covariance():
    X, y = _data(seed=3, per=20, d=8)
    m = GaussianModel.fit(X, y, 4)
    z, _, _ = _standardize(X)
    R = z - m.class_means[y]
    lam_lw = ledoit_wolf_shrinkage(R, assume_centered=True)
    assert m.shrinkage_lw == pytest.approx(lam_lw, rel=1e-9)
    assert m.shrinkage == pytest.approx(min(max(lam_lw, 0.1), 1.0), rel=1e-9)
    mu = np.trace(R.T @ R / len(X)) / 8
    S = (1 - m.shrinkage) * (R.T @ R / len(X)) + m.shrinkage * mu * np.eye(8)
    np.testing.assert_allclose(m.cov, S, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(m.chol @ m.chol.T, S, rtol=1e-10, atol=1e-12)


def test_distances_brute_force():
    X, y = _data(seed=5)
    m = GaussianModel.fit(X, y, 4)
    rs = np.random.RandomState(9)
    for _ in range(5):
        x = X[rs.randint(len(X))] + rs.normal(0, 0.5, size=X.shape[1]) * m.scale
        zq = (x - m.mean) / m.scale
        d = m.distances(x)
        for k in range(4):
            v = zq - m.class_means[k]
            assert d[k] == pytest.approx(float(v @ np.linalg.solve(m.cov, v)), rel=1e-9)


def test_loo_brute_force_and_threshold():
    X, y = _data(seed=6, per=6)
    cfg = ModelConfig()
    m = GaussianModel.fit(X, y, 4, cfg)
    z = (X - m.mean) / m.scale
    yy = np.array(y)
    ref = []
    for i in range(len(X)):
        k = yy[i]
        others = z[(yy == k) & (np.arange(len(X)) != i)]
        v = z[i] - others.mean(axis=0)
        ref.append(float(v @ np.linalg.solve(m.cov, v)))
    np.testing.assert_allclose(m.loo, ref, rtol=1e-9)
    assert m.abs_threshold == pytest.approx(1.5 * np.quantile(ref, 0.95), rel=1e-9)


def test_quantile_and_median_match_numpy():
    rs = np.random.RandomState(1)
    for n in (1, 2, 3, 7, 20, 101):
        v = rs.normal(size=n)
        for q in (0.0, 0.25, 0.5, 0.95, 1.0):
            assert quantile_linear(v, q) == np.quantile(v, q)
        assert median(v) == np.median(v)
    acc = 0.0
    for v in [0.1] * 10:
        acc = acc + v
    # plain left-to-right sum (Python 3.12's built-in sum() is compensated)
    assert seq_sum([0.1] * 10) == acc == 0.9999999999999999


def test_cholesky_and_jitter_path():
    A = np.array([[4.0, 2.0], [2.0, 3.0]])
    np.testing.assert_allclose(cholesky(A), np.linalg.cholesky(A), rtol=1e-15)
    # singular PSD matrix: plain Cholesky fails, the first jitter (1e-6 * mu) succeeds
    v = np.array([1.0, 2.0, 3.0])
    S = np.outer(v, v)
    mu = np.trace(S) / 3
    assert cholesky(S) is None
    L, jitter = cholesky_with_jitter(S, mu)
    assert L is not None and jitter == pytest.approx(1e-6 * mu)
    np.testing.assert_allclose(L @ L.T, S + jitter * np.eye(3), rtol=1e-9, atol=1e-12)
    # strongly indefinite: every retry fails (jitter 1e-6..1e-2 * mu, never accumulated)
    L, jitter = cholesky_with_jitter(np.diag([1.0, -1.0]), 1.0)
    assert L is None and jitter == pytest.approx(1e-1)
    # needs the 3rd retry: smallest eigenvalue -5e-5 * mu
    L, jitter = cholesky_with_jitter(np.diag([1.0, -5e-5]), 1.0)
    assert L is not None and jitter == pytest.approx(1e-4)


def test_forward_substitution():
    L = np.array([[2.0, 0.0], [1.0, 3.0]])
    v = np.array([4.0, 5.0])
    y = np.linalg.solve(L, v)
    assert forward_sq_norm(L, v) == pytest.approx(float(y @ y), rel=1e-15)


def test_degenerate_single_sample_per_class():
    X = np.array([[1.0, 2.0, 3.0], [2.0, 0.0, 1.0], [0.5, 1.0, 4.0]])
    m = GaussianModel.fit(X, [0, 1, 2], 3)
    assert m.degenerate and m.shrinkage == 1.0
    np.testing.assert_array_equal(m.cov, np.eye(3))
    assert m.loo == [] and m.abs_threshold == 1.5 * 2 * 3
    z = (X[0] - m.mean) / m.scale
    assert m.distances(X[0])[0] == pytest.approx(0.0, abs=1e-24)
    assert m.distances(X[0])[1] == pytest.approx(float(np.sum((z - m.class_means[1]) ** 2)), rel=1e-12)


def test_degenerate_n_minus_k_below_two_and_zero_mu():
    X = np.array([[1.0, 2.0], [1.5, 2.5], [3.0, 1.0]])
    m = GaussianModel.fit(X, [0, 0, 1], 2)        # n - K = 1 < 2
    assert m.degenerate and m.shrinkage == 1.0
    assert len(m.loo) == 2                       # class 0 has 2 samples
    X = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 1.0], [3.0, 1.0], [5.0, 0.0], [5.0, 0.0]])
    m = GaussianModel.fit(X, [0, 0, 1, 1, 2, 2], 3)  # residuals all zero -> mu = 0
    assert m.degenerate and m.mu == 0.0 and np.array_equal(m.cov, np.eye(2))


def test_single_class_and_constant_feature():
    rs = np.random.RandomState(4)
    X = rs.normal(size=(10, 5))
    X[:, 2] = 7.0
    m = GaussianModel.fit(X, [0] * 10, 1)
    assert m.scale[2] == 1e-6 and not m.degenerate
    assert len(m.loo) == 10 and len(m.distances(X[0])) == 1
    assert GaussianModel.fit(np.zeros((0, 5)), [], 1) is None
