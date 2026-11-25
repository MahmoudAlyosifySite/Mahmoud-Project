"""Digital signal processing primitives (SPEC 5.2-5.4).

- RBJ-cookbook biquads (high-pass, low-pass, 0 dB-peak band-pass), returned
  as ``(b0, b1, b2, a1, a2)`` with ``a0`` normalised to 1.
- ``lfilter``: direct form I, zero initial state, written as an explicit
  loop so that the evaluation order
  ``b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]`` (left to right)
  is unambiguous.
- ``filtfilt_z``: the zero-phase filter of SPEC 5.4 (odd extension of ``P``
  samples, step-free start through subtracting the first sample).
- ``resample``: linear interpolation onto a grid with gap flags (SPEC 5.2).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

Coeffs = tuple[float, float, float, float, float]

#: Q of the Butterworth-equivalent biquads, computed as ``1 / sqrt(2)``
#: (= 0.7071067811865475; note ``sqrt(0.5)`` is one ulp larger).
Q_BUTTERWORTH = 1.0 / math.sqrt(2.0)


def _rbj_common(fc: float, fs: float, q: float) -> tuple[float, float, float]:
    w0 = 2.0 * math.pi * fc / fs
    c = math.cos(w0)
    s = math.sin(w0)
    alpha = s / (2.0 * q)
    return c, s, alpha


def _normalise(b0: float, b1: float, b2: float, a0: float, a1: float, a2: float) -> Coeffs:
    return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


def biquad_highpass(fc: float, fs: float, q: float) -> Coeffs:
    """RBJ high-pass: ``b = [(1+c)/2, -(1+c), (1+c)/2]``, ``a = [1+α, -2c, 1-α]``."""
    c, _, alpha = _rbj_common(fc, fs, q)
    return _normalise((1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0, 1.0 + alpha, -2.0 * c, 1.0 - alpha)


def biquad_lowpass(fc: float, fs: float, q: float) -> Coeffs:
    """RBJ low-pass: ``b = [(1-c)/2, (1-c), (1-c)/2]``, ``a = [1+α, -2c, 1-α]``."""
    c, _, alpha = _rbj_common(fc, fs, q)
    return _normalise((1.0 - c) / 2.0, 1.0 - c, (1.0 - c) / 2.0, 1.0 + alpha, -2.0 * c, 1.0 - alpha)


def biquad_bandpass(fc: float, fs: float, q: float) -> Coeffs:
    """RBJ band-pass with 0 dB peak: ``b = [α, 0, -α]``, ``a = [1+α, -2c, 1-α]``."""
    c, _, alpha = _rbj_common(fc, fs, q)
    return _normalise(alpha, 0.0, -alpha, 1.0 + alpha, -2.0 * c, 1.0 - alpha)


def biquad(kind: str, fc: float, fs: float, q: float) -> Coeffs:
    """Dispatch on ``kind`` in {"highpass", "lowpass", "bandpass"}."""
    if kind == "highpass":
        return biquad_highpass(fc, fs, q)
    if kind == "lowpass":
        return biquad_lowpass(fc, fs, q)
    if kind == "bandpass":
        return biquad_bandpass(fc, fs, q)
    raise ValueError(f"unknown biquad kind {kind!r}")


def lfilter(c: Coeffs, x: Sequence[float]) -> np.ndarray:
    """Direct form I with zero initial state (SPEC 5.3)."""
    b0, b1, b2, a1, a2 = (float(v) for v in c)
    xs = [float(v) for v in np.asarray(x, dtype=np.float64).ravel()]
    out = [0.0] * len(xs)
    x1 = x2 = y1 = y2 = 0.0
    for i, xi in enumerate(xs):
        yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        out[i] = yi
        x2, x1 = x1, xi
        y2, y1 = y1, yi
    return np.asarray(out, dtype=np.float64)


def filtfilt_z(c: Coeffs, x: Sequence[float], pad: int, dc_gain: float) -> np.ndarray:
    """Zero-phase filter of SPEC 5.4.

    ``dc_gain`` is 1 for low-pass and 0 for high-pass and band-pass.
    """
    xs = [float(v) for v in np.asarray(x, dtype=np.float64).ravel()]
    n = len(xs)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.asarray([dc_gain * xs[0]], dtype=np.float64)
    p = min(max(int(pad), 0), n - 1)
    pre = [2.0 * xs[0] - xs[k] for k in range(p, 0, -1)]
    post = [2.0 * xs[n - 1] - xs[k] for k in range(n - 2, n - 2 - p, -1)]
    xp = pre + xs + post
    c1 = xp[0]
    y = lfilter(c, [v - c1 for v in xp])
    y = [float(v) + dc_gain * c1 for v in y]
    r = y[::-1]
    c2 = r[0]
    z = lfilter(c, [v - c2 for v in r])
    z = [float(v) + dc_gain * c2 for v in z]
    zr = z[::-1]
    return np.asarray(zr[p:p + n], dtype=np.float64)


def resample(t: Sequence[float], v: Sequence[float], grid: Sequence[float], gap_tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation onto ``grid`` with missing flags (SPEC 5.2).

    ``t`` must already be sorted and deduplicated. For each grid time ``g``:
    ``j`` = largest index with ``t[j] <= g`` (or -1);
    ``j == -1`` -> ``v[0]``, missing;
    ``j == last`` -> ``v[last]``, missing iff ``g - t[last] > gap_tol``;
    else ``w = (g - t[j]) / (t[j+1] - t[j])``, ``v[j] + w * (v[j+1] - v[j])``,
    missing iff ``t[j+1] - t[j] > gap_tol``.
    A stream without samples gives zeros, all missing.
    """
    ts = np.asarray(t, dtype=np.float64).ravel()
    vs = np.asarray(v, dtype=np.float64).ravel()
    g = np.asarray(grid, dtype=np.float64).ravel()
    m = ts.size
    if m == 0:
        return np.zeros(g.size, dtype=np.float64), np.ones(g.size, dtype=bool)
    values = np.empty(g.size, dtype=np.float64)
    missing = np.empty(g.size, dtype=bool)
    j_all = np.searchsorted(ts, g, side="right") - 1
    for k in range(g.size):
        j = int(j_all[k])
        gk = float(g[k])
        if j < 0:
            values[k] = vs[0]
            missing[k] = True
        elif j == m - 1:
            values[k] = vs[m - 1]
            missing[k] = (gk - float(ts[m - 1])) > gap_tol
        else:
            t0 = float(ts[j])
            t1 = float(ts[j + 1])
            w = (gk - t0) / (t1 - t0)
            v0 = float(vs[j])
            values[k] = v0 + w * (float(vs[j + 1]) - v0)
            missing[k] = (t1 - t0) > gap_tol
    return values, missing
