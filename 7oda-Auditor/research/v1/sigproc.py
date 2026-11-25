"""Vectorized NumPy implementation of the parts of SPEC section 5 that the v1
input tensor (SPEC 14.1) needs: stream cleaning, resampling onto the grid,
the 5 Hz zero-phase high-pass, the normalized energy envelope, the lag-0
template and the alignment lag.

This is the fast path used to build training data in bulk. The reference
path goes through ``auditor_ref.signal.process``; ``tests/test_v1_data.py``
checks that both give the same tensor (same lag, max |difference| < 1e-6) on
synthetic recordings with jittered, unsorted and duplicated timestamps and
delivery gaps. The phone reproduces the tensor with the Swift implementation
of section 5, which follows the same specification.

Everything here is float64, as in the specification. ``scipy.signal.lfilter``
evaluates the same recursion as the direct form I of 5.3 (zero initial state);
the two differ only by floating-point rounding (about 1e-15 relative).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lfilter


@dataclass(frozen=True)
class SignalParams:
    """Defaults of SPEC 5.1 (only the fields the v1 tensor needs)."""

    fs: float = 100.0
    pre_roll: float = 0.25
    post_roll: float = 0.25
    hp_cutoff: float = 5.0
    hp_q: float = 1.0 / math.sqrt(2.0)
    pad: int = 30
    gap_tolerance: float = 0.025
    lag_min: int = -10
    lag_max: int = 25
    template_decay: float = 0.03
    accel_noise_floor: float = 1e-6
    gyro_noise_floor: float = 1e-6


DEFAULT_PARAMS = SignalParams()


# ---------------------------------------------------------------------------
# 4 / 5.2  Stream cleaning and resampling
# ---------------------------------------------------------------------------

def sort_dedup(t, values):
    """Stable sort by timestamp, then drop every sample whose timestamp is not
    strictly greater than the previous kept sample (SPEC 4).

    ``values`` is a 2-D array-like ``[C, M]`` (one row per field). Returns
    ``(t_sorted, values_sorted)`` as float64 arrays.
    """
    t = np.asarray(t, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if v.ndim == 1:
        v = v[None, :]
    if t.size == 0:
        return t.copy(), v.reshape(v.shape[0], 0)
    order = np.argsort(t, kind="stable")
    ts = t[order]
    vs = v[:, order]
    # For a sorted array the last kept sample always has the largest timestamp
    # seen so far, so "strictly greater than the previous kept sample" is the
    # same as "strictly greater than the previous sample".
    keep = np.empty(ts.size, dtype=bool)
    keep[0] = True
    keep[1:] = ts[1:] > ts[:-1]
    return ts[keep], vs[:, keep]


def resample(ts, v, grid, gap_tolerance):
    """Linear interpolation onto ``grid`` with the edge and gap rules of 5.2.

    ``ts`` must be sorted and deduplicated, ``v`` is ``[C, M]``. Returns
    ``(values [C, N], missing [N])``.
    """
    grid = np.asarray(grid, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        v = v[None, :]
    n_grid = grid.size
    m = ts.size
    if m == 0:
        return np.zeros((v.shape[0], n_grid)), np.ones(n_grid, dtype=bool)
    j = np.searchsorted(ts, grid, side="right") - 1  # largest j with ts[j] <= g
    out = np.empty((v.shape[0], n_grid))
    missing = np.empty(n_grid, dtype=bool)

    before = j < 0
    out[:, before] = v[:, :1]
    missing[before] = True

    last = j == m - 1
    out[:, last] = v[:, -1:]
    missing[last] = (grid[last] - ts[m - 1]) > gap_tolerance

    mid = ~(before | last)
    if np.any(mid):
        jm = j[mid]
        t0 = ts[jm]
        t1 = ts[jm + 1]
        w = (grid[mid] - t0) / (t1 - t0)
        v0 = v[:, jm]
        v1 = v[:, jm + 1]
        out[:, mid] = v0 + w * (v1 - v0)
        missing[mid] = (t1 - t0) > gap_tolerance
    return out, missing


def grid_for(hapticStart: float, duration: float, p: SignalParams = DEFAULT_PARAMS):
    """``t0``, ``N`` and the grid times of 5.2."""
    t0 = hapticStart - p.pre_roll
    n = int((p.pre_roll + duration + p.post_roll) * p.fs + 0.5)
    n = max(n, 0)
    grid = t0 + np.arange(n, dtype=np.float64) / p.fs
    return t0, n, grid


# ---------------------------------------------------------------------------
# 5.3 / 5.4  Biquads and the zero-phase filter
# ---------------------------------------------------------------------------

def biquad_highpass(fc: float, fs: float, q: float):
    """RBJ high-pass, normalized so that a0 = 1. Returns ``(b, a)``."""
    w0 = 2.0 * math.pi * fc / fs
    c = math.cos(w0)
    s = math.sin(w0)
    alpha = s / (2.0 * q)
    a0 = 1.0 + alpha
    b = np.array([(1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0]) / a0
    a = np.array([a0, -2.0 * c, 1.0 - alpha]) / a0
    return b, a


def biquad_lowpass(fc: float, fs: float, q: float):
    """RBJ low-pass, normalized so that a0 = 1. Returns ``(b, a)``."""
    w0 = 2.0 * math.pi * fc / fs
    c = math.cos(w0)
    s = math.sin(w0)
    alpha = s / (2.0 * q)
    a0 = 1.0 + alpha
    b = np.array([(1.0 - c) / 2.0, (1.0 - c), (1.0 - c) / 2.0]) / a0
    a = np.array([a0, -2.0 * c, 1.0 - alpha]) / a0
    return b, a


def filtfilt_z(x, b, a, pad: int, dc_gain: float):
    """SPEC 5.4 applied to every row of ``x`` ``[C, n]``."""
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[None, :]
    n = x.shape[-1]
    if n == 0:
        out = x.copy()
    elif n == 1:
        out = dc_gain * x
    else:
        p = min(pad, n - 1)
        pre = 2.0 * x[:, :1] - x[:, np.arange(p, 0, -1)]
        post = 2.0 * x[:, n - 1:n] - x[:, np.arange(n - 2, n - 2 - p, -1)]
        xp = np.concatenate([pre, x, post], axis=1)
        c1 = xp[:, :1]
        y = lfilter(b, a, xp - c1, axis=1) + dc_gain * c1
        r = y[:, ::-1]
        c2 = r[:, :1]
        z = lfilter(b, a, r - c2, axis=1) + dc_gain * c2
        out = z[:, ::-1][:, p:p + n]
    return out[0] if squeeze else np.ascontiguousarray(out)


# ---------------------------------------------------------------------------
# 3.3 / 5.6  Template and alignment
# ---------------------------------------------------------------------------

def curve_at(ramp: dict, tau):
    """Piecewise-linear ramp intensity (3.3), clamped outside ``[0, d]``."""
    pts = ramp["curve"]
    xs = np.array([float(c["time"]) for c in pts])
    ys = np.array([float(c["intensity"]) for c in pts])
    return np.interp(tau, xs, ys)


def template_lag0(challenge: dict, n: int, p: SignalParams = DEFAULT_PARAMS):
    """Template amplitude ``h[n]`` at lag 0 (5.6), length ``n``."""
    tau = np.arange(n, dtype=np.float64) / p.fs - p.pre_roll
    h = np.zeros(n)
    dec = p.template_decay
    for ev in challenge["events"]:
        t = float(ev["time"])
        if ev["kind"] == "tap":
            on = tau >= t
            h[on] += float(ev["intensity"]) * np.exp(-(tau[on] - t) / dec)
        elif ev["kind"] == "ramp":
            d = float(ev["duration"])
            inside = (tau >= t) & (tau < t + d)
            after = tau >= t + d
            if np.any(inside):
                h[inside] += curve_at(ev, tau[inside] - t)
            if np.any(after):
                end = float(curve_at(ev, d))
                h[after] += end * np.exp(-(tau[after] - t - d) / dec)
    return h


def _pearson(x, y) -> float:
    xm = x - x.mean()
    ym = y - y.mean()
    vx = float(np.dot(xm, xm))
    vy = float(np.dot(ym, ym))
    if vx == 0.0 or vy == 0.0:
        return 0.0
    return float(np.dot(xm, ym) / math.sqrt(vx * vy))


def align(x, h, p: SignalParams = DEFAULT_PARAMS):
    """Lag search of 5.6. Returns ``(lag, ncc, peakMargin, ncc_by_lag)``."""
    n = x.size
    lags = np.arange(p.lag_min, p.lag_max + 1)
    nccs = np.zeros(lags.size)
    for i, L in enumerate(lags):
        lo = max(0, int(L))
        hi = min(n, n + int(L))
        if hi - lo < 10:
            continue
        nccs[i] = _pearson(x[lo:hi], h[lo - L:hi - L])
    # argmax, ties -> smaller |L|, then smaller L
    best_i = None
    for i in sorted(range(lags.size), key=lambda k: (abs(int(lags[k])), int(lags[k]))):
        if best_i is None or nccs[i] > nccs[best_i]:
            best_i = i
    lag = int(lags[best_i])
    ncc = float(nccs[best_i])
    far = np.abs(lags - lag) >= 4
    peak_margin = ncc - float(nccs[far].max()) if np.any(far) else ncc
    return lag, ncc, peak_margin, nccs


# ---------------------------------------------------------------------------
# The whole fast path
# ---------------------------------------------------------------------------

@dataclass
class FastProcessed:
    """The subset of ``ProcessedSignal`` (SPEC 10.1) that the v1 tensor uses."""

    n: int
    accel_hp: np.ndarray          # [3, N]
    gyro_hp: np.ndarray           # [3, N]
    gravity: np.ndarray           # [3, N] device-motion gravity (raw, resampled)
    template: np.ndarray          # [N]   h at lag 0
    lag: int
    ncc: float
    peak_margin: float
    gap_fraction: float
    sample_rate: float
    extras: dict = field(default_factory=dict)


def _stream(rec: dict, key: str, fields):
    s = rec.get(key) or {}
    t = s.get("t") or []
    vals = [s.get(f) or [] for f in fields]
    if len(t) == 0:
        return np.zeros(0), np.zeros((len(fields), 0))
    return sort_dedup(t, vals)


def process_fast(rec: dict, p: SignalParams = DEFAULT_PARAMS) -> FastProcessed:
    """Fast equivalent of ``auditor_ref.signal.process`` for the v1 tensor."""
    ch = rec["challenge"]
    _, n, grid = grid_for(float(rec["hapticStart"]), float(ch["duration"]), p)

    ta, va = _stream(rec, "accel", ("x", "y", "z"))
    tg, vg = _stream(rec, "gyro", ("x", "y", "z"))
    tm, vm = _stream(rec, "motion", ("gravityX", "gravityY", "gravityZ"))
    accel, miss_a = resample(ta, va, grid, p.gap_tolerance)
    gyro, miss_g = resample(tg, vg, grid, p.gap_tolerance)
    gravity, _ = resample(tm, vm, grid, p.gap_tolerance)

    gap_fraction = float(np.mean(miss_a | miss_g)) if n > 0 else 0.0
    sample_rate = 1.0 / float(np.median(np.diff(ta))) if ta.size >= 2 else 0.0

    b, a = biquad_highpass(p.hp_cutoff, p.fs, p.hp_q)
    accel_hp = filtfilt_z(accel, b, a, p.pad, 0.0)
    gyro_hp = filtfilt_z(gyro, b, a, p.pad, 0.0)

    ea = np.sum(accel_hp * accel_hp, axis=0)
    eg = np.sum(gyro_hp * gyro_hp, axis=0)
    pre_end = int(p.pre_roll * p.fs + 0.5) - 5
    if pre_end < 1 or n == 0:
        na, ng = p.accel_noise_floor, p.gyro_noise_floor
    else:
        na = max(float(np.mean(ea[:pre_end])), p.accel_noise_floor)
        ng = max(float(np.mean(eg[:pre_end])), p.gyro_noise_floor)
    env = ea / na + eg / ng
    x = np.sqrt(env)
    h = template_lag0(ch, n, p)
    lag, ncc, peak_margin, _ = align(x, h, p)
    return FastProcessed(n=n, accel_hp=accel_hp, gyro_hp=gyro_hp, gravity=gravity,
                         template=h, lag=lag, ncc=ncc, peak_margin=peak_margin,
                         gap_fraction=gap_fraction, sample_rate=sample_rate,
                         extras={"envelope": env, "noise_accel": na, "noise_gyro": ng})
