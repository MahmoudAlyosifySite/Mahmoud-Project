"""v0 features (SPEC section 7): 36 hand-made features, ``featureVersion = 1``.

``extract_absolute(proc)`` gives the absolute features; ``extract(proc,
calibration)`` subtracts the calibration's table features on the indices
``CAL_RELATIVE`` (SPEC 7.6).
"""

from __future__ import annotations

import math

import numpy as np

from . import dsp
from ._numerics import TieAudit, mean, median, note, note_argmax, seq_sum, solve_partial_pivot
from .challenge import curve_at
from .signal import Processed

FEATURE_VERSION = 1

FEATURE_NAMES = (
    "tap_alpha_ax", "tap_alpha_ay", "tap_alpha_az", "tap_alpha_gx", "tap_alpha_gy", "tap_alpha_gz",
    "tap_beta_total", "tap_gamma_total",
    "tap_share_ax", "tap_share_ay", "tap_share_az",
    "tap_share_gx", "tap_share_gy", "tap_share_gz",
    "tap_log_accel_gyro_ratio", "tap_decay", "tap_settle",
    "ramp_gain_accel", "ramp_exponent_accel", "ramp_gain_gyro", "ramp_exponent_gyro",
    "ramp_rise", "ramp_decay",
    "ramp_share_ax", "ramp_share_ay", "ramp_share_az",
    "ramp_share_gx", "ramp_share_gy", "ramp_share_gz",
    "quiet_tremor_accel", "quiet_tremor_gyro", "quiet_rms_accel", "quiet_rms_gyro",
    "gravity_x", "gravity_y", "gravity_z",
)
assert len(FEATURE_NAMES) == 36

#: Calibration-relative indices R of SPEC 7.6.
CAL_RELATIVE = (0, 1, 2, 3, 4, 5, 6, 7, 14, 17, 18, 19, 20)

TAP_RIDGE_LAMBDA = 0.01
RAMP_RIDGE_LAMBDA = 0.001


def ridge(rows: list[list[float]], targets: list[float], lam: float) -> list[float]:
    """Ridge fit with an unpenalised intercept (SPEC 7.2).

    Solves ``(X^T X + diag(0, lam, ..., lam)) theta = X^T y`` by Gaussian
    elimination with partial pivoting; ``X^T X`` and ``X^T y`` are
    accumulated row by row (left-to-right sums), then ``lam`` is added to
    the slope diagonal entries. ``ridge3`` = rows ``[1, u, v]``, ``ridge2`` =
    rows ``[1, u]``.
    """
    k = len(rows[0])
    a = [[0.0] * k for _ in range(k)]
    b = [0.0] * k
    for x, y in zip(rows, targets):
        for r in range(k):
            xr = x[r]
            for c in range(k):
                a[r][c] = a[r][c] + xr * x[c]
            b[r] = b[r] + xr * y
    for r in range(1, k):
        a[r][r] = a[r][r] + lam
    return solve_partial_pivot(a, b)


def ls_slope(y: list[float]) -> float:
    """Least-squares slope of ``y`` against ``0, 1, ..., m-1``.

    ``slope = sum((x - xbar) * (y - ybar)) / sum((x - xbar)^2)``.
    """
    m = len(y)
    xs = [float(i) for i in range(m)]
    xbar = mean(xs)
    ybar = mean(y)
    sxy = 0.0
    sxx = 0.0
    for xi, yi in zip(xs, y):
        dx = xi - xbar
        sxy = sxy + dx * (yi - ybar)
        sxx = sxx + dx * dx
    return sxy / sxx


def _sumsq(arr: np.ndarray) -> float:
    return seq_sum(arr * arr)


def tap_measurements(proc: Processed, audit: TieAudit | None = None) -> list[dict]:
    """Per-tap window measurements of SPEC 7.1 (skipped taps omitted)."""
    events = proc.challenge.get("events") or []
    n = proc.n
    env = proc.envelope
    end_next = proc.end_index + 25
    out = []
    for ei, ev in enumerate(events):
        if ev["kind"] != "tap":
            continue
        o = proc.index(ev["time"])
        nxt = proc.index(events[ei + 1]["time"]) if ei + 1 < len(events) else end_next
        ws = max(o - 1, 0)
        we = min(o + 9, nxt - 1, n)
        if ws >= we:
            continue
        energies = [_sumsq(proc.accel_hp[c, ws:we]) for c in range(3)]
        energies += [_sumsq(proc.gyro_hp[c, ws:we]) for c in range(3)]
        ea = energies[0] + energies[1] + energies[2]
        eg = energies[3] + energies[4] + energies[5]
        # peak: first max of env over [ws, min(o + 3, we))
        pe = min(o + 3, we)
        if pe <= ws:
            p = ws
        else:
            p = ws
            best = env[ws]
            for i in range(ws + 1, pe):
                if env[i] > best:
                    best = env[i]
                    p = i
            note_argmax(audit, f"tap{ei}.peak", [float(env[i]) for i in range(ws, pe)], p - ws)
        # decay
        last = min(p + 4, we - 1)
        pts = list(range(p, last + 1))
        if len(pts) >= 3:
            decay = -ls_slope([math.log(float(env[i]) + 0.1) for i in pts])
        else:
            decay = 0.0
        # settle
        settle = we - p
        thr = 0.1 * float(env[p])
        k = 0
        while p + k < we:
            val = float(env[p + k])
            note(audit, f"tap{ei}.settle{k}", val, thr)
            if val < thr:
                settle = k
                break
            k += 1
        out.append({
            "event": ei,
            "onset": o,
            "window": (ws, we),
            "energies": energies,
            "EA": ea,
            "EG": eg,
            "peak": p,
            "decay": decay,
            "settle": settle,
            "intensity": float(ev["intensity"]),
            "sharpness": float(ev["sharpness"]),
        })
    return out


def tap_features(meas: list[dict]) -> list[float]:
    """Features 0-16 (SPEC 7.3); all zero if every tap was skipped."""
    if not meas:
        return [0.0] * 17
    rows = [[1.0, math.log(m["intensity"]), m["sharpness"] - 0.5] for m in meas]
    f = []
    for c in range(6):
        theta = ridge(rows, [math.log(m["energies"][c] + 1e-7) for m in meas], TAP_RIDGE_LAMBDA)
        f.append(theta[0])
    theta = ridge(rows, [math.log(m["EA"] / 1e-4 + m["EG"] / 1e-4 + 1e-6) for m in meas], TAP_RIDGE_LAMBDA)
    f.append(theta[1])
    f.append(theta[2])
    for c in range(3):
        f.append(median([m["energies"][c] / (m["EA"] + 1e-12) for m in meas]))
    for c in range(3, 6):
        f.append(median([m["energies"][c] / (m["EG"] + 1e-12) for m in meas]))
    f.append(median([math.log((m["EA"] + 1e-7) / (m["EG"] + 1e-7)) for m in meas]))
    f.append(median([m["decay"] for m in meas]))
    f.append(mean([float(m["settle"]) for m in meas]))
    return f


def ramp_measurements(proc: Processed, audit: TieAudit | None = None) -> list[dict]:
    """Per-ramp measurements of SPEC 7.4."""
    cfg = proc.config
    events = proc.challenge.get("events") or []
    n = proc.n
    lag = proc.lag
    amp_a = np.sqrt(proc.energy_accel)
    amp_g = np.sqrt(proc.energy_gyro)
    out = []
    for ei, ev in enumerate(events):
        if ev["kind"] != "ramp":
            continue
        tj = ev["time"]
        dj = ev["duration"]
        s = proc.clamp(proc.index(tj))
        e = proc.clamp(proc.index(tj + dj))

        def intensity(i: int) -> float:
            tau = float(i - lag) / cfg.fs - cfg.pre_roll - tj
            tau = min(max(tau, 0.0), dj)
            return curve_at(ev, tau)

        pts = list(range(s + 2, e))
        u = [math.log(intensity(i)) for i in pts]
        ya = [math.log(float(amp_a[i]) + 1e-4) for i in pts]
        yg = [math.log(float(amp_g[i]) + 1e-4) for i in pts]
        if len(pts) >= 3:
            rows = [[1.0, ui] for ui in u]
            rho_a, kappa_a = ridge(rows, ya, RAMP_RIDGE_LAMBDA)
            rho_g, kappa_g = ridge(rows, yg, RAMP_RIDGE_LAMBDA)
        else:
            kappa_a = kappa_g = 1.0
            if pts:
                rho_a = mean(ya) - mean(u)
                rho_g = mean(yg) - mean(u)
            else:
                rho_a = rho_g = 0.0
        # rise
        if s >= e:
            rise = 0
        else:
            mx = max(float(v) for v in amp_a[s:e])
            thr = 0.63 * mx
            rise = 0
            k = 0
            while s + k < e:
                val = float(amp_a[s + k])
                note(audit, f"ramp{ei}.rise{k}", val, thr)
                if val >= thr:
                    rise = k
                    break
                k += 1
        # decay after the ramp end
        ref_idx = [i for i in (e - 2, e - 1) if s <= i < e]
        ref = mean([float(amp_a[i]) for i in ref_idx]) if ref_idx else 0.0
        thr = 0.37 * ref
        decay_t = 10
        for k in range(10):
            if e + k >= n:
                continue
            val = float(amp_a[e + k])
            note(audit, f"ramp{ei}.decay{k}", val, thr)
            if val < thr:
                decay_t = k
                break
        denom_a = seq_sum(proc.energy_accel[s:e]) + 1e-12
        denom_g = seq_sum(proc.energy_gyro[s:e]) + 1e-12
        shares_a = [_sumsq(proc.accel_hp[c, s:e]) / denom_a for c in range(3)]
        shares_g = [_sumsq(proc.gyro_hp[c, s:e]) / denom_g for c in range(3)]
        out.append({
            "event": ei,
            "range": (s, e),
            "rhoA": rho_a, "kappaA": kappa_a, "rhoG": rho_g, "kappaG": kappa_g,
            "rise": rise, "decayT": decay_t,
            "sharesA": shares_a, "sharesG": shares_g,
        })
    return out


def ramp_features(meas: list[dict]) -> list[float]:
    """Features 17-28 (SPEC 7.4): means over ramps (0 if there are none)."""
    if not meas:
        return [0.0] * 12
    f = [
        mean([m["rhoA"] for m in meas]),
        mean([m["kappaA"] for m in meas]),
        mean([m["rhoG"] for m in meas]),
        mean([m["kappaG"] for m in meas]),
        mean([float(m["rise"]) for m in meas]),
        mean([float(m["decayT"]) for m in meas]),
    ]
    for c in range(3):
        f.append(mean([m["sharesA"][c] for m in meas]))
    for c in range(3):
        f.append(mean([m["sharesG"][c] for m in meas]))
    return f


def tremor_bandpassed(proc: Processed) -> tuple[np.ndarray, np.ndarray]:
    """``bA``, ``bG`` of SPEC 7.5 (cached on ``proc``)."""
    if "tremor_bp" not in proc.cache:
        cfg = proc.config
        bp = cfg.tremor_bandpass()
        if proc.n:
            ba = np.vstack([dsp.filtfilt_z(bp, proc.accel[c], cfg.pad, 0.0) for c in range(3)])
            bg = np.vstack([dsp.filtfilt_z(bp, proc.gyro[c], cfg.pad, 0.0) for c in range(3)])
        else:
            ba = bg = np.zeros((3, 0))
        proc.cache["tremor_bp"] = (ba, bg)
    return proc.cache["tremor_bp"]


def quiet_features(proc: Processed, audit: TieAudit | None = None) -> list[float]:
    """Features 29-35 (SPEC 7.5)."""
    q0, q1 = proc.quiet_range
    ba, bg = tremor_bandpassed(proc)
    tre_a = ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2]
    tre_g = bg[0] * bg[0] + bg[1] * bg[1] + bg[2] * bg[2]
    f = [
        math.log(mean(tre_a[q0:q1]) + 1e-10),
        math.log(mean(tre_g[q0:q1]) + 1e-10),
        math.log(math.sqrt(mean(proc.energy_accel[q0:q1])) + 1e-6),
        math.log(math.sqrt(mean(proc.energy_gyro[q0:q1])) + 1e-6),
    ]
    gx = mean(proc.gravity[0])
    gy = mean(proc.gravity[1])
    gz = mean(proc.gravity[2])
    norm = math.sqrt(gx * gx + gy * gy + gz * gz)
    note(audit, "features.gravityNorm", norm, 1e-6)
    if norm < 1e-6:
        f += [0.0, 0.0, -1.0]
    else:
        f += [gx / norm, gy / norm, gz / norm]
    return f


def extract_absolute(proc: Processed, audit: TieAudit | None = None) -> np.ndarray:
    """The 36 absolute features of SPEC 7.3-7.5."""
    f = tap_features(tap_measurements(proc, audit))
    f += ramp_features(ramp_measurements(proc, audit))
    f += quiet_features(proc, audit)
    return np.asarray(f, dtype=np.float64)


def relative(features_abs, calibration: dict | None) -> np.ndarray:
    """Apply SPEC 7.6 to an absolute feature vector."""
    f = np.array(features_abs, dtype=np.float64, copy=True)
    if calibration is not None:
        table = calibration["tableFeatures"]
        for r in CAL_RELATIVE:
            f[r] = f[r] - float(table[r])
    return f


def extract(proc: Processed, calibration: dict | None = None, audit: TieAudit | None = None) -> np.ndarray:
    """Features of SPEC 7 (calibration-relative on ``CAL_RELATIVE`` when given)."""
    return relative(extract_absolute(proc, audit), calibration)
