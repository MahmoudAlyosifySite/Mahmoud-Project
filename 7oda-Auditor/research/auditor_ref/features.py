"""v0 features (SPEC section 7): 39 hand-made features, ``featureVersion = 2``.

Every amplitude statistic is a ratio of sums: energies are summed over samples and
over events first, and a logarithm (if any) is taken only of the aggregate. Shapes
(axis shares and normalised cross products) divide sums of simultaneously sampled
channels of one stream, so the unknown sampling phase of an aliased carrier, the
actuator's frequency response and the grid interpolation cancel. Tap windows are
split into the driven burst ``[o - 1, o + 2)`` and the free response ``[o + 2, o + 9)``
(the phone ringing on the hand after the click).

All sums are left-to-right (``seq_sum``); no comparisons are made on computed values,
so no tie audit is needed. Empty windows contribute zero sums, and every ratio
carries an epsilon, so all features are finite for any input (a recording with no
taps or no ramps gives 0 for that group's features).

``extract_absolute(proc)`` gives the absolute features; ``extract(proc, calibration)``
subtracts the calibration's table features on the indices ``CAL_RELATIVE`` (SPEC 7.5).
The previous set (featureVersion 1, d = 36) is kept in ``features_v1`` for comparisons.
"""

from __future__ import annotations

import math

import numpy as np

from . import dsp
from ._numerics import TieAudit, mean, note, seq_sum
from .challenge import curve_at
from .signal import Processed

FEATURE_VERSION = 2

_A = ("ax", "ay", "az")
_G = ("gx", "gy", "gz")
_X = ("axy", "axz", "ayz")
_PAIRS = ((0, 1), (0, 2), (1, 2))

FEATURE_NAMES = tuple(
    ["tap_gain_accel", "tap_log_accel_gyro"]
    + [f"tap_burst_share_{c}" for c in _A] + [f"tap_burst_cross_{c}" for c in _X]
    + [f"tap_burst_share_{c}" for c in _G]
    + [f"tap_tail_share_{c}" for c in _A] + [f"tap_tail_cross_{c}" for c in _X]
    + [f"tap_tail_share_{c}" for c in _G] + ["tap_tail_log_accel_gyro"]
    + ["ramp_gain_accel", "ramp_log_accel_gyro"]
    + [f"ramp_share_{c}" for c in _A] + [f"ramp_cross_{c}" for c in _X] + [f"ramp_share_{c}" for c in _G]
    + ["quiet_tremor_accel", "quiet_tremor_gyro", "quiet_rms_accel", "quiet_rms_gyro",
       "gravity_x", "gravity_y", "gravity_z"]
)
assert len(FEATURE_NAMES) == 39

#: Calibration-relative indices R of SPEC 7.5 (log gains and log accel/gyro ratios).
CAL_RELATIVE = (0, 1, 20, 21, 22)

EPS = 1e-12
BURST = (-1, 2)       # burst window [o - 1, o + 2)
TAIL = (2, 9)         # free-response window [o + 2, o + 9)
RAMP_SKIP = 2         # ramp core window [s + 2, e)


def _ln_ratio(a: float, b: float) -> float:
    return math.log((a + EPS) / (b + EPS))


class _Shape:
    """Running per-channel energy and cross-product totals of one stream (left-to-right)."""

    def __init__(self, cross: bool):
        self.e = [0.0, 0.0, 0.0]
        self.x = [0.0, 0.0, 0.0] if cross else None

    def add(self, sig: np.ndarray, a: int, b: int) -> None:
        if b <= a:
            return
        for c in range(3):
            v = sig[c, a:b]
            self.e[c] = self.e[c] + seq_sum(v * v)
        if self.x is not None:
            for k, (c, d) in enumerate(_PAIRS):
                self.x[k] = self.x[k] + seq_sum(sig[c, a:b] * sig[d, a:b])

    @property
    def total(self) -> float:
        return (self.e[0] + self.e[1]) + self.e[2]

    def shares(self) -> list[float]:
        t = self.total
        return [self.e[c] / (t + EPS) for c in range(3)]

    def cross(self) -> list[float]:
        t = self.total
        return [self.x[k] / (t + EPS) for k in range(3)]


def tap_windows(proc: Processed) -> list[dict]:
    """Per-tap windows (time order); ``u = min(next - 1, N)`` bounds all three."""
    events = proc.challenge.get("events") or []
    n = proc.n
    end_next = proc.end_index + 25
    out = []
    for ei, ev in enumerate(events):
        if ev["kind"] != "tap":
            continue
        o = proc.index(ev["time"])
        nxt = proc.index(events[ei + 1]["time"]) if ei + 1 < len(events) else end_next
        u = min(nxt - 1, n)
        out.append({
            "event": ei, "onset": o, "intensity": float(ev["intensity"]),
            "full": (max(o + BURST[0], 0), min(o + TAIL[1], u)),
            "burst": (max(o + BURST[0], 0), min(o + BURST[1], u)),
            "tail": (max(o + TAIL[0], 0), min(o + TAIL[1], u)),
        })
    return out


def tap_features(proc: Processed) -> list[float]:
    """Features 0-20 (SPEC 7.1-7.2)."""
    ea_full = eg_full = i2 = 0.0
    burst_a, burst_g = _Shape(True), _Shape(False)
    tail_a, tail_g = _Shape(True), _Shape(False)
    for w in tap_windows(proc):
        a, b = w["full"]
        if b <= a:
            continue                      # a tap whose full window is empty is skipped entirely
        ea_full = ea_full + seq_sum(proc.energy_accel[a:b])
        eg_full = eg_full + seq_sum(proc.energy_gyro[a:b])
        i2 = i2 + w["intensity"] * w["intensity"]
        burst_a.add(proc.accel_hp, *w["burst"])
        burst_g.add(proc.gyro_hp, *w["burst"])
        tail_a.add(proc.accel_hp, *w["tail"])
        tail_g.add(proc.gyro_hp, *w["tail"])
    f = [_ln_ratio(ea_full, i2), _ln_ratio(ea_full, eg_full)]
    f += burst_a.shares() + burst_a.cross() + burst_g.shares()
    f += tail_a.shares() + tail_a.cross() + tail_g.shares()
    f.append(_ln_ratio(tail_a.total, tail_g.total))
    return f


def ramp_windows(proc: Processed) -> list[dict]:
    """Per-ramp core windows ``[s + 2, e)`` (time order; s, e clamped to [0, N])."""
    events = proc.challenge.get("events") or []
    out = []
    for ei, ev in enumerate(events):
        if ev["kind"] != "ramp":
            continue
        s = proc.clamp(proc.index(ev["time"]))
        e = proc.clamp(proc.index(ev["time"] + ev["duration"]))
        out.append({"event": ei, "ramp": ev, "core": (s + RAMP_SKIP, e)})
    return out


def ramp_features(proc: Processed) -> list[float]:
    """Features 21-31 (SPEC 7.3)."""
    cfg = proc.config
    lag = proc.lag
    shape_a, shape_g = _Shape(True), _Shape(False)
    i2 = 0.0
    for w in ramp_windows(proc):
        a, b = w["core"]
        if b <= a:
            continue
        ev = w["ramp"]
        tj, dj = float(ev["time"]), float(ev["duration"])
        for i in range(a, b):
            tau = float(i - lag) / cfg.fs - cfg.pre_roll - tj
            tau = min(max(tau, 0.0), dj)
            c = curve_at(ev, tau)
            i2 = i2 + c * c
        shape_a.add(proc.accel_hp, a, b)
        shape_g.add(proc.gyro_hp, a, b)
    f = [_ln_ratio(shape_a.total, i2), _ln_ratio(shape_a.total, shape_g.total)]
    f += shape_a.shares() + shape_a.cross() + shape_g.shares()
    return f


def tremor_bandpassed(proc: Processed) -> tuple[np.ndarray, np.ndarray]:
    """``bA``, ``bG`` of SPEC 7.4 (cached on ``proc``)."""
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
    """Features 32-38 (SPEC 7.4)."""
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
    """The 39 absolute features of SPEC 7.2-7.4."""
    f = tap_features(proc) + ramp_features(proc) + quiet_features(proc, audit)
    return np.asarray(f, dtype=np.float64)


def relative(features_abs, calibration: dict | None) -> np.ndarray:
    """Apply SPEC 7.5 to an absolute feature vector."""
    f = np.array(features_abs, dtype=np.float64, copy=True)
    if calibration is not None:
        table = calibration["tableFeatures"]
        for r in CAL_RELATIVE:
            f[r] = f[r] - float(table[r])
    return f


def extract(proc: Processed, calibration: dict | None = None, audit: TieAudit | None = None) -> np.ndarray:
    """Features of SPEC 7 (calibration-relative on ``CAL_RELATIVE`` when given)."""
    return relative(extract_absolute(proc, audit), calibration)
