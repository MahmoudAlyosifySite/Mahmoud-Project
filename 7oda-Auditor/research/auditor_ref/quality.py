"""Quality gate (SPEC section 6).

``evaluate(proc, calibration, mode)`` returns
``{"metrics": {...QualityMetrics...}, "issues": [...], "passed": bool}``.
Metric keys use the JSON names of ``QualityMetrics`` (SPEC 10.1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import dsp
from ._numerics import TieAudit, mean, note
from .signal import Processed

ISSUES = ("sensorGaps", "lowSampleRate", "resting", "moving", "weakAlignment", "weakSignal", "notResting")

MESSAGES = {
    "sensorGaps": "Try again",
    "lowSampleRate": "Motion sensors are too slow on this device",
    "resting": "Hold the phone in one hand",
    "moving": "Hold still",
    "weakAlignment": "Try again",
    "weakSignal": "Vibration too weak — remove a thick case?",
    "notResting": "Put the phone flat on a table and don't touch it",
}

METRIC_KEYS = (
    "gapFraction", "sampleRate", "quietGyroRMS", "quietAccelRMS", "gravityZ",
    "motionRMS", "gravityAngleDeg", "ncc", "peakMargin", "snr", "lag",
)


@dataclass(frozen=True)
class QualityConfig:
    """Defaults of SPEC 6.2 (plus the 0.05 rad/s ``notResting`` motion limit of 6.3)."""

    max_gap_fraction: float = 0.10
    min_sample_rate: float = 70.0
    resting_gyro_factor: float = 2.5
    resting_gyro_absolute: float = 0.004
    flat_gravity_z: float = 0.95
    max_motion_rms: float = 0.30
    max_gravity_angle_deg: float = 10.0
    min_ncc: float = 0.30
    min_peak_margin: float = 0.05
    min_snr: float = 2.0
    table_max_motion_rms: float = 0.05


DEFAULT_CONFIG = QualityConfig()


def motion_lowpassed(proc: Processed) -> np.ndarray:
    """``rL_c = filtfiltZ(r_c, LP 3 Hz)`` (cached on ``proc``)."""
    if "rotation_lp" not in proc.cache:
        cfg = proc.config
        lp = cfg.motion_lowpass()
        if proc.n:
            proc.cache["rotation_lp"] = np.vstack([dsp.filtfilt_z(lp, proc.rotation[c], cfg.pad, 1.0) for c in range(3)])
        else:
            proc.cache["rotation_lp"] = np.zeros((3, 0))
    return proc.cache["rotation_lp"]


def _norm3(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def gravity_angle_deg(proc: Processed, audit: TieAudit | None = None) -> float:
    """Angle between mean gravity over the first 20 and the last 20 grid samples."""
    n = proc.n
    if n == 0:
        return 0.0
    g = proc.gravity
    k = min(20, n)
    a = (mean(g[0, :k]), mean(g[1, :k]), mean(g[2, :k]))
    b = (mean(g[0, n - k:]), mean(g[1, n - k:]), mean(g[2, n - k:]))
    na = _norm3(a)
    nb = _norm3(b)
    note(audit, "quality.gravityAngle.normA", na, 1e-9)
    note(audit, "quality.gravityAngle.normB", nb, 1e-9)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    cosv = dot / (na * nb)
    cosv = min(max(cosv, -1.0), 1.0)
    return math.acos(cosv) * 180.0 / math.pi


def compute_metrics(proc: Processed, audit: TieAudit | None = None) -> dict:
    """QualityMetrics of SPEC 6.1."""
    q0, q1 = proc.quiet_range
    e0, e1 = proc.event_range
    rl = motion_lowpassed(proc)
    motion_energy = rl[0] * rl[0] + rl[1] * rl[1] + rl[2] * rl[2]
    snr = mean(proc.envelope[e0:e1]) / max(mean(proc.envelope[q0:q1]), 1e-9)
    return {
        "gapFraction": float(proc.gap_fraction),
        "sampleRate": float(proc.sample_rate),
        "quietGyroRMS": math.sqrt(mean(proc.energy_gyro[q0:q1])),
        "quietAccelRMS": math.sqrt(mean(proc.energy_accel[q0:q1])),
        "gravityZ": abs(mean(proc.gravity[2])) if proc.n else 0.0,
        "motionRMS": math.sqrt(mean(motion_energy)),
        "gravityAngleDeg": gravity_angle_deg(proc, audit),
        "ncc": float(proc.alignment.ncc),
        "peakMargin": float(proc.alignment.peak_margin),
        "snr": float(snr),
        "lag": int(proc.alignment.lag),
    }


def _and(audit, label, a_val, a_thr, a_res, b_val, b_thr, b_res):
    # Record a comparison only where it can change the conjunction.
    if b_res:
        note(audit, label + ".a", a_val, a_thr)
    if a_res:
        note(audit, label + ".b", b_val, b_thr)
    return a_res and b_res


def _or(audit, label, a_val, a_thr, a_res, b_val, b_thr, b_res):
    # Record a comparison only where it can change the disjunction.
    if not b_res:
        note(audit, label + ".a", a_val, a_thr)
    if not a_res:
        note(audit, label + ".b", b_val, b_thr)
    return a_res or b_res


def evaluate(
    proc: Processed,
    calibration: dict | None = None,
    mode: str = "handheld",
    config: QualityConfig | None = None,
    audit: TieAudit | None = None,
) -> dict:
    """Quality report of SPEC 6.3.

    Issue order: handheld 1-6 (sensorGaps, lowSampleRate, resting, moving,
    weakAlignment, weakSignal); onTable: sensorGaps, lowSampleRate,
    weakAlignment, weakSignal, notResting (notResting appended last).
    """
    cfg = config or DEFAULT_CONFIG
    if mode not in ("handheld", "onTable"):
        raise ValueError(f"unknown quality mode {mode!r}")
    m = compute_metrics(proc, audit)
    issues: list[str] = []

    gaps = m["gapFraction"] > cfg.max_gap_fraction
    note(audit, "quality.sensorGaps", m["gapFraction"], cfg.max_gap_fraction)
    slow = m["sampleRate"] < cfg.min_sample_rate
    note(audit, "quality.lowSampleRate", m["sampleRate"], cfg.min_sample_rate)
    low_ncc = m["ncc"] < cfg.min_ncc
    flat_peak = m["peakMargin"] < cfg.min_peak_margin
    weak_align = _or(audit, "quality.weakAlignment", m["ncc"], cfg.min_ncc, low_ncc,
                     m["peakMargin"], cfg.min_peak_margin, flat_peak)
    weak_signal = m["snr"] < cfg.min_snr
    note(audit, "quality.weakSignal", m["snr"], cfg.min_snr)

    if gaps:
        issues.append("sensorGaps")
    if slow:
        issues.append("lowSampleRate")
    if mode == "handheld":
        if calibration is not None:
            rest_threshold = cfg.resting_gyro_factor * float(calibration["quietGyroRMS"])
        else:
            rest_threshold = cfg.resting_gyro_absolute
        flat = m["gravityZ"] >= cfg.flat_gravity_z
        quiet = m["quietGyroRMS"] < rest_threshold
        if _and(audit, "quality.resting", m["gravityZ"], cfg.flat_gravity_z, flat,
                m["quietGyroRMS"], rest_threshold, quiet):
            issues.append("resting")
        fast = m["motionRMS"] > cfg.max_motion_rms
        tilted = m["gravityAngleDeg"] > cfg.max_gravity_angle_deg
        if _or(audit, "quality.moving", m["motionRMS"], cfg.max_motion_rms, fast,
               m["gravityAngleDeg"], cfg.max_gravity_angle_deg, tilted):
            issues.append("moving")
        if weak_align:
            issues.append("weakAlignment")
        if weak_signal:
            issues.append("weakSignal")
    else:
        if weak_align:
            issues.append("weakAlignment")
        if weak_signal:
            issues.append("weakSignal")
        not_flat = m["gravityZ"] < cfg.flat_gravity_z
        shaky = m["motionRMS"] > cfg.table_max_motion_rms
        if _or(audit, "quality.notResting", m["gravityZ"], cfg.flat_gravity_z, not_flat,
               m["motionRMS"], cfg.table_max_motion_rms, shaky):
            issues.append("notResting")
    return {"metrics": m, "issues": issues, "passed": not issues}
