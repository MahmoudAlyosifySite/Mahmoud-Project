"""Recording -> compact item (quality report + v0 features), using auditor_ref only."""
from __future__ import annotations

import copy
import datetime as _dt
from typing import Iterable

import numpy as np

import importlib

from auditor_ref import calibration as calib_mod
from auditor_ref import features, quality, signal

# Feature names and index groups of SPEC section 7 (d = 36)
_AX = ("ax", "ay", "az", "gx", "gy", "gz")
FEATURE_NAMES = ([f"tap_alpha_{c}" for c in _AX] + ["tap_beta_total", "tap_gamma_total"]
                 + [f"tap_share_{c}" for c in _AX] + ["tap_log_accel_gyro_ratio", "tap_decay", "tap_settle",
                 "ramp_gain_accel", "ramp_exponent_accel", "ramp_gain_gyro", "ramp_exponent_gyro", "ramp_rise",
                 "ramp_decay"] + [f"ramp_share_{c}" for c in _AX]
                 + ["quiet_tremor_accel", "quiet_tremor_gyro", "quiet_rms_accel", "quiet_rms_gyro",
                    "gravity_x", "gravity_y", "gravity_z"])
TAP = list(range(0, 17))
RAMP = list(range(17, 29))
QUIET = list(range(29, 33))
GRAVITY = [33, 34, 35]
GYRO_ONLY = [3, 4, 5, 11, 12, 13, 14, 19, 20, 26, 27, 28, 30, 32]
MASKS = {
    "all": list(range(36)),
    "taps+quiet+gravity (no ramps)": TAP + QUIET + GRAVITY,
    "taps only": TAP,
    "ramps+quiet+gravity (no taps)": RAMP + QUIET + GRAVITY,
    "no gyro-derived features": [i for i in range(36) if i not in GYRO_ONLY],
    "no gravity": [i for i in range(36) if i not in GRAVITY],
}
METRIC_KEYS = ("ncc", "snr", "lag", "peakMargin", "gapFraction", "motionRMS", "gravityAngleDeg", "quietGyroRMS")


def default_mode(rec: dict) -> str:
    return "onTable" if rec.get("purpose") == "calibration" else "handheld"


def zero_gyro(rec: dict) -> dict:
    """The same recording as an accelerometer-only phone would give (gyro stream zeroed)."""
    r = dict(rec)
    g = rec["gyro"]
    r["gyro"] = {"t": g["t"], "x": [0.0] * len(g["x"]), "y": [0.0] * len(g["y"]), "z": [0.0] * len(g["z"])}
    return r


#: Feature sets that can be computed next to the adopted one (item key ``f_abs_<name>``, absolute values).
EXTRA_FEATURES = {"v1": "auditor_ref.features_v1", "v2": "auditor_ref.features_v2"}


def extra_extractor(name: str):
    return importlib.import_module(EXTRA_FEATURES[name]).extract_absolute


def item_from(rec: dict, calibration: dict | None, mode: str | None = None, nogyro: bool = False,
              extra: tuple = ()) -> dict:
    proc = signal.process(rec)
    rep = quality.evaluate(proc, calibration, mode or default_mode(rec))
    f_abs = features.extract_absolute(proc)
    lab = rec.get("label") or {}
    sim = rec.get("sim") or {}
    item = {
        "id": rec.get("id"), "purpose": rec.get("purpose"), "person": lab.get("personID"),
        "name": lab.get("personName"), "hand": lab.get("hand"), "accessory": lab.get("accessory", "none"),
        "posture": lab.get("posture"), "session": lab.get("sessionID"), "createdAt": rec.get("createdAt"),
        "hapticStart": float(rec.get("hapticStart", 0.0)), "duration": float(rec["challenge"]["duration"]),
        "passed": bool(rep["passed"]), "issues": list(rep["issues"]),
        "metrics": {k: float(rep["metrics"][k]) for k in METRIC_KEYS if k in rep["metrics"]},
        "f_abs": np.asarray(f_abs, float), "study": sim.get("study"), "grip": sim.get("grip"),
    }
    for name in extra:
        item[f"f_abs_{name}"] = np.asarray(extra_extractor(name)(proc), float)
    if nogyro:
        p2 = signal.process(zero_gyro(rec))
        r2 = quality.evaluate(p2, calibration, mode or default_mode(rec))
        item["f_abs_nogyro"] = np.asarray(features.extract_absolute(p2), float)
        item["passed_nogyro"] = bool(r2["passed"])
    return item, proc, rep


def build_calibration(recs: Iterable[dict], device: str = "sim", now=None, extra: tuple = ()):
    """SPEC 8 from on-table recordings. Returns (calibration dict or None, items)."""
    procs, reps, items = [], [], []
    for r in recs:
        it, p, rep = item_from(r, None, "onTable", extra=extra)
        procs.append(p)
        reps.append(rep)
        items.append(it)
    now = now or _dt.datetime(2026, 12, 1, tzinfo=_dt.timezone.utc)
    return calib_mod.build(procs, reps, device, now), items


def relative(f_abs: np.ndarray, calibration: dict | None) -> np.ndarray:
    return np.asarray(features.relative(f_abs, calibration), float)


def replay(rec: dict, new_challenge: dict, new_start: float) -> dict:
    """An attacker feeds a recorded response into a check with a different challenge.
    Timestamps are moved so the old hapticStart lands on the new one; if the new pattern is
    longer, the streams are padded by repeating their quiet beginning (no visible gaps)."""
    r = copy.deepcopy(rec)
    shift = new_start - rec["hapticStart"]
    r["challenge"] = new_challenge
    r["hapticStart"] = new_start
    need_end = new_start + new_challenge["duration"] + 0.35
    for s in ("accel", "gyro", "motion"):
        st = r[s]
        t = [v + shift for v in st["t"]]
        keys = [k for k in st if k != "t"]
        if t and t[-1] < need_end:
            dt = 0.01
            n_add = int((need_end - t[-1]) / dt) + 1
            src = list(range(min(25, len(t))))
            for j in range(n_add):
                i = src[j % len(src)]
                t.append(t[-1] + dt)
                for k in keys:
                    st[k].append(st[k][i])
        st["t"] = t
    r["purpose"] = "check"
    return r
