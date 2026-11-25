"""Recording -> compact item (quality report + SPEC 7 features), using auditor_ref only."""
from __future__ import annotations

import copy
import datetime as _dt
from typing import Iterable

import numpy as np

import importlib

from auditor_ref import calibration as calib_mod
from auditor_ref import features, quality, signal

# Feature names and index groups of SPEC section 7 (featureVersion 2, d = 39)
FEATURE_NAMES = list(features.FEATURE_NAMES)


def _where(pred) -> list:
    return [i for i, n in enumerate(FEATURE_NAMES) if pred(n)]


TAP = _where(lambda n: n.startswith("tap_"))
RAMP = _where(lambda n: n.startswith("ramp_"))
QUIET = _where(lambda n: n.startswith("quiet_"))
GRAVITY = _where(lambda n: n.startswith("gravity_"))
FREE_RESPONSE = _where(lambda n: n.startswith("tap_tail_"))
GYRO_ONLY = _where(lambda n: "gyro" in n or n.endswith(("_gx", "_gy", "_gz")))
MASKS = {
    "all": list(range(len(FEATURE_NAMES))),
    "taps+quiet+gravity (no ramps)": TAP + QUIET + GRAVITY,
    "taps only": TAP,
    "ramps+quiet+gravity (no taps)": RAMP + QUIET + GRAVITY,
    "no free-response (tap tail) features": [i for i in range(len(FEATURE_NAMES)) if i not in FREE_RESPONSE],
    "no gyro-derived features": [i for i in range(len(FEATURE_NAMES)) if i not in GYRO_ONLY],
    "no gravity": [i for i in range(len(FEATURE_NAMES)) if i not in GRAVITY],
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
EXTRA_FEATURES = {"v1": "auditor_ref.features_v1"}


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
