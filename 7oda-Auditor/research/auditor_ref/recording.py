"""Recording: raw sensor data and its JSON / JSON Lines form (SPEC section 4).

A recording is a plain dict with exactly the keys of SPEC 4::

    schema, id, createdAt, purpose, label (optional), challenge, hapticStart,
    accel {t, x, y, z}, gyro {t, x, y, z},
    motion {t, gravityX..Z, userAccelX..Z, rotationX..Z, attitudeW..Z},
    device (optional)

Dates are ISO-8601 UTC without fractional seconds (``2026-11-24T10:00:00Z``).
"""

from __future__ import annotations

import datetime as _dt
import json
import uuid
from pathlib import Path
from typing import Iterable

import numpy as np

SCHEMA = "7oda-auditor.recording.v1"

SENSOR_KEYS = ("t", "x", "y", "z")
MOTION_KEYS = (
    "t",
    "gravityX", "gravityY", "gravityZ",
    "userAccelX", "userAccelY", "userAccelZ",
    "rotationX", "rotationY", "rotationZ",
    "attitudeW", "attitudeX", "attitudeY", "attitudeZ",
)
PURPOSES = ("calibration", "enrollment", "accessory", "check", "test", "signalCheck")
HANDS = ("right", "left")
ACCESSORIES = ("none", "watch", "ring", "bracelet", "other")
POSTURES = ("sitting", "standing", "freshGrip", "unspecified")


def format_date(value: _dt.datetime | str | None = None) -> str:
    """ISO-8601 UTC string without fractional seconds."""
    if value is None:
        value = _dt.datetime.now(_dt.timezone.utc)
    if isinstance(value, str):
        return value
    if value.tzinfo is not None:
        value = value.astimezone(_dt.timezone.utc).replace(tzinfo=None)
    return value.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def empty_sensor_stream() -> dict:
    return {k: [] for k in SENSOR_KEYS}


def empty_motion_stream() -> dict:
    return {k: [] for k in MOTION_KEYS}


def sort_dedup(stream: dict) -> dict:
    """Stable sort by ``t`` and drop samples whose ``t`` is not strictly
    greater than the previous kept sample (SPEC 4).

    Works for sensor streams and the motion stream (every key is permuted
    with ``t``). Returns a new dict of float64 numpy arrays with the same keys.
    """
    t = np.asarray(stream.get("t", []), dtype=np.float64)
    keys = list(stream.keys())
    order = np.argsort(t, kind="stable")
    t_sorted = t[order]
    keep = []
    last = None
    for i, ti in enumerate(t_sorted):
        if last is None or ti > last:
            keep.append(i)
            last = ti
    idx = order[np.asarray(keep, dtype=np.intp)] if keep else np.zeros(0, dtype=np.intp)
    out = {}
    for k in keys:
        arr = np.asarray(stream[k], dtype=np.float64)
        if arr.shape[0] != t.shape[0]:
            raise ValueError(f"stream key {k!r} has {arr.shape[0]} values, t has {t.shape[0]}")
        out[k] = arr[idx]
    return out


def make_recording(
    challenge: dict,
    haptic_start: float,
    accel: dict,
    gyro: dict,
    motion: dict | None = None,
    *,
    purpose: str = "check",
    label: dict | None = None,
    device: dict | None = None,
    rec_id: str | None = None,
    created_at: _dt.datetime | str | None = None,
) -> dict:
    """Assemble a recording dict in the SPEC 4 JSON form.

    Arrays may be lists or numpy arrays; they are converted to lists of
    Python floats. ``label`` and ``device`` are written only when given.
    """
    if purpose not in PURPOSES:
        raise ValueError(f"unknown purpose {purpose!r}")

    def _stream(src: dict | None, keys: tuple[str, ...]) -> dict:
        if src is None:
            return {k: [] for k in keys}
        return {k: [float(v) for v in np.asarray(src[k], dtype=np.float64).ravel()] for k in keys}

    rec = {
        "schema": SCHEMA,
        "id": rec_id or str(uuid.uuid4()).upper(),
        "createdAt": format_date(created_at),
        "purpose": purpose,
    }
    if label is not None:
        rec["label"] = dict(label)
    rec["challenge"] = challenge
    rec["hapticStart"] = float(haptic_start)
    rec["accel"] = _stream(accel, SENSOR_KEYS)
    rec["gyro"] = _stream(gyro, SENSOR_KEYS)
    rec["motion"] = _stream(motion, MOTION_KEYS)
    if device is not None:
        rec["device"] = dict(device)
    return rec


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def dumps(rec: dict) -> str:
    """One recording as a compact JSON line (keys sorted, like the app)."""
    return json.dumps(rec, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json_default)


def save_jsonl(path: str | Path, recs: Iterable[dict]) -> None:
    """Write recordings as JSON Lines (one recording per line)."""
    with open(path, "w", encoding="utf-8") as fh:
        for rec in recs:
            fh.write(dumps(rec))
            fh.write("\n")


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSON Lines export; blank lines are skipped."""
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
