"""Challenge: the random haptic pattern (SPEC section 3).

``generate(seed)`` returns the JSON form of SPEC 3.4 as a plain dict::

    {"version": 1, "seed": "0x...", "duration": float,
     "events": [{"kind": "tap", "time", "intensity", "sharpness"},
                {"kind": "ramp", "time", "duration", "sharpness",
                 "curve": [{"time", "intensity"} x 3]}, ...]}

The draw order of SPEC 3.2 is normative and reproduced line by line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

from .rng import Xoshiro256, seed_to_str


@dataclass(frozen=True)
class ChallengeConfig:
    """Defaults of SPEC 3.1 (ranges are ``(low, high)`` of ``uniform(a, b)``)."""

    lead_silence: tuple[float, float] = (0.05, 0.10)
    tap_count: tuple[int, int] = (6, 8)
    tap_intensity: tuple[float, float] = (0.4, 1.0)
    tap_sharpness: tuple[float, float] = (0.0, 1.0)
    tap_gap: tuple[float, float] = (0.08, 0.15)
    ramp_count: int = 2
    ramp_duration: tuple[float, float] = (0.15, 0.25)
    ramp_sharpness: tuple[float, float] = (0.2, 0.8)
    ramp_low: tuple[float, float] = (0.25, 0.5)
    ramp_mid: tuple[float, float] = (0.25, 1.0)
    ramp_high: tuple[float, float] = (0.75, 1.0)
    ramp_gap: tuple[float, float] = (0.08, 0.12)

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = ChallengeConfig()


def generate(seed: int, config: ChallengeConfig | None = None) -> dict:
    """Generate the challenge for ``seed`` (SPEC 3.2).

    The ramp-slot draw of SPEC 3.2 is written for exactly two ramps
    (``rampCount = 2`` is fixed in SPEC 3.1); other ramp counts are rejected.
    """
    cfg = config or DEFAULT_CONFIG
    if cfg.ramp_count != 2:
        raise ValueError("SPEC 3.2 defines the draw order for exactly 2 ramps")
    rng = Xoshiro256(seed)
    lead = rng.uniform_range(*cfg.lead_silence)
    n = rng.int_in_range(*cfg.tap_count)
    s1 = 1 + min(int(math.floor(rng.uniform() * float(n))), n - 1)
    s2 = 1 + min(int(math.floor(rng.uniform() * float(n - 1))), n - 2)
    if s2 >= s1:
        s2 += 1
    slots = sorted([s1, s2])
    taps = []
    for _ in range(n):
        intensity = rng.uniform_range(*cfg.tap_intensity)
        sharpness = rng.uniform_range(*cfg.tap_sharpness)
        gap = rng.uniform_range(*cfg.tap_gap)
        taps.append((intensity, sharpness, gap))
    ramps = []
    for _ in range(2):
        duration = rng.uniform_range(*cfg.ramp_duration)
        sharpness = rng.uniform_range(*cfg.ramp_sharpness)
        ascending = rng.uniform() < 0.5
        low = rng.uniform_range(*cfg.ramp_low)
        mid = rng.uniform_range(*cfg.ramp_mid)
        high = rng.uniform_range(*cfg.ramp_high)
        gap = rng.uniform_range(*cfg.ramp_gap)
        ramps.append((duration, sharpness, ascending, low, mid, high, gap))

    t = lead
    events: list[dict] = []
    for i in range(n):
        intensity, sharpness, gap = taps[i]
        events.append({"kind": "tap", "time": t, "intensity": intensity, "sharpness": sharpness})
        t = t + gap
        for j in range(2):
            if slots[j] != i + 1:
                continue
            d, r_sharp, ascending, low, mid, high, r_gap = ramps[j]
            if ascending:
                values = (low, mid, high)
            else:
                values = (high, mid, low)
            curve = [
                {"time": 0.0, "intensity": values[0]},
                {"time": d / 2.0, "intensity": values[1]},
                {"time": d, "intensity": values[2]},
            ]
            events.append({"kind": "ramp", "time": t, "duration": d, "sharpness": r_sharp, "curve": curve})
            t = t + (d + r_gap)
    return {"version": 1, "seed": seed_to_str(seed), "duration": t, "events": events}


def curve_at(ramp: dict, tau: float) -> float:
    """Ramp intensity at ``tau`` seconds from the ramp start (SPEC 3.3).

    Piecewise-linear through the control points, clamped to the first/last
    value outside them. Exact form: ``k`` = largest index with
    ``curve[k].time <= tau``; if none, ``curve[0].intensity``; if ``k`` is the
    last point, ``curve[last].intensity``; otherwise
    ``w = (tau - t_k) / (t_{k+1} - t_k)`` and
    ``value = c_k + w * (c_{k+1} - c_k)``.
    """
    pts = ramp["curve"]
    if not pts:
        return 0.0
    k = -1
    for idx, p in enumerate(pts):
        if p["time"] <= tau:
            k = idx
    if k < 0:
        return float(pts[0]["intensity"])
    if k == len(pts) - 1:
        return float(pts[-1]["intensity"])
    t0 = pts[k]["time"]
    t1 = pts[k + 1]["time"]
    c0 = pts[k]["intensity"]
    c1 = pts[k + 1]["intensity"]
    w = (tau - t0) / (t1 - t0)
    return c0 + w * (c1 - c0)


def taps(spec: dict) -> list[dict]:
    """Tap events in time (list) order."""
    return [e for e in spec["events"] if e["kind"] == "tap"]


def ramps(spec: dict) -> list[dict]:
    """Ramp events in time (list) order."""
    return [e for e in spec["events"] if e["kind"] == "ramp"]
