"""SPEC section 3: challenge generation, structure, ranges and the ramp curve."""

import math

import numpy as np
import pytest

from auditor_ref.challenge import curve_at, generate
from auditor_ref.rng import MASK64, Xoshiro256

SEEDS = [0, 1, 2, 3, 1234, MASK64, 0x8000000000000000, 0xDEADBEEFCAFEBABE] + list(range(100, 160))


def test_determinism():
    for s in SEEDS[:10]:
        assert generate(s) == generate(s)
    assert generate(1) != generate(2)


@pytest.mark.parametrize("seed", SEEDS)
def test_structure(seed):
    spec = generate(seed)
    ev = spec["events"]
    taps = [e for e in ev if e["kind"] == "tap"]
    ramps = [e for e in ev if e["kind"] == "ramp"]
    assert spec["version"] == 1
    assert 6 <= len(taps) <= 8 and len(ramps) == 2
    assert ev[0]["kind"] == "tap"
    for a, b in zip(ev, ev[1:]):
        assert b["time"] > a["time"]
        assert not (a["kind"] == "ramp" and b["kind"] == "ramp")
    assert spec["duration"] > ev[-1]["time"]
    assert set(taps[0]) == {"kind", "time", "intensity", "sharpness"}
    assert set(ramps[0]) == {"kind", "time", "duration", "sharpness", "curve"}


@pytest.mark.parametrize("seed", SEEDS)
def test_parameter_ranges_and_duration(seed):
    spec = generate(seed)
    ev = spec["events"]
    assert 0.05 <= ev[0]["time"] < 0.10
    t = ev[0]["time"]
    for i, e in enumerate(ev):
        assert e["time"] == t
        nxt = ev[i + 1]["time"] if i + 1 < len(ev) else spec["duration"]
        if e["kind"] == "tap":
            assert 0.4 <= e["intensity"] < 1.0
            assert 0.0 <= e["sharpness"] < 1.0
            gap = nxt - e["time"]
            assert 0.08 - 1e-12 <= gap < 0.15 + 1e-12
        else:
            d = e["duration"]
            assert 0.15 <= d < 0.25
            assert 0.2 <= e["sharpness"] < 0.8
            c = e["curve"]
            assert [p["time"] for p in c] == [0.0, d / 2.0, d]
            lo, mid, hi = (c[0]["intensity"], c[1]["intensity"], c[2]["intensity"])
            if lo > hi:
                lo, hi = hi, lo
            assert 0.25 <= lo < 0.5 and 0.25 <= mid < 1.0 and 0.75 <= hi < 1.0
            rgap = nxt - (e["time"] + d)
            assert 0.08 - 1e-12 <= rgap < 0.12 + 1e-12
        t = nxt


def test_duration_distribution():
    durs = [generate(s)["duration"] for s in range(2000)]
    assert 1.3 < float(np.median(durs)) < 1.6
    assert min(durs) > 0.95 and max(durs) < 2.0


def test_slot_coverage():
    seen_last = seen_first = asc = desc = False
    counts = set()
    for s in range(300):
        ev = generate(s)["events"]
        counts.add(sum(e["kind"] == "tap" for e in ev))
        kinds = [e["kind"] for e in ev]
        seen_last |= kinds[-1] == "ramp"
        seen_first |= kinds[1] == "ramp"
        for e in ev:
            if e["kind"] == "ramp":
                if e["curve"][0]["intensity"] < e["curve"][2]["intensity"]:
                    asc = True
                else:
                    desc = True
    assert counts == {6, 7, 8} and seen_last and seen_first and asc and desc


def test_replay_draw_order_by_hand():
    seed = 0x00000000000004D2
    r = Xoshiro256(seed)
    lead = 0.05 + (0.10 - 0.05) * r.uniform()
    n = 6 + min(int(math.floor(r.uniform() * 3.0)), 2)
    s1 = 1 + min(int(math.floor(r.uniform() * n)), n - 1)
    s2 = 1 + min(int(math.floor(r.uniform() * (n - 1))), n - 2)
    if s2 >= s1:
        s2 += 1
    slots = sorted([s1, s2])
    taps = [(0.4 + 0.6 * r.uniform(), 0.0 + 1.0 * r.uniform(), 0.08 + (0.15 - 0.08) * r.uniform()) for _ in range(n)]
    ramps = []
    for _ in range(2):
        d = 0.15 + (0.25 - 0.15) * r.uniform()
        sh = 0.2 + (0.8 - 0.2) * r.uniform()
        asc = r.uniform() < 0.5
        lo = 0.25 + (0.5 - 0.25) * r.uniform()
        mid = 0.25 + (1.0 - 0.25) * r.uniform()
        hi = 0.75 + (1.0 - 0.75) * r.uniform()
        gap = 0.08 + (0.12 - 0.08) * r.uniform()
        ramps.append((d, sh, asc, lo, mid, hi, gap))
    t = lead
    expected = []
    for i in range(n):
        expected.append({"kind": "tap", "time": t, "intensity": taps[i][0], "sharpness": taps[i][1]})
        t += taps[i][2]
        for j in range(2):
            if slots[j] == i + 1:
                d, sh, asc, lo, mid, hi, gap = ramps[j]
                vals = (lo, mid, hi) if asc else (hi, mid, lo)
                expected.append({"kind": "ramp", "time": t, "duration": d, "sharpness": sh,
                                 "curve": [{"time": 0.0, "intensity": vals[0]},
                                           {"time": d / 2, "intensity": vals[1]},
                                           {"time": d, "intensity": vals[2]}]})
                t += d + gap
    spec = generate(seed)
    assert spec["seed"] == "0x00000000000004d2"
    assert spec["events"] == expected
    assert spec["duration"] == t


def test_curve_at():
    ramp = {"curve": [{"time": 0.0, "intensity": 0.3}, {"time": 0.1, "intensity": 0.7}, {"time": 0.2, "intensity": 0.9}]}
    assert curve_at(ramp, -1.0) == 0.3
    assert curve_at(ramp, 0.0) == 0.3
    assert curve_at(ramp, 0.1) == 0.7
    assert curve_at(ramp, 0.2) == 0.9
    assert curve_at(ramp, 5.0) == 0.9
    assert curve_at(ramp, 0.05) == pytest.approx(0.5, abs=1e-15)
    assert curve_at(ramp, 0.15) == pytest.approx(0.8, abs=1e-15)
    w = (0.13 - 0.1) / (0.2 - 0.1)
    assert curve_at(ramp, 0.13) == 0.7 + w * (0.9 - 0.7)
