"""SPEC section 7 (features, featureVersion 2) and section 8 (calibration)."""

import copy
import math

import numpy as np
import pytest

from auditor_ref import calibration, features, features_v1, quality, synth
from auditor_ref.challenge import curve_at
from auditor_ref.features import CAL_RELATIVE, FEATURE_NAMES
from auditor_ref.signal import process
from auditor_ref.synth import TABLE, SynthOptions

EPS = 1e-12
PAIRS = ((0, 1), (0, 2), (1, 2))


@pytest.fixture(scope="module")
def rec():
    return synth.synth_recording(21, challenge_seed=303)


@pytest.fixture(scope="module")
def proc(rec):
    return process(rec)


@pytest.fixture(scope="module")
def table_cal():
    procs, reps = [], []
    for i in range(20):
        p = process(synth.synth_recording(500 + i, hand=TABLE, challenge_seed=900 + i, purpose="calibration"))
        procs.append(p)
        reps.append(quality.evaluate(p, mode="onTable"))
    return procs, reps


def _spec_tap_ramp(p):
    """Features 0-31 transcribed from the SPEC 7.1-7.3 text with plain loops."""
    ev = p.challenge["events"]
    aH, gH, EA, EG = p.accel_hp, p.gyro_hp, p.energy_accel, p.energy_gyro
    sea = seg = si = 0.0
    tot = {(s, v): [0.0, 0.0, 0.0] for s in "AGX" for v in "BT"}
    for i, e in enumerate(ev):
        if e["kind"] != "tap":
            continue
        o = p.index(e["time"])
        nxt = p.index(ev[i + 1]["time"]) if i + 1 < len(ev) else p.end_index + 25
        u = min(nxt - 1, p.n)
        W = range(max(o - 1, 0), min(o + 9, u))
        if len(W) == 0:
            continue
        sea += sum(float(EA[n]) for n in W)
        seg += sum(float(EG[n]) for n in W)
        si += e["intensity"] * e["intensity"]
        for v, win in (("B", range(max(o - 1, 0), min(o + 2, u))), ("T", range(max(o + 2, 0), min(o + 9, u)))):
            for c in range(3):
                tot[("A", v)][c] += sum(float(aH[c, n] * aH[c, n]) for n in win)
                tot[("G", v)][c] += sum(float(gH[c, n] * gH[c, n]) for n in win)
            for k, (c, d) in enumerate(PAIRS):
                tot[("X", v)][k] += sum(float(aH[c, n] * aH[d, n]) for n in win)
    A = {v: (tot[("A", v)][0] + tot[("A", v)][1]) + tot[("A", v)][2] for v in "BT"}
    G = {v: (tot[("G", v)][0] + tot[("G", v)][1]) + tot[("G", v)][2] for v in "BT"}
    f = [math.log((sea + EPS) / (si + EPS)), math.log((sea + EPS) / (seg + EPS))]
    for v in "BT":
        f += [x / (A[v] + EPS) for x in tot[("A", v)]] + [x / (A[v] + EPS) for x in tot[("X", v)]]
        f += [x / (G[v] + EPS) for x in tot[("G", v)]]
    f.append(math.log((A["T"] + EPS) / (G["T"] + EPS)))
    ri = 0.0
    ra, rg, rx = [0.0] * 3, [0.0] * 3, [0.0] * 3
    for e in ev:
        if e["kind"] != "ramp":
            continue
        s, en = p.clamp(p.index(e["time"])), p.clamp(p.index(e["time"] + e["duration"]))
        C = range(s + 2, en)
        if len(C) == 0:
            continue
        for n in C:
            tau = min(max(float(n - p.lag) / 100.0 - 0.25 - e["time"], 0.0), e["duration"])
            ri += curve_at(e, tau) ** 2
        for c in range(3):
            ra[c] += sum(float(aH[c, n] * aH[c, n]) for n in C)
            rg[c] += sum(float(gH[c, n] * gH[c, n]) for n in C)
        for k, (c, d) in enumerate(PAIRS):
            rx[k] += sum(float(aH[c, n] * aH[d, n]) for n in C)
    RA, RG = (ra[0] + ra[1]) + ra[2], (rg[0] + rg[1]) + rg[2]
    f += [math.log((RA + EPS) / (ri + EPS)), math.log((RA + EPS) / (RG + EPS))]
    f += [x / (RA + EPS) for x in ra] + [x / (RA + EPS) for x in rx] + [x / (RG + EPS) for x in rg]
    return f


def test_names():
    assert features.FEATURE_VERSION == 2 and len(FEATURE_NAMES) == 39 and len(set(FEATURE_NAMES)) == 39
    assert FEATURE_NAMES[0] == "tap_gain_accel" and FEATURE_NAMES[20] == "tap_tail_log_accel_gyro"
    assert FEATURE_NAMES[21] == "ramp_gain_accel" and FEATURE_NAMES[38] == "gravity_z"
    assert CAL_RELATIVE == (0, 1, 20, 21, 22)
    # quiet and grip features are those of featureVersion 1
    assert FEATURE_NAMES[32:] == features_v1.FEATURE_NAMES[29:]


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_matches_spec_transcription(seed):
    p = process(synth.synth_recording(seed, hand=synth.make_hand(seed), challenge_seed=seed + 40))
    f = features.extract_absolute(p)
    assert f.shape == (39,) and np.all(np.isfinite(f))
    np.testing.assert_allclose(f[:32], _spec_tap_ramp(p), rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(f[32:], features_v1.extract_absolute(p)[29:])
    assert np.linalg.norm(f[36:39]) == pytest.approx(1.0, abs=1e-12)


def test_shares_and_cross_bounds(proc):
    f = features.extract_absolute(proc)
    for a in (2, 8, 11, 17, 23, 29):                     # share triples sum to 1
        assert np.all(f[a:a + 3] >= 0) and f[a:a + 3].sum() == pytest.approx(1.0, abs=1e-6)   # eps in the quotient
    for a in (5, 14, 26):                                # |sum xy| <= (sum x^2 + sum y^2) / 2
        assert np.all(np.abs(f[a:a + 3]) <= 0.5 + 1e-12)


def test_tap_windows_follow_spec(proc):
    ev = proc.challenge["events"]
    wins = features.tap_windows(proc)
    assert [w["event"] for w in wins] == [i for i, e in enumerate(ev) if e["kind"] == "tap"]
    for w in wins:
        i, o = w["event"], w["onset"]
        nxt = proc.index(ev[i + 1]["time"]) if i + 1 < len(ev) else proc.end_index + 25
        u = min(nxt - 1, proc.n)
        assert w["full"] == (max(o - 1, 0), min(o + 9, u))
        assert w["burst"] == (max(o - 1, 0), min(o + 2, u)) and w["tail"] == (max(o + 2, 0), min(o + 9, u))
    for w in features.ramp_windows(proc):
        e = proc.challenge["events"][w["event"]]
        assert w["core"] == (proc.clamp(proc.index(e["time"])) + 2, proc.clamp(proc.index(e["time"] + e["duration"])))


def test_accel_and_gyro_scaling(rec, proc):
    k = 3.0
    f0 = features.extract_absolute(proc)
    for stream, sign in (("accel", 1.0), ("gyro", -1.0)):
        r = copy.deepcopy(rec)
        for c in ("x", "y", "z"):
            r[stream][c] = [k * v for v in r[stream][c]]
        f1 = features.extract_absolute(process(r))
        ratios = [1, 20, 22]                             # log accel/gyro ratios
        np.testing.assert_allclose(f1[ratios] - f0[ratios], sign * 2.0 * math.log(k), atol=1e-6)
        gains = [0, 21]
        np.testing.assert_allclose(f1[gains] - f0[gains], 2.0 * math.log(k) if stream == "accel" else 0.0, atol=1e-6)
        shapes = [i for i in range(32) if i not in ratios + gains]
        np.testing.assert_allclose(f1[shapes], f0[shapes], atol=1e-9)


def test_degenerate_recordings_stay_finite(rec):
    for opts in (SynthOptions(drop_gyro=True), SynthOptions(signal_scale=0.0), SynthOptions(gap=(0.0, 3.0))):
        f = features.extract_absolute(process(synth.synth_recording(3, opts=opts, challenge_seed=8)))
        assert np.all(np.isfinite(f))
    r = copy.deepcopy(rec)
    for s in ("accel", "gyro", "motion"):
        for key in r[s]:
            r[s][key] = []
    p = process(r)
    f = features.extract_absolute(p)
    assert np.all(np.isfinite(f))
    # a zero signal: every share, cross product and accel/gyro ratio is 0; the gains are ln(eps / (SI + eps))
    si = sum(e["intensity"] ** 2 for e in p.challenge["events"] if e["kind"] == "tap")
    assert f[0] == pytest.approx(math.log(EPS / (si + EPS)), rel=1e-12) and f[21] < -20
    assert all(f[i] == 0.0 for i in range(32) if i not in (0, 21))


def test_group_fallbacks(rec):
    # without ramps every ramp feature is 0; without taps every tap feature is 0
    for kind, sl in (("ramp", slice(21, 32)), ("tap", slice(0, 21))):
        r = copy.deepcopy(rec)
        r["challenge"]["events"] = [e for e in r["challenge"]["events"] if e["kind"] != kind]
        f = features.extract_absolute(process(r))
        assert np.all(np.isfinite(f)) and np.all(f[sl] == 0.0)


def test_gravity_fallback(rec):
    r = copy.deepcopy(rec)
    n = len(r["motion"]["t"])
    for k in ("gravityX", "gravityY", "gravityZ"):
        r["motion"][k] = [0.0] * n
    f = features.extract_absolute(process(r))
    assert list(f[36:39]) == [0.0, 0.0, -1.0]


def test_calibration_relative(proc):
    table = [0.1 * k for k in range(39)]
    cal = {"tableFeatures": table, "quietGyroRMS": 0.001}
    fa = features.extract_absolute(proc)
    fr = features.extract(proc, cal)
    for k in range(39):
        assert fr[k] == (fa[k] - table[k] if k in CAL_RELATIVE else fa[k])
    np.testing.assert_array_equal(features.extract(proc, None), fa)


def test_calibration_build(table_cal):
    procs, reps = table_cal
    passed = [r["passed"] for r in reps]
    assert sum(passed) >= 12
    cal = calibration.build(procs, reps, "iPhone16,1", "2026-11-24T10:00:00Z")
    assert cal["patternsUsed"] == sum(passed) and cal["version"] == 1 and len(cal["tableFeatures"]) == 39
    used = [i for i, ok in enumerate(passed) if ok]
    feats = np.array([features.extract_absolute(procs[i]) for i in used])
    np.testing.assert_allclose(cal["tableFeatures"], np.median(feats, axis=0), rtol=1e-12)
    assert cal["quietGyroRMS"] == pytest.approx(np.median([reps[i]["metrics"]["quietGyroRMS"] for i in used]), rel=1e-12)
    assert cal["createdAt"] == "2026-11-24T10:00:00Z" and cal["device"] == "iPhone16,1"
    assert set(cal) == {"version", "createdAt", "tableFeatures", "quietGyroRMS", "quietAccelRMS", "sampleRate",
                        "ncc", "snr", "patternsUsed", "device"}


def test_calibration_fails_with_too_few_passing(table_cal):
    procs, reps = table_cal
    bad = [dict(r, passed=(i < 11)) for i, r in enumerate(reps)]
    assert calibration.build(procs, bad, "x", "2026-11-24T10:00:00Z") is None
    ok = [dict(r, passed=(i < 12)) for i, r in enumerate(reps)]
    assert calibration.build(procs, ok, "x", "2026-11-24T10:00:00Z")["patternsUsed"] == 12
