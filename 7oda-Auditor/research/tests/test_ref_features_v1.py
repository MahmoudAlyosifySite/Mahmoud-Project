"""Feature set v1 (featureVersion 1, d = 36; ``auditor_ref.features_v1``), kept for comparisons."""

import copy
import math

import numpy as np
import pytest

from auditor_ref import features_v1 as features
from auditor_ref import synth
from auditor_ref.features_v1 import CAL_RELATIVE, FEATURE_NAMES, ls_slope, ridge
from auditor_ref.signal import process
from auditor_ref.synth import SynthOptions


@pytest.fixture(scope="module")
def rec():
    return synth.synth_recording(21, challenge_seed=303)


@pytest.fixture(scope="module")
def proc(rec):
    return process(rec)


def test_names():
    assert len(FEATURE_NAMES) == 36 and len(set(FEATURE_NAMES)) == 36
    assert FEATURE_NAMES[14] == "tap_log_accel_gyro_ratio" and FEATURE_NAMES[35] == "gravity_z"
    assert CAL_RELATIVE == (0, 1, 2, 3, 4, 5, 6, 7, 14, 17, 18, 19, 20)


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_finite_and_gravity_unit(seed):
    p = process(synth.synth_recording(seed, hand=synth.make_hand(seed), challenge_seed=seed + 40))
    f = features.extract_absolute(p)
    assert f.shape == (36,) and np.all(np.isfinite(f))
    assert np.linalg.norm(f[33:36]) == pytest.approx(1.0, abs=1e-12)
    assert np.all(f[8:14] >= 0) and np.all(f[8:14] <= 1)


def test_degenerate_recordings_stay_finite():
    for opts in (SynthOptions(drop_gyro=True), SynthOptions(signal_scale=0.0), SynthOptions(gap=(0.0, 3.0))):
        f = features.extract_absolute(process(synth.synth_recording(3, opts=opts, challenge_seed=8)))
        assert np.all(np.isfinite(f))


def test_gravity_fallback(rec):
    r = copy.deepcopy(rec)
    n = len(r["motion"]["t"])
    for k in ("gravityX", "gravityY", "gravityZ"):
        r["motion"][k] = [0.0] * n
    f = features.extract_absolute(process(r))
    assert list(f[33:36]) == [0.0, 0.0, -1.0]


def test_accel_scaling_shifts_alpha(rec, proc):
    k = 3.0
    r = copy.deepcopy(rec)
    for c in ("x", "y", "z"):
        r["accel"][c] = [k * v for v in r["accel"][c]]
    f0 = features.extract_absolute(proc)
    f1 = features.extract_absolute(process(r))
    np.testing.assert_allclose(f1[0:3] - f0[0:3], 2.0 * math.log(k), atol=0.02)
    # the envelope is invariant (EA/nA), so gyro features do not move
    np.testing.assert_allclose(f1[3:6], f0[3:6], atol=1e-9)
    assert f1[17] - f0[17] == pytest.approx(math.log(k), abs=0.05)


def test_tap_windows_follow_spec(proc):
    meas = features.tap_measurements(proc)
    ev = proc.challenge["events"]
    env = proc.envelope
    assert meas
    for m in meas:
        i = m["event"]
        o = proc.index(ev[i]["time"])
        nxt = proc.index(ev[i + 1]["time"]) if i + 1 < len(ev) else proc.end_index + 25
        ws, we = m["window"]
        assert (ws, we) == (max(o - 1, 0), min(o + 9, nxt - 1, proc.n))
        pe = min(o + 3, we)
        assert m["peak"] == ws + int(np.argmax(env[ws:pe]))
        p = m["peak"]
        below = [k for k in range(we - p) if env[p + k] < 0.1 * env[p]]
        assert m["settle"] == (below[0] if below else we - p)
        pts = np.arange(p, min(p + 4, we - 1) + 1)
        if pts.size >= 3:
            slope = np.polyfit(pts - p, np.log(env[pts] + 0.1), 1)[0]
            assert m["decay"] == pytest.approx(-slope, rel=1e-9, abs=1e-12)
        e = m["energies"]
        assert e[0] == pytest.approx(np.sum(proc.accel_hp[0, ws:we] ** 2), rel=1e-12)
        assert e[5] == pytest.approx(np.sum(proc.gyro_hp[2, ws:we] ** 2), rel=1e-12)


def test_tap_features_by_hand(proc):
    meas = features.tap_measurements(proc)
    f = features.extract_absolute(proc)
    X = np.array([[1.0, math.log(m["intensity"]), m["sharpness"] - 0.5] for m in meas])
    P = np.diag([0.0, 0.01, 0.01])
    for c in range(6):
        y = np.log(np.array([m["energies"][c] for m in meas]) + 1e-7)
        theta = np.linalg.solve(X.T @ X + P, X.T @ y)
        assert f[c] == pytest.approx(theta[0], rel=1e-10, abs=1e-12)
    ea = np.array([m["EA"] for m in meas])
    eg = np.array([m["EG"] for m in meas])
    theta = np.linalg.solve(X.T @ X + P, X.T @ np.log(ea / 1e-4 + eg / 1e-4 + 1e-6))
    assert f[6] == pytest.approx(theta[1], rel=1e-10) and f[7] == pytest.approx(theta[2], rel=1e-10)
    assert f[8] == pytest.approx(np.median([m["energies"][0] / (m["EA"] + 1e-12) for m in meas]), rel=1e-12)
    assert f[13] == pytest.approx(np.median([m["energies"][5] / (m["EG"] + 1e-12) for m in meas]), rel=1e-12)
    assert f[14] == pytest.approx(np.median(np.log((ea + 1e-7) / (eg + 1e-7))), rel=1e-12)
    assert f[15] == pytest.approx(np.median([m["decay"] for m in meas]), rel=1e-12)
    assert f[16] == pytest.approx(np.mean([m["settle"] for m in meas]), rel=1e-12)
    assert features.tap_features([]) == [0.0] * 17


def test_ramp_features_by_hand(proc):
    meas = features.ramp_measurements(proc)
    assert len(meas) == 2
    amp_a = np.sqrt(proc.energy_accel)
    for m in meas:
        ev = proc.challenge["events"][m["event"]]
        s, e = m["range"]
        assert (s, e) == (proc.clamp(proc.index(ev["time"])), proc.clamp(proc.index(ev["time"] + ev["duration"])))
        idx = np.arange(s + 2, e)
        tau = np.clip((idx - proc.lag) / 100.0 - 0.25 - ev["time"], 0.0, ev["duration"])
        c = ev["curve"]
        u = np.log(np.interp(tau, [p["time"] for p in c], [p["intensity"] for p in c]))
        X = np.column_stack([np.ones_like(u), u])
        theta = np.linalg.solve(X.T @ X + np.diag([0.0, 0.001]), X.T @ np.log(amp_a[idx] + 1e-4))
        assert (m["rhoA"], m["kappaA"]) == pytest.approx(tuple(theta), rel=1e-9)
        seg = amp_a[s:e]
        assert m["rise"] == int(np.argmax(seg >= 0.63 * seg.max()))
        ref = (amp_a[e - 2] + amp_a[e - 1]) / 2.0
        after = [k for k in range(10) if e + k < proc.n and amp_a[e + k] < 0.37 * ref]
        assert m["decayT"] == (after[0] if after else 10)
        den = proc.energy_accel[s:e].sum() + 1e-12
        assert m["sharesA"][1] == pytest.approx((proc.accel_hp[1, s:e] ** 2).sum() / den, rel=1e-12)


def test_quiet_features(proc):
    f = features.extract_absolute(proc)
    q0, q1 = proc.quiet_range
    ba, bg = features.tremor_bandpassed(proc)
    assert f[29] == pytest.approx(math.log(np.mean((ba[:, q0:q1] ** 2).sum(axis=0)) + 1e-10), rel=1e-12)
    assert f[30] == pytest.approx(math.log(np.mean((bg[:, q0:q1] ** 2).sum(axis=0)) + 1e-10), rel=1e-12)
    assert f[31] == pytest.approx(math.log(math.sqrt(np.mean(proc.energy_accel[q0:q1])) + 1e-6), rel=1e-12)
    g = proc.gravity.mean(axis=1)
    np.testing.assert_allclose(f[33:36], g / np.linalg.norm(g), atol=1e-12)


def test_ridge_and_slope_helpers():
    rows = [[1.0, 0.1 * i, (-1) ** i * 0.3] for i in range(6)]
    y = [0.5 + 0.2 * i for i in range(6)]
    X = np.array(rows)
    theta = np.linalg.solve(X.T @ X + np.diag([0.0, 0.01, 0.01]), X.T @ np.array(y))
    np.testing.assert_allclose(ridge(rows, y, 0.01), theta, rtol=1e-12)
    # intercept is not penalised: a huge lambda drives the slopes to 0, the intercept to mean(y)
    big = ridge(rows, y, 1e12)
    assert big[0] == pytest.approx(np.mean(y), rel=1e-9) and abs(big[1]) < 1e-9
    assert ls_slope([1.0, 3.0, 5.0, 7.0]) == pytest.approx(2.0)


def test_calibration_relative(proc):
    table = [0.1 * k for k in range(36)]
    cal = {"tableFeatures": table, "quietGyroRMS": 0.001}
    fa = features.extract_absolute(proc)
    fr = features.extract(proc, cal)
    for k in range(36):
        assert fr[k] == (fa[k] - table[k] if k in CAL_RELATIVE else fa[k])
    np.testing.assert_array_equal(features.extract(proc, None), fa)
