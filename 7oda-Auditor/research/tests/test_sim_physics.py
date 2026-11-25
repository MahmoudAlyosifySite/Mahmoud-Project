"""Sanity tests of the physics simulator (fast)."""
import copy
import datetime as dt
import json

import numpy as np
import pytest

from auditor_ref import quality, signal
from auditor_ref.challenge import generate
from sim import make_phone, make_population, simulate, simulate_batch
from sim import actuator, dynamics
from sim.imu import quat_to_gravity
from sim.params import realize


@pytest.fixture(scope="module")
def people():
    return make_population(6, seed=11)


def _rng(s=0):
    return np.random.Generator(np.random.PCG64(s))


def test_determinism_and_batch_independence(people):
    ph = make_phone(5, "none")
    a = simulate_batch(people[:3], "right", [1, 2, 3], ph, _rng(1), seeds=[10, 11, 12])
    b = simulate_batch(people[:3], "right", [1, 2, 3], ph, _rng(99), seeds=[10, 11, 12])
    assert json.dumps(a) == json.dumps(b)
    # a trial depends only on its inputs and seed, not on the rest of the batch
    c = simulate_batch([people[5], people[1]], ["left", "right"], [9, 2], ph, _rng(2), seeds=[77, 11])
    assert json.dumps(c[1]) == json.dumps(a[1])
    d = simulate(people[0], "right", 1, ph, _rng(1))
    e = simulate(people[0], "right", 1, ph, _rng(1))
    assert json.dumps(d) == json.dumps(e)
    assert json.dumps(simulate(people[0], "right", 1, ph, _rng(2))) != json.dumps(d)


def test_recording_form(people):
    ph = make_phone(5, "lpf")
    rec = simulate(people[2], "left", 42, ph, _rng(3), posture="standing", accessory="watch", session=2)
    for key in ("schema", "id", "createdAt", "purpose", "label", "challenge", "hapticStart", "accel", "gyro",
                "motion", "device"):
        assert key in rec
    assert rec["schema"] == "7oda-auditor.recording.v1"
    dt.datetime.strptime(rec["createdAt"], "%Y-%m-%dT%H:%M:%SZ")
    assert rec["label"] == {"personID": people[2].pid, "personName": people[2].name, "hand": "left",
                            "accessory": "watch", "posture": "standing", "sessionID": f"{people[2].pid}-S2"}
    assert rec["challenge"] == generate(42)
    for s in ("accel", "gyro"):
        assert len({len(v) for v in rec[s].values()}) == 1
    assert len({len(v) for v in rec["motion"].values()}) == 1
    t = np.array(rec["accel"]["t"])
    hs, dur = rec["hapticStart"], rec["challenge"]["duration"]
    assert t.min() >= hs - 0.35 - 1e-9 and t.max() <= hs + dur + 0.35 + 1e-9
    m = rec["motion"]
    q = np.array([m["attitudeW"], m["attitudeX"], m["attitudeY"], m["attitudeZ"]]).T
    g = np.array([m["gravityX"], m["gravityY"], m["gravityZ"]]).T
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1, atol=1e-9)
    np.testing.assert_allclose(quat_to_gravity(q), g, atol=1e-9)
    acc = np.array([rec["accel"][c] for c in "xyz"]).mean(1)
    np.testing.assert_allclose(acc, g.mean(0), atol=0.03)      # accel = gravity + small user acceleration
    json.dumps(rec)
    tab = simulate(None, "right", 5, ph, _rng(4), condition="onTable")
    assert tab["label"] is None and tab["purpose"] == "calibration"
    assert np.mean(tab["motion"]["gravityZ"]) < -0.999


def _ringdown(contacts_scale_c, ph, person):
    """Gyro/accel energy after a single tap relative to its peak window."""
    tp = realize(person, "right", ph, _rng(5))
    tp.contacts.cn *= contacts_scale_c
    tp.contacts.ct *= contacts_scale_c
    ch = {"version": 1, "seed": "0x0", "duration": 0.3,
          "events": [{"kind": "tap", "time": 0.05, "intensity": 1.0, "sharpness": 0.5}]}
    h = 1 / 2000
    T = int(0.6 / h)
    u = actuator.force_waveform([ch], [0.0], ph, 0.0, h / 2, 2 * T + 1, zoh_step=h)
    y = dynamics.integrate(dynamics.discretize(dynamics.build_system([tp.contacts], ph), h),
                           np.ascontiguousarray(u[:, 0:2 * T:2].T), np.ascontiguousarray(u[:, 1:2 * T:2].T))
    e = (y[:, 0, :3].astype(float) ** 2).sum(1)
    k0 = int(0.05 / h)
    return e[k0 + int(0.04 / h): k0 + int(0.25 / h)].sum() / e[k0: k0 + int(0.04 / h)].sum()


def test_more_damping_faster_decay(people):
    ph = make_phone(5, "none")
    r1 = _ringdown(1.0, ph, people[0])
    r3 = _ringdown(3.0, ph, people[0])
    assert r3 < 0.6 * r1


def test_fast_integrator_matches_rk4(people):
    ph = make_phone(4, "none")
    tp = realize(people[1], "left", ph, _rng(2))
    sysm = dynamics.build_system([tp.contacts], ph)
    ch = generate(7)
    ch = {**ch, "events": ch["events"][:4], "duration": ch["events"][3]["time"] + 0.1}
    h, T = 1 / 2000, int((ch["duration"] + 0.4) / (1 / 2000))
    u = actuator.force_waveform([ch], [0.02], ph, -0.1, h / 2, 2 * T + 1, zoh_step=h)
    y = dynamics.integrate(dynamics.discretize(sysm, h), np.ascontiguousarray(u[:, 0:2 * T:2].T),
                           np.ascontiguousarray(u[:, 1:2 * T:2].T))[:, 0, :].T
    hr = h / 2
    ur = actuator.force_waveform([ch], [0.02], ph, -0.1, hr / 2, 4 * T + 1)[0]
    yr = dynamics.integrate_reference(sysm, ur, hr, 2)
    n = min(y.shape[1], yr.shape[1])
    err = np.sqrt(((y[:, :n] - yr[:, :n]) ** 2).sum(1) / (yr[:, :n] ** 2).sum(1))
    assert err.max() < 0.06


def _single_ramp(sharpness, d=0.6):
    return {"version": 1, "seed": "0x0", "duration": d + 0.1,
            "events": [{"kind": "ramp", "time": 0.05, "duration": d, "sharpness": sharpness,
                        "curve": [{"time": 0.0, "intensity": 1.0}, {"time": d / 2, "intensity": 1.0},
                                  {"time": d, "intensity": 1.0}]}]}


def _ramp_spectrum(ph, person, sharpness):
    rec = simulate(person, "right", _single_ramp(sharpness), ph, _rng(6), fidget=False)
    t = np.array(rec["accel"]["t"]) - rec["hapticStart"]
    x = np.array(rec["accel"]["x"])
    sel = (t > 0.15) & (t < 0.6)
    x = x[sel] - x[sel].mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size), 512)) ** 2
    f = np.fft.rfftfreq(512, 0.01)
    return f, spec


def test_aliasing_folds_the_carrier(people):
    ph = make_phone(8, "none")
    sharp = 0.5
    fc = ph.f0 * (0.7 + 0.6 * sharp)
    folded = abs(fc - 100.0 * round(fc / 100.0))
    if folded < 8:                       # keep away from DC (removed by the processing anyway)
        sharp = 0.2
        fc = ph.f0 * (0.7 + 0.6 * sharp)
        folded = abs(fc - 100.0 * round(fc / 100.0))
    f, s = _ramp_spectrum(ph, people[0], sharp)
    band = (f > 4)
    peak = f[band][np.argmax(s[band])]
    assert abs(peak - folded) < 2.5, (peak, folded, fc)
    near = s[(np.abs(f - folded) < 2.5)].sum()
    f2, s2 = _ramp_spectrum(make_phone(8, "lpf"), people[0], sharp)
    near_lpf = s2[(np.abs(f2 - folded) < 2.5)].sum()
    assert near_lpf < near / 10


def test_latency_shifts_alignment_lag(people):
    lags = {}
    for lat in (0.010, 0.040):
        ph = make_phone(5, "none", latency_mean=lat, latency_sd=0.0)
        recs = simulate_batch([people[i % 6] for i in range(12)], "right", list(range(200, 212)), ph, _rng(7),
                              seeds=list(range(12)), fidget=False)
        lags[lat] = np.median([signal.process(r).alignment.lag for r in recs])
    assert 2 <= lags[0.040] - lags[0.010] <= 4


def test_gaps_jitter_duplicates_at_configured_rates(people):
    ph = make_phone(5, "none", drop_rate=0.05, burst_rate=0.0, long_gap_rate=0.0, dup_rate=0.02,
                    timestamp_jitter=0.0005)
    recs = simulate_batch(people[0], "right", list(range(8)), ph, _rng(8))
    n_exp, n_got, dups, d_jit = 0, 0, 0, []
    for r in recs:
        t = np.array(r["accel"]["t"])
        u = np.unique(t)
        dups += t.size - u.size
        n_got += u.size
        n_exp += (r["challenge"]["duration"] + 0.7) * ph.fs_imu
        d = np.diff(u)
        d_jit += list(d[d < 0.015])
    miss = 1 - n_got / n_exp
    assert 0.03 < miss < 0.07
    assert 0.01 < dups / n_got < 0.03
    assert abs(np.std(d_jit) - np.sqrt(2) * 0.0005) < 0.0002


@pytest.mark.parametrize("mode,min_pass", [("none", 0.85), ("lpf", 0.6)])
def test_quality_gates_mostly_pass(people, mode, min_pass):
    ph = make_phone(12, mode)
    tab = simulate_batch(None, "right", list(range(100, 112)), ph, _rng(9), conditions="onTable")
    tq = [quality.evaluate(signal.process(r), None, "onTable") for r in tab]
    assert np.mean([q["passed"] for q in tq]) >= 0.8
    cal = {"quietGyroRMS": float(np.median([q["metrics"]["quietGyroRMS"] for q in tq]))}
    recs = simulate_batch([people[i % 6] for i in range(24)], ["right", "left"] * 12, list(range(24)), ph, _rng(10),
                          postures=["sitting", "standing"] * 12)
    hq = [quality.evaluate(signal.process(r), cal, "handheld") for r in recs]
    assert np.mean([q["passed"] for q in hq]) >= min_pass
    # a handheld recording judged as if on a table is rejected, and vice versa
    assert not any(quality.evaluate(signal.process(r), None, "onTable")["passed"] for r in recs[:6])
    assert not any(quality.evaluate(signal.process(r), cal, "handheld")["passed"] for r in tab[:6])


def test_hands_differ_but_roughly_mirror(people):
    p = people[3]
    R, L = p.hands["right"], p.hands["left"]
    mirrored = R.pos * np.array([-1, 1, 1])
    dev = np.abs(L.pos - mirrored).max()
    assert 0 < dev < 0.01
    assert not np.allclose(L.kn, R.kn)
    # gravity x tends to flip sign between hands across people
    sx = [np.sign(np.sin(np.deg2rad(q.hands["right"].angles["sitting"][2])))
          * np.sign(np.sin(np.deg2rad(q.hands["left"].angles["sitting"][2]))) for q in make_population(30, 3)]
    assert np.mean(np.array(sx) < 0) > 0.6


def test_conditions_and_accessories_change_the_response(people):
    ph = make_phone(5, "none")
    base = dict(seeds=[3], fidget=False)
    a = simulate_batch(people[0], "right", [5], ph, _rng(0), accessories="none", grips=1, **base)[0]
    w = simulate_batch(people[0], "right", [5], ph, _rng(0), accessories="watch", grips=1, **base)[0]
    two = simulate_batch(people[0], "right", [5], ph, _rng(0), conditions="twoHanded", grips=1, **base)[0]
    assert two["sim"]["grip"] == "twoHanded"
    ea = np.var(a["accel"]["x"])
    assert abs(np.var(w["accel"]["x"]) - ea) > 1e-3 * ea
    assert abs(np.var(two["accel"]["x"]) - ea) > 1e-3 * ea
