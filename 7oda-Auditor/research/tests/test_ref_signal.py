"""SPEC section 5: grid, cleaning, envelope, template, alignment and ranges."""

import copy
import math

import numpy as np
import pytest

from auditor_ref import signal, synth
from auditor_ref.challenge import curve_at, generate
from auditor_ref.recording import sort_dedup
from auditor_ref.signal import Alignment, SignalConfig, process


@pytest.fixture(scope="module")
def rec():
    return synth.synth_recording(11, challenge_seed=4242)


@pytest.fixture(scope="module")
def proc(rec):
    return process(rec)


def test_grid(rec, proc):
    cfg = SignalConfig()
    dur = rec["challenge"]["duration"]
    assert proc.n == int((0.25 + dur + 0.25) * 100.0 + 0.5)
    assert proc.grid_start == rec["hapticStart"] - 0.25
    assert proc.grid[7] == proc.grid_start + 7.0 / 100.0
    assert proc.accel.shape == (3, proc.n) and proc.rotation.shape == (3, proc.n)
    assert cfg.tremor_f0 == 9.797958971132712


@pytest.mark.parametrize("latency", [0.0, 0.03, 0.065, 0.12, 0.2])
def test_alignment_recovers_latency(latency):
    r = synth.synth_recording(5, opts=synth.SynthOptions(latency=latency), challenge_seed=77)
    p = process(r)
    assert abs(p.lag - latency * 100.0) <= 1.0
    assert p.ncc > 0.5 and p.peak_margin > 0.05


def test_envelope_normalisation(proc):
    pre_end = 20
    np.testing.assert_allclose(proc.energy_accel, (proc.accel_hp ** 2).sum(axis=0), rtol=1e-13)
    assert proc.noise_accel == pytest.approx(np.mean(proc.energy_accel[:pre_end]), rel=1e-12)
    assert proc.noise_gyro == pytest.approx(np.mean(proc.energy_gyro[:pre_end]), rel=1e-12)
    np.testing.assert_allclose(proc.envelope, proc.energy_accel / proc.noise_accel + proc.energy_gyro / proc.noise_gyro, rtol=1e-15)
    assert np.mean(proc.envelope[:pre_end]) == pytest.approx(2.0, rel=1e-12)
    np.testing.assert_allclose(proc.amplitude, np.sqrt(proc.envelope))


def test_noise_floor_used_for_silent_streams():
    r = synth.synth_recording(3, opts=synth.SynthOptions(drop_gyro=True), challenge_seed=5)
    p = process(r)
    assert p.noise_gyro == 1e-6
    assert np.all(p.energy_gyro == 0.0)
    assert p.gap_fraction == 1.0


def test_template(proc):
    ch = proc.challenge
    cfg = proc.config
    h = proc.template
    t0 = ch["events"][0]["time"]
    for k in range(proc.n):
        tau = k / cfg.fs - cfg.pre_roll
        if tau < t0:
            assert h[k] == 0.0
    assert np.all(h >= 0.0)
    # brute-force re-evaluation
    for k in (40, 77, 120, proc.n - 1):
        tau = k / cfg.fs - cfg.pre_roll
        val = 0.0
        for e in ch["events"]:
            if e["kind"] == "tap" and tau >= e["time"]:
                val += e["intensity"] * math.exp(-(tau - e["time"]) / 0.03)
            elif e["kind"] == "ramp":
                if e["time"] <= tau < e["time"] + e["duration"]:
                    val += curve_at(e, tau - e["time"])
                elif tau >= e["time"] + e["duration"]:
                    val += curve_at(e, e["duration"]) * math.exp(-(tau - e["time"] - e["duration"]) / 0.03)
        assert h[k] == pytest.approx(val, rel=1e-12, abs=1e-15)


def test_alignment_definition(proc):
    x, h = proc.amplitude, proc.template
    n = proc.n
    for lag in (-10, -3, 0, proc.lag, 25):
        lo, hi = max(0, lag), min(n, n + lag)
        ref = np.corrcoef(x[lo:hi], h[lo - lag:hi - lag])[0, 1]
        assert proc.alignment.ncc_by_lag[lag] == pytest.approx(ref, abs=1e-12)
    vals = proc.alignment.ncc_by_lag
    assert proc.ncc == max(vals.values())
    far = [v for L, v in vals.items() if abs(L - proc.lag) >= 4]
    assert proc.peak_margin == proc.ncc - max(far)


def test_alignment_tie_breaking():
    cfg = SignalConfig()
    # constant envelope: every lag has ncc 0 -> lag 0
    a = signal.align(np.ones(150), np.arange(150, dtype=float), cfg)
    assert a.lag == 0 and a.ncc == 0.0 and a.peak_margin == 0.0
    lag, ncc, pm = signal.select_lag({-3: 0.4, -1: 0.9, 0: 0.2, 1: 0.9, 3: 0.7, 5: 0.6})
    assert (lag, ncc) == (-1, 0.9) and pm == 0.9 - 0.7
    lag, ncc, pm = signal.select_lag({-2: 0.9, 0: 0.1, 2: 0.9})
    assert lag == -2 and pm == 0.9 - 0.9
    lag, ncc, pm = signal.select_lag({4: 0.3, 5: 0.8, 6: 0.3})
    assert lag == 5 and pm == 0.8  # no lag at distance >= 4


def test_short_overlap_gives_zero():
    cfg = SignalConfig(lag_min=-12, lag_max=12)
    x = np.arange(20, dtype=float)
    a = signal.align(x, x.copy(), cfg)
    assert a.ncc_by_lag[11] == 0.0 and a.ncc_by_lag[-11] == 0.0
    assert a.ncc_by_lag[10] != 0.0 and a.lag == 0


def test_ranges(proc):
    ch = proc.challenge
    onset = int((0.25 + ch["events"][0]["time"]) * 100 + 0.5) + proc.lag
    end = int((0.25 + ch["duration"]) * 100 + 0.5) + proc.lag
    assert proc.index(ch["events"][0]["time"]) == onset
    assert proc.quiet_range == (0, onset - 2)
    assert proc.event_range == (onset, min(end, proc.n))


def test_quiet_window_fallback(proc):
    p = copy.copy(proc)
    p.alignment = Alignment(lag=-40, ncc=0.0, peak_margin=0.0)
    assert p.onset_first - 2 < 10
    assert p.quiet_range == (max(0, min(p.end_index + 5, p.n - 10)), p.n)
    assert p.quiet_range[1] - p.quiet_range[0] >= 10
    assert p.event_range == (p.clamp(p.onset_first), p.clamp(p.end_index))


def test_sort_dedup():
    s = {"t": [3.0, 1.0, 2.0, 1.0, 3.0], "x": [30, 10, 20, 11, 31], "y": [0, 0, 0, 0, 0], "z": [1, 2, 3, 4, 5]}
    out = sort_dedup(s)
    assert list(out["t"]) == [1.0, 2.0, 3.0]
    assert list(out["x"]) == [10, 20, 30]  # first occurrence of each duplicate (stable)
    assert list(out["z"]) == [2, 3, 1]


def test_unsorted_and_duplicates_are_cleaned(rec, proc):
    r = copy.deepcopy(rec)
    for key in ("accel", "gyro", "motion"):
        st = r[key]
        n = len(st["t"])
        perm = list(range(n))[::-1]
        for k in st:
            st[k] = [st[k][i] for i in perm]
        # exact-time duplicates appended after the originals are dropped
        for k in st:
            st[k] = st[k] + [st[k][i] if k == "t" else st[k][i] + 1.0 for i in range(0, n, 7)]
    p = process(r)
    np.testing.assert_array_equal(p.accel, proc.accel)
    np.testing.assert_array_equal(p.gyro, proc.gyro)
    np.testing.assert_array_equal(p.gravity, proc.gravity)
    assert p.lag == proc.lag and p.ncc == proc.ncc
    assert p.sample_rate == proc.sample_rate


def test_sample_rate_and_gap_fraction():
    r = synth.synth_recording(4, opts=synth.SynthOptions(gap=(0.6, 0.3)), challenge_seed=9)
    p = process(r)
    ta = sort_dedup(r["accel"])["t"]
    assert p.sample_rate == 1.0 / float(np.median(np.diff(ta)))
    miss = p.accel_missing | p.gyro_missing
    assert p.gap_fraction == np.count_nonzero(miss) / p.n
    assert 0.12 < p.gap_fraction < 0.25


def test_empty_streams():
    r = synth.synth_recording(4, challenge_seed=9)
    r["accel"] = {"t": [], "x": [], "y": [], "z": []}
    p = process(r)
    assert np.all(p.accel == 0.0) and p.sample_rate == 0.0 and p.gap_fraction == 1.0
    assert np.all(np.isfinite(p.envelope))


def test_speed(rec):
    import time
    t = time.perf_counter()
    for _ in range(5):
        process(rec)
    assert (time.perf_counter() - t) / 5 < 0.1
