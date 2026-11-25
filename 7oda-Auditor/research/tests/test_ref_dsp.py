"""SPEC 5.2-5.4: biquads, lfilter, filtfilt_z and resampling."""

import math

import numpy as np
import pytest
import scipy.signal as ss

from auditor_ref import dsp
from auditor_ref.rng import Xoshiro256

Q = dsp.Q_BUTTERWORTH


def _ba(c):
    b0, b1, b2, a1, a2 = c
    return np.array([b0, b1, b2]), np.array([1.0, a1, a2])


def _noise(seed, n):
    r = Xoshiro256(seed)
    return np.array([r.normal() for _ in range(n)])


def test_q_constant():
    assert Q == 1.0 / math.sqrt(2.0) == 0.7071067811865475


@pytest.mark.parametrize("kind,fc,fs", [("highpass", 5.0, 100.0), ("lowpass", 3.0, 100.0),
                                        ("highpass", 1.0, 50.0), ("lowpass", 20.0, 200.0),
                                        ("highpass", 12.5, 100.0), ("lowpass", 40.0, 100.0)])
def test_rbj_butterworth_equals_scipy_butter(kind, fc, fs):
    c = dsp.biquad(kind, fc, fs, Q)
    b, a = ss.butter(2, fc, btype="high" if kind == "highpass" else "low", fs=fs)
    np.testing.assert_allclose(c, [b[0] / a[0], b[1] / a[0], b[2] / a[0], a[1] / a[0], a[2] / a[0]], rtol=0, atol=1e-12)


@pytest.mark.parametrize("f0,q,fs", [(math.sqrt(96.0), math.sqrt(96.0) / 4.0, 100.0), (25.0, 2.0, 100.0), (7.0, 0.8, 50.0)])
def test_bandpass_matches_prewarped_bilinear_and_unit_peak(f0, q, fs):
    c = dsp.biquad_bandpass(f0, fs, q)
    # analog prototype H(s) = (w/Q) s / (s^2 + (w/Q) s + w^2) with w pre-warped at f0
    w = 2.0 * fs * math.tan(math.pi * f0 / fs)
    bz, az = ss.bilinear([0.0, w / q, 0.0], [1.0, w / q, w * w], fs=fs)
    bz = np.atleast_1d(bz)[-3:] / az[0]
    az = az / az[0]
    np.testing.assert_allclose(c, [bz[0], bz[1], bz[2], az[1], az[2]], rtol=0, atol=1e-12)
    b, a = _ba(c)
    _, h = ss.freqz(b, a, worN=[f0], fs=fs)
    assert abs(abs(h[0]) - 1.0) < 1e-12
    assert abs(np.angle(h[0])) < 1e-9


def test_dc_gains():
    for c, g in ((dsp.biquad_highpass(5.0, 100.0, Q), 0.0), (dsp.biquad_lowpass(3.0, 100.0, Q), 1.0),
                 (dsp.biquad_bandpass(9.8, 100.0, 2.45), 0.0)):
        b0, b1, b2, a1, a2 = c
        assert (b0 + b1 + b2) / (1.0 + a1 + a2) == pytest.approx(g, abs=1e-12)


def test_coefficient_formula_order():
    fc, fs, q = 5.0, 100.0, Q
    w0 = 2.0 * math.pi * fc / fs
    cw, sw = math.cos(w0), math.sin(w0)
    al = sw / (2.0 * q)
    a0 = 1.0 + al
    assert dsp.biquad_highpass(fc, fs, q) == ((1.0 + cw) / 2.0 / a0, -(1.0 + cw) / a0, (1.0 + cw) / 2.0 / a0,
                                              -2.0 * cw / a0, (1.0 - al) / a0)


@pytest.mark.parametrize("n", [0, 1, 2, 7, 64, 300])
def test_lfilter_equals_scipy(n):
    x = _noise(n + 11, n)
    for c in (dsp.biquad_highpass(5.0, 100.0, Q), dsp.biquad_lowpass(3.0, 100.0, Q), dsp.biquad_bandpass(9.8, 100.0, 2.45)):
        b, a = _ba(c)
        np.testing.assert_allclose(dsp.lfilter(c, x), ss.lfilter(b, a, x) if n else np.zeros(0), rtol=1e-12, atol=1e-13)


def test_lfilter_direct_form_one_by_hand():
    c = (0.5, -0.25, 0.125, -0.3, 0.2)
    x = [1.0, 2.0, -1.0, 0.5]
    y = []
    for i in range(4):
        xm1 = x[i - 1] if i >= 1 else 0.0
        xm2 = x[i - 2] if i >= 2 else 0.0
        ym1 = y[i - 1] if i >= 1 else 0.0
        ym2 = y[i - 2] if i >= 2 else 0.0
        y.append(0.5 * x[i] + -0.25 * xm1 + 0.125 * xm2 - (-0.3) * ym1 - 0.2 * ym2)
    assert list(dsp.lfilter(c, x)) == y


def test_filtfilt_edge_lengths():
    hp = dsp.biquad_highpass(5.0, 100.0, Q)
    lp = dsp.biquad_lowpass(3.0, 100.0, Q)
    assert dsp.filtfilt_z(hp, [], 30, 0.0).size == 0
    assert list(dsp.filtfilt_z(hp, [2.5], 30, 0.0)) == [0.0]
    assert list(dsp.filtfilt_z(lp, [2.5], 30, 1.0)) == [2.5]
    for n in (2, 3, 5, 17, 30, 31):
        x = _noise(n, n)
        for c, g in ((hp, 0.0), (lp, 1.0)):
            y = dsp.filtfilt_z(c, x, 30, g)
            assert y.shape == (n,) and np.all(np.isfinite(y))
            p = min(30, n - 1)
            b, a = _ba(c)
            np.testing.assert_allclose(y, ss.filtfilt(b, a, x, padtype="odd", padlen=p), rtol=1e-9, atol=1e-11)


def test_filtfilt_constants():
    hp = dsp.biquad_highpass(5.0, 100.0, Q)
    lp = dsp.biquad_lowpass(3.0, 100.0, Q)
    bp = dsp.biquad_bandpass(math.sqrt(96.0), 100.0, math.sqrt(96.0) / 4.0)
    x = np.full(120, 0.8123)
    assert np.max(np.abs(dsp.filtfilt_z(hp, x, 30, 0.0))) < 1e-12
    assert np.max(np.abs(dsp.filtfilt_z(bp, x, 30, 0.0))) < 1e-12
    np.testing.assert_allclose(dsp.filtfilt_z(lp, x, 30, 1.0), x, rtol=0, atol=1e-12)


def test_filtfilt_linearity():
    hp = dsp.biquad_highpass(5.0, 100.0, Q)
    x1 = _noise(1, 150)
    x2 = _noise(2, 150)
    y = dsp.filtfilt_z(hp, 2.0 * x1 - 3.0 * x2, 30, 0.0)
    np.testing.assert_allclose(y, 2.0 * dsp.filtfilt_z(hp, x1, 30, 0.0) - 3.0 * dsp.filtfilt_z(hp, x2, 30, 0.0), atol=1e-12)


@pytest.mark.parametrize("kind", ["highpass", "lowpass", "bandpass"])
def test_filtfilt_matches_scipy(kind):
    fc, q = {"highpass": (5.0, Q), "lowpass": (3.0, Q), "bandpass": (math.sqrt(96.0), math.sqrt(96.0) / 4.0)}[kind]
    c = dsp.biquad(kind, fc, 100.0, q)
    g = 1.0 if kind == "lowpass" else 0.0
    t = np.arange(240) / 100.0
    x = 0.3 * np.sin(2 * np.pi * 1.3 * t) + 0.1 * np.sin(2 * np.pi * 17.0 * t) + 0.05 * _noise(3, 240) + 0.7
    y = dsp.filtfilt_z(c, x, 30, g)
    b, a = _ba(c)
    # identical in exact arithmetic to scipy's odd-padded filtfilt with the same pad
    np.testing.assert_allclose(y, ss.filtfilt(b, a, x, padtype="odd", padlen=30), rtol=1e-10, atol=1e-12)
    # and close to scipy's default filtfilt away from the edges
    np.testing.assert_allclose(y[60:-60], ss.filtfilt(b, a, x)[60:-60], atol=2e-3)


def test_filtfilt_padding_construction():
    # pad 2, n 4: pre = [2x0 - x2, 2x0 - x1], post = [2x3 - x2, 2x3 - x1]
    c = (1.0, 0.0, 0.0, 0.0, 0.0)  # identity filter (DC gain 1)
    x = [1.0, 4.0, 2.0, 8.0]
    assert list(dsp.filtfilt_z(c, x, 2, 1.0)) == x
    # two-tap average: y = 0.5 (xp[n] + xp[n-1]) forward and backward
    avg = (0.5, 0.5, 0.0, 0.0, 0.0)
    xp = [2 * 1.0 - 2.0, 2 * 1.0 - 4.0] + x + [2 * 8.0 - 2.0, 2 * 8.0 - 4.0]
    c1 = xp[0]
    y = [0.5 * (xp[i] - c1) + 0.5 * ((xp[i - 1] - c1) if i else 0.0) + c1 for i in range(len(xp))]
    r = y[::-1]
    c2 = r[0]
    z = [0.5 * (r[i] - c2) + 0.5 * ((r[i - 1] - c2) if i else 0.0) + c2 for i in range(len(r))]
    assert list(dsp.filtfilt_z(avg, x, 2, 1.0)) == z[::-1][2:6]


def test_resample_hand_cases():
    t = [0.0, 0.01, 0.02, 0.06, 0.07]
    v = [0.0, 1.0, 3.0, 5.0, 6.0]
    grid = [-0.005, 0.0, 0.005, 0.01, 0.015, 0.03, 0.065, 0.07, 0.09, 0.1]
    vals, miss = dsp.resample(t, v, grid, 0.025)
    w = (0.015 - 0.01) / (0.02 - 0.01)
    w2 = (0.03 - 0.02) / (0.06 - 0.02)
    w3 = (0.065 - 0.06) / (0.07 - 0.06)
    assert list(vals) == [0.0, 0.0, 0.0 + ((0.005 - 0.0) / 0.01) * 1.0, 1.0, 1.0 + w * 2.0,
                          3.0 + w2 * 2.0, 5.0 + w3 * 1.0, 6.0, 6.0, 6.0]
    assert list(miss) == [True, False, False, False, False, True, False, False, False, True]


def test_resample_empty_and_single():
    vals, miss = dsp.resample([], [], [0.0, 0.1], 0.025)
    assert list(vals) == [0.0, 0.0] and list(miss) == [True, True]
    vals, miss = dsp.resample([1.0], [2.0], [0.9, 1.0, 1.02, 1.03], 0.025)
    assert list(vals) == [2.0, 2.0, 2.0, 2.0]
    assert list(miss) == [True, False, False, True]
    vals, miss = dsp.resample([0.0, 1.0], [1.0, 2.0], [], 0.025)
    assert vals.size == 0 and miss.size == 0
