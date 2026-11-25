"""v1 input tensor (SPEC 14.1): contract, fast path vs reference path, datasets."""
import importlib.util

import numpy as np
import pytest

from v1 import data as D
from v1 import sigproc, standin

HAVE_REF = importlib.util.find_spec("auditor_ref") is not None
HAVE_SIM = importlib.util.find_spec("sim") is not None


def spec_loop(accel_hp, gyro_hp, gravity, template, lag, n):
    """Literal transcription of SPEC 14.1."""
    x = np.zeros((10, 256))
    for u in range(256):
        k = u + lag
        if not (0 <= k < n):
            continue
        for c in range(3):
            x[c, u] = accel_hp[c][k] / 0.02
            x[3 + c, u] = gyro_hp[c][k] / 0.05
            x[6 + c, u] = gravity[c][k]
        x[9, u] = template[u] if u < n else 0.0
    return np.clip(x, -50, 50).astype(np.float32)


@pytest.mark.parametrize("n,lag", [(150, 0), (240, -10), (240, 25), (201, 7), (254, -3), (30, 12)])
def test_assemble_matches_spec(n, lag):
    rng = np.random.default_rng(n + lag)
    a = rng.normal(0, 0.05, (3, n))
    g = rng.normal(0, 0.1, (3, n))
    G = rng.normal(0, 1, (3, n))
    a[0, n // 2] = 5.0                     # 250 after scaling: must clip to 50
    g[1, n // 3] = -9.0
    t = np.abs(rng.normal(0, 1, n))
    got = D.assemble_tensor(a, g, G, t, lag, n)
    want = spec_loop(a, g, G, t, lag, n)
    assert got.dtype == np.float32 and got.shape == (10, 256)
    np.testing.assert_array_equal(got, want)
    assert got.max() <= 50 and got.min() >= -50


def test_pattern_starts_at_u25():
    rec = standin.simulate_recording(standin.hand_params(standin.person_params(np.random.default_rng(1)), "right",
                                                         np.random.default_rng(2)), 1234, np.random.default_rng(3))
    x, fp = D.tensor_from_recording(rec, return_processed=True)
    first = rec["challenge"]["events"][0]["time"]
    u = np.arange(256)
    u0 = int(np.argmax(u / 100.0 - 0.25 >= first))   # first template sample at or after the tap
    assert u0 >= 25 and np.all(x[9, :u0] == 0) and x[9, u0] > 0.25
    # the signal is read at n = u + lag
    u = np.arange(256)
    k = u + fp.lag
    ok = (k >= 0) & (k < fp.n)
    np.testing.assert_allclose(x[0, u[ok]], np.clip(fp.accel_hp[0, k[ok]] / 0.02, -50, 50), rtol=1e-6, atol=1e-6)
    assert np.all(x[:, ~ok] == 0)


def _recordings():
    recs = standin.generate(5, 3, seed=11, messy=True) + standin.generate(3, 2, seed=12)
    if HAVE_SIM:
        from sim import core, params
        people = params.make_population(3, seed=5)
        for mode in ("none", "lpf"):
            ph = params.make_phone(3, mode)
            recs += core.simulate_batch([people[i % 3] for i in range(8)], ["right", "left"] * 4,
                                        list(range(8)), ph, np.random.default_rng(1),
                                        conditions=["handheld"] * 7 + ["twoHanded"])
    return recs


@pytest.mark.skipif(not HAVE_REF, reason="auditor_ref not available")
def test_fast_path_equals_reference_path():
    from auditor_ref import signal as S

    for rec in _recordings():
        p = S.process(rec)
        ref = D.tensor_from_processed(p)
        fast, fp = D.tensor_from_recording(rec, return_processed=True)
        assert fp.lag == p.alignment.lag
        assert fp.n == p.n
        np.testing.assert_allclose(fast, ref, rtol=0, atol=1e-6)
        assert abs(fp.ncc - p.alignment.ncc) < 1e-9


@pytest.mark.skipif(not HAVE_REF, reason="auditor_ref not available")
def test_fast_dsp_matches_reference_dsp():
    from auditor_ref import dsp

    rng = np.random.default_rng(0)
    b, a = sigproc.biquad_highpass(5.0, 100.0, 1 / np.sqrt(2))
    coeffs = dsp.biquad_highpass(5.0, 100.0, 1 / np.sqrt(2))
    np.testing.assert_allclose(np.r_[b, a[1:]], np.asarray(coeffs, float), rtol=0, atol=1e-15)
    for n in (0, 1, 2, 5, 31, 200):
        x = rng.normal(size=n)
        want = np.asarray(dsp.filtfilt_z(coeffs, x, 30, 0.0), float)
        got = sigproc.filtfilt_z(x, b, a, 30, 0.0)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_sort_dedup_and_resample_rules():
    t, v = sigproc.sort_dedup([3.0, 1.0, 2.0, 2.0, 4.0], [[30, 10, 20, 21, 40]])
    np.testing.assert_array_equal(t, [1, 2, 3, 4])
    np.testing.assert_array_equal(v[0], [10, 20, 30, 40])    # first of the duplicates is kept
    vals, miss = sigproc.resample(np.array([0.0, 0.01, 0.05]), np.array([[0.0, 1.0, 5.0]]),
                                  np.array([-0.01, 0.005, 0.03, 0.05, 0.08]), 0.025)
    np.testing.assert_allclose(vals[0], [0.0, 0.5, 3.0, 5.0, 5.0])
    np.testing.assert_array_equal(miss, [True, False, True, False, True])


def test_tensorset_roundtrip(tmp_path):
    ts = D.tensorset_from_recordings(standin.generate(2, 2, seed=3))
    ts.extra["check"] = np.arange(len(ts), dtype=np.int32)
    path = str(tmp_path / "s.npz")
    ts.save(path)
    back = D.TensorSet.load(path)
    np.testing.assert_array_equal(back.x, ts.x)
    assert back.person_ids == ts.person_ids and np.all(back.extra["check"] == ts.extra["check"])
    ident = back.identity()
    assert set(np.unique(ident)) == {0, 1, 2, 3}
    both = D.TensorSet.concat([back, back])
    assert len(both) == 2 * len(ts) and both.person_ids == ts.person_ids


def test_augment_presets_and_device_loading(tmp_path):
    import torch
    from v1 import trainlib as TL

    assert D.AugmentConfig.named("none").enabled is False
    light = D.AugmentConfig.named("light")
    assert light.case_prob == 0.0 and light.gain_sigma < D.AugmentConfig().gain_sigma
    rng = np.random.default_rng(0)
    for i in range(2):
        n = 3 + i
        ts = D.TensorSet(rng.normal(size=(n, 10, 256)).astype(np.float16), np.zeros(n, np.int32),
                         np.array([0, 1, 2, 0, 1][:n], np.int8), np.zeros(n, np.int8), np.zeros(n, np.int8),
                         np.zeros(n, np.int32), np.ones(n, np.float32), [f"P{i}"], [f"P{i}-S0"])
        ts.save(str(tmp_path / f"shard_{i}.npz"))
    host = TL.load_sets(str(tmp_path / "shard_*.npz"))
    dev = TL.load_sets(str(tmp_path / "shard_*.npz"), device="cpu")
    assert isinstance(dev.x, torch.Tensor) and dev.x.dtype == torch.float16
    np.testing.assert_array_equal(dev.x.numpy(), host.x)
    np.testing.assert_array_equal(dev.person, host.person)
    dd = D.DeviceData(dev, "cpu")
    assert dd.n == 7 and torch.equal(dd.identity, torch.from_numpy(host.identity()))
