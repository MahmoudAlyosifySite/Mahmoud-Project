"""SPEC section 6: every quality rule fires on a crafted case and not on a clean one."""

import copy
import math

import numpy as np
import pytest

from auditor_ref import quality, synth
from auditor_ref.challenge import generate
from auditor_ref.signal import process
from auditor_ref.synth import TABLE, SynthOptions


def _proc(seed=1, hand=None, challenge_seed=101, **opts):
    return process(synth.synth_recording(seed, hand=hand, opts=SynthOptions(**opts), challenge_seed=challenge_seed))


@pytest.fixture(scope="module")
def clean():
    return _proc()


@pytest.fixture(scope="module")
def table():
    return _proc(hand=TABLE)


def test_clean_handheld_passes(clean):
    rep = quality.evaluate(clean)
    assert rep["passed"] and rep["issues"] == []
    assert list(rep["metrics"]) == list(quality.METRIC_KEYS)
    assert isinstance(rep["metrics"]["lag"], int)


def test_metric_definitions(clean):
    m = quality.compute_metrics(clean)
    q0, q1 = clean.quiet_range
    e0, e1 = clean.event_range
    assert m["quietGyroRMS"] == pytest.approx(math.sqrt(np.mean(clean.energy_gyro[q0:q1])), rel=1e-12)
    assert m["quietAccelRMS"] == pytest.approx(math.sqrt(np.mean(clean.energy_accel[q0:q1])), rel=1e-12)
    assert m["gravityZ"] == pytest.approx(abs(np.mean(clean.gravity[2])), rel=1e-12)
    assert m["snr"] == pytest.approx(np.mean(clean.envelope[e0:e1]) / np.mean(clean.envelope[q0:q1]), rel=1e-12)
    a = clean.gravity[:, :20].mean(axis=1)
    b = clean.gravity[:, -20:].mean(axis=1)
    ang = math.degrees(math.acos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))
    assert m["gravityAngleDeg"] == pytest.approx(ang, abs=1e-6)
    rl = quality.motion_lowpassed(clean)
    assert m["motionRMS"] == pytest.approx(math.sqrt(np.mean((rl ** 2).sum(axis=0))), rel=1e-12)
    assert (m["ncc"], m["peakMargin"], m["lag"]) == (clean.ncc, clean.peak_margin, clean.lag)


def test_sensor_gaps():
    rep = quality.evaluate(_proc(gap=(0.5, 0.35)))
    assert rep["issues"][0] == "sensorGaps" and not rep["passed"]


def test_low_sample_rate():
    rep = quality.evaluate(_proc(rate=50.0))
    assert rep["metrics"]["sampleRate"] == pytest.approx(50.0, rel=0.02)
    assert rep["issues"] == ["lowSampleRate"]


def test_resting_without_and_with_calibration(table):
    rep = quality.evaluate(table, mode="handheld")
    assert rep["issues"] == ["resting"]
    q = rep["metrics"]["quietGyroRMS"]
    # with a calibration the threshold is 2.5 x calibration.quietGyroRMS
    assert quality.evaluate(table, calibration={"quietGyroRMS": q / 2.0})["issues"] == ["resting"]
    assert quality.evaluate(table, calibration={"quietGyroRMS": q / 3.0})["issues"] == []


def test_resting_needs_flat_phone(clean):
    m = quality.compute_metrics(clean)
    assert m["gravityZ"] < 0.95
    assert "resting" not in quality.evaluate(clean, calibration={"quietGyroRMS": 1.0})["issues"]


def test_moving_by_rotation_and_by_tilt():
    rep = quality.evaluate(_proc(motion=0.6))
    assert rep["metrics"]["motionRMS"] > 0.3 and rep["issues"] == ["moving"]
    rep = quality.evaluate(_proc(tilt_deg=25.0))
    assert rep["metrics"]["gravityAngleDeg"] > 10 and rep["metrics"]["motionRMS"] < 0.3
    assert rep["issues"] == ["moving"]


def test_weak_signal_and_alignment():
    rep = quality.evaluate(_proc(signal_scale=0.03))
    assert "weakSignal" in rep["issues"]
    assert rep["issues"] == [i for i in ("weakAlignment", "weakSignal") if i in rep["issues"]]


def test_weak_alignment_with_a_different_challenge():
    # the phone played one pattern, the recording claims another (replay-like)
    rec = synth.synth_recording(1, challenge_seed=101)
    rec = copy.deepcopy(rec)
    rec["challenge"] = generate(999)
    rep = quality.evaluate(process(rec))
    assert "weakAlignment" in rep["issues"]


def test_on_table_mode(table, clean):
    assert quality.evaluate(table, mode="onTable")["issues"] == []
    rep = quality.evaluate(clean, mode="onTable")
    assert rep["issues"] == ["notResting"]
    rep = quality.evaluate(_proc(hand=TABLE, gap=(0.5, 0.35)), mode="onTable")
    assert rep["issues"][0] == "sensorGaps" and "resting" not in rep["issues"]


def test_on_table_order_puts_not_resting_last():
    p = _proc(gap=(0.5, 0.35), signal_scale=0.03)
    issues = quality.evaluate(p, mode="onTable")["issues"]
    assert issues[0] == "sensorGaps" and issues[-1] == "notResting"
    assert "moving" not in issues and "resting" not in issues


def test_messages_and_bad_mode(clean):
    assert set(quality.MESSAGES) == set(quality.ISSUES)
    assert quality.MESSAGES["notResting"] == "Put the phone flat on a table and don't touch it"
    with pytest.raises(ValueError):
        quality.evaluate(clean, mode="pocket")
