"""The synthetic recording generator and the recording helpers (SPEC section 4)."""

import json
import re

import numpy as np
import pytest

from auditor_ref import features, recording, synth
from auditor_ref.recording import MOTION_KEYS, SENSOR_KEYS, format_date, load_jsonl, save_jsonl, sort_dedup
from auditor_ref.signal import process
from auditor_ref.synth import TABLE, SynthOptions

TOP_KEYS = {"schema", "id", "createdAt", "purpose", "challenge", "hapticStart", "accel", "gyro", "motion"}


def test_deterministic_and_json_form():
    a = synth.synth_recording(3, challenge_seed=5, label=synth.label("P1", "Amira"), device={"model": "iPhone16,1", "system": "iOS 26.1", "app": "0.1.0 (1)"})
    b = synth.synth_recording(3, challenge_seed=5, label=synth.label("P1", "Amira"), device={"model": "iPhone16,1", "system": "iOS 26.1", "app": "0.1.0 (1)"})
    assert json.dumps(a) == json.dumps(b)
    assert set(a) == TOP_KEYS | {"label", "device"}
    assert a["schema"] == "7oda-auditor.recording.v1"
    assert re.match(r"^[0-9A-F]{8}-[0-9A-F]{4}-4[0-9A-F]{3}-[89AB][0-9A-F]{3}-[0-9A-F]{12}$", a["id"])
    assert re.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$", a["createdAt"])
    assert set(a["accel"]) == set(SENSOR_KEYS) and set(a["motion"]) == set(MOTION_KEYS)
    for key in ("accel", "gyro", "motion"):
        lens = {len(v) for v in a[key].values()}
        assert len(lens) == 1 and lens.pop() > 150
    assert set(a["label"]) == {"personID", "personName", "hand", "accessory", "posture", "sessionID"}
    c = synth.synth_recording(4, challenge_seed=5)
    assert "label" not in c and "device" not in c
    assert c["accel"]["x"] != a["accel"]["x"]


def test_recording_window_and_clock():
    r = synth.synth_recording(1, challenge_seed=2)
    t = np.array(r["accel"]["t"])
    hs = r["hapticStart"]
    assert t.min() >= hs - 0.25 - 0.1 and t.max() <= hs + r["challenge"]["duration"] + 0.35
    assert np.all(np.diff(t) > 0)
    assert abs(np.median(np.diff(t)) - 0.01) < 0.001


def test_options():
    r = synth.synth_recording(1, opts=SynthOptions(duplicates=0.2), challenge_seed=2)
    t = r["gyro"]["t"]
    assert len(set(t)) < len(t)
    r = synth.synth_recording(1, opts=SynthOptions(unsorted=True), challenge_seed=2)
    assert np.any(np.diff(r["accel"]["t"]) < 0)
    clean = synth.synth_recording(1, challenge_seed=2)
    np.testing.assert_array_equal(sort_dedup(r["accel"])["t"], np.array(clean["accel"]["t"]))
    r = synth.synth_recording(1, opts=SynthOptions(gap=(0.5, 0.3)), challenge_seed=2)
    assert np.max(np.diff(r["accel"]["t"])) > 0.25 and np.max(np.diff(r["motion"]["t"])) < 0.05
    r = synth.synth_recording(1, opts=SynthOptions(drop_gyro=True), challenge_seed=2)
    assert r["gyro"]["t"] == []
    r = synth.synth_recording(1, opts=SynthOptions(rate=50.0), challenge_seed=2)
    assert abs(np.median(np.diff(r["accel"]["t"])) - 0.02) < 0.002


def test_table_and_hands_differ():
    r = synth.synth_recording(1, hand=TABLE, challenge_seed=2, purpose="calibration")
    g = np.mean([r["motion"]["gravityX"], r["motion"]["gravityY"], r["motion"]["gravityZ"]], axis=1)
    np.testing.assert_allclose(g, [0, 0, -1], atol=0.01)
    assert np.std(np.diff(r["gyro"]["x"])) < 0.01
    f = np.array([features.extract_absolute(process(synth.synth_recording(s, hand=synth.make_hand(h), challenge_seed=s)))
                  for h in (1, 2) for s in (10, 11, 12)])
    a, b = f[:3], f[3:]
    sd = np.sqrt((a.var(axis=0, ddof=1) + b.var(axis=0, ddof=1)) / 2.0) + 1e-9
    separation = np.abs(a.mean(axis=0) - b.mean(axis=0)) / sd
    assert np.count_nonzero(separation > 3.0) >= 5   # two synthetic hands differ on several features


def test_jsonl_round_trip(tmp_path):
    recs = [synth.synth_recording(s, challenge_seed=s) for s in (1, 2)]
    path = tmp_path / "recordings.jsonl"
    save_jsonl(path, recs)
    lines = path.read_text().splitlines()
    assert len(lines) == 2 and all(line.startswith("{") for line in lines)
    assert load_jsonl(path) == recs


def test_make_recording_and_dates():
    import datetime as dt
    ch = {"version": 1, "seed": "0x0000000000000001", "duration": 1.0, "events": []}
    rec = recording.make_recording(ch, 10.0, {"t": [1.0], "x": [0.0], "y": [0.0], "z": [-1.0]},
                                   {"t": [], "x": [], "y": [], "z": []}, None, purpose="signalCheck",
                                   created_at=dt.datetime(2026, 11, 24, 10, 0, 0, 123456, tzinfo=dt.timezone.utc))
    assert rec["createdAt"] == "2026-11-24T10:00:00Z" and rec["motion"]["gravityZ"] == []
    assert format_date(dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.timezone(dt.timedelta(hours=2)))) == "2026-01-02T01:04:05Z"
    with pytest.raises(ValueError):
        recording.make_recording(ch, 0.0, rec["accel"], rec["gyro"], purpose="other")


def test_sort_dedup_motion_stream():
    m = {k: [3.0, 1.0, 1.0, 2.0] for k in MOTION_KEYS}
    m["gravityZ"] = [-0.3, -0.1, -0.15, -0.2]
    out = sort_dedup(m)
    assert list(out["t"]) == [1.0, 2.0, 3.0] and list(out["gravityZ"]) == [-0.1, -0.2, -0.3]
    with pytest.raises(ValueError):
        sort_dedup({"t": [1.0, 2.0], "x": [1.0]})
