#!/usr/bin/env python3
"""Write the parity fixtures of SPEC section 11.

The Python reference (``research/auditor_ref``) computes every expected value;
the Swift tests in ``AuditorCore`` read the files and must reproduce them.

Files (in ``AuditorCore/Tests/AuditorCoreTests/Fixtures/`` by default):

- ``rng_vectors.json``       SplitMix64 / xoshiro256** / uniform / int draws
- ``challenge_vectors.json`` generated challenges for 64 seeds
- ``dsp_vectors.json``       biquad coefficients, lfilter / filtfiltZ outputs,
                             resampling and stream cleaning
- ``pipeline_vectors.json``  full recordings with signal, quality and features
- ``model_vectors.json``     Gaussian model, hand head and decisions

Rules: floats are written with ``json`` default repr (full precision), seeds
as ``0x`` + 16 hex digits, infinities as ``null``, never NaN. Generation is
deterministic (running twice gives byte-identical files). Cases whose discrete
outcomes (argmax/argmin choices, threshold tests) are within a relative margin
of 1e-3 of flipping are re-drawn.

Usage: ``python scripts/make_fixtures.py [--out DIR]``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402

from auditor_ref import calibration, dsp, features, quality, signal, synth  # noqa: E402
from auditor_ref._numerics import TieAudit  # noqa: E402
from auditor_ref.challenge import generate  # noqa: E402
from auditor_ref.identifier import Identifier, IdentifierConfig  # noqa: E402
from auditor_ref.model import ModelConfig  # noqa: E402
from auditor_ref.recording import sort_dedup  # noqa: E402
from auditor_ref.rng import MASK64, SplitMix64, Xoshiro256, seed_to_str  # noqa: E402

OUT_DEFAULT = HERE.parents[2] / "AuditorCore" / "Tests" / "AuditorCoreTests" / "Fixtures"
TIE_MARGIN = 1e-3
DEVICE = {"model": "iPhone16,1", "system": "iOS 26.1", "app": "0.1.0 (1)"}
CREATED = "2026-11-24T10:00:00Z"


def generator_info(name: str) -> dict:
    return {"script": "research/scripts/make_fixtures.py", "spec": "docs/SPEC.md section 11", "file": name}


# --------------------------------------------------------------------------- JSON
def clean(obj):
    """Convert to JSON-ready Python types: tuples -> lists, numpy -> Python,
    +/-inf -> None. NaN is an error."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [clean(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        if math.isnan(v):
            raise ValueError("NaN in fixture")
        return None if math.isinf(v) else v
    return obj


def write_json(path: Path, obj) -> None:
    text = json.dumps(clean(obj), allow_nan=False, separators=(",", ":"), ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def margin_ok(audit: TieAudit) -> bool:
    return audit.min_margin() >= TIE_MARGIN


# --------------------------------------------------------------------------- RNG
RNG_SEEDS = [0, 1, 2, 42, 1234, 0xFFFFFFFF, 0x8000000000000000, 0x0123456789ABCDEF, 0xDEADBEEFCAFEBABE, MASK64]
INT_RANGES = [(6, 8), (0, 0), (-5, 5), (1, 100), (0, 1), (-3, -1)]
UNIFORM_RANGES = [(0.05, 0.10), (0.4, 1.0), (-2.0, 3.0)]


def make_rng() -> dict:
    cases = []
    for seed in RNG_SEEDS:
        sm = SplitMix64(seed)
        splitmix = [seed_to_str(sm.next()) for _ in range(5)]
        x = Xoshiro256(seed)
        xo = [seed_to_str(x.next()) for _ in range(10)]
        u = Xoshiro256(seed)
        uni = [u.uniform() for _ in range(5)]
        ints = []
        for lo, hi in INT_RANGES:
            g = Xoshiro256(seed)
            ints.append({"lo": lo, "hi": hi, "values": [g.int_in_range(lo, hi) for _ in range(10)]})
        ranges = []
        for a, b in UNIFORM_RANGES:
            g = Xoshiro256(seed)
            ranges.append({"a": a, "b": b, "values": [g.uniform_range(a, b) for _ in range(5)]})
        cases.append({"seed": seed_to_str(seed), "splitmix": splitmix, "xoshiro": xo, "uniform": uni,
                      "ints": ints, "uniformRange": ranges})
    return {"generator": generator_info("rng_vectors.json"),
            "note": "each list is drawn from a fresh generator seeded with `seed`",
            "cases": cases}


# --------------------------------------------------------------------------- challenge
def challenge_seeds() -> list[int]:
    seeds = [0, 1, 2, 3, 1234, MASK64, 0x8000000000000000, 0xFFFFFFFF, 0xDEADBEEFCAFEBABE, 0x0123456789ABCDEF]
    sm = SplitMix64(0x70DA_A0D1_70C4_A11E)
    while len(seeds) < 64:
        seeds.append(sm.next())
    return seeds


def make_challenges() -> dict:
    cases = [{"seed": seed_to_str(s), "spec": generate(s)} for s in challenge_seeds()]
    counts = {sum(e["kind"] == "tap" for e in c["spec"]["events"]) for c in cases}
    last_ramp = any(c["spec"]["events"][-1]["kind"] == "ramp" for c in cases)
    assert counts == {6, 7, 8} and last_ramp, "challenge fixture lacks coverage"
    return {"generator": generator_info("challenge_vectors.json"), "cases": cases}


# --------------------------------------------------------------------------- DSP
F0 = math.sqrt(8.0 * 12.0)
Q = dsp.Q_BUTTERWORTH
DEFAULT_FILTERS = [("highpass", 5.0, 100.0, Q, 0.0), ("lowpass", 3.0, 100.0, Q, 1.0), ("bandpass", F0, 100.0, F0 / 4.0, 0.0)]


def _normals(seed: int, n: int, scale: float = 1.0, offset: float = 0.0) -> list[float]:
    r = Xoshiro256(seed)
    return [offset + scale * r.normal() for _ in range(n)]


def make_dsp() -> dict:
    biquads = []
    params = [("highpass", 5.0, 100.0, Q), ("lowpass", 3.0, 100.0, Q), ("bandpass", F0, 100.0, F0 / 4.0),
              ("highpass", 1.0, 50.0, Q), ("highpass", 12.5, 100.0, 0.5), ("highpass", 0.5, 100.0, Q),
              ("lowpass", 20.0, 200.0, Q), ("lowpass", 40.0, 100.0, 2.0), ("bandpass", 25.0, 100.0, 2.0),
              ("bandpass", 7.0, 50.0, 0.8)]
    for kind, fc, fs, q in params:
        biquads.append({"kind": kind, "fc": fc, "fs": fs, "q": q, "coeffs": list(dsp.biquad(kind, fc, fs, q))})

    filters = []

    def add_filter(kind, fc, fs, q, dc, pad, x, label):
        c = dsp.biquad(kind, fc, fs, q)
        filters.append({"name": label, "kind": kind, "fc": fc, "fs": fs, "q": q, "pad": pad, "dcGain": dc,
                        "x": list(x), "lfilter": list(dsp.lfilter(c, x)), "filtfilt": list(dsp.filtfilt_z(c, x, pad, dc))})

    seed = 1
    for kind, fc, fs, q, dc in DEFAULT_FILTERS:
        for n in (0, 1, 2, 3, 5, 17, 30, 31, 64, 207):
            seed += 1
            add_filter(kind, fc, fs, q, dc, 30, _normals(seed, n, 0.05, 0.3), f"{kind}-n{n}")
        add_filter(kind, fc, fs, q, dc, 30, [0.8123] * 40, f"{kind}-constant")
        add_filter(kind, fc, fs, q, dc, 30, [0.0] * 20 + [1.0] * 30, f"{kind}-step")
    hp = DEFAULT_FILTERS[0]
    for pad, n in ((0, 10), (5, 40), (100, 50), (1, 2)):
        seed += 1
        add_filter(hp[0], hp[1], hp[2], hp[3], hp[4], pad, _normals(seed, n, 0.02, -0.7), f"highpass-pad{pad}-n{n}")

    resample = []

    def add_resample(label, t, v, grid, tol=0.025):
        vals, miss = dsp.resample(t, v, grid, tol)
        resample.append({"name": label, "t": list(t), "v": list(v), "grid": list(grid), "gapTolerance": tol,
                         "values": list(vals), "missing": [bool(m) for m in miss]})

    r = Xoshiro256(77)
    t = []
    tk = 100.0031
    for _ in range(60):
        t.append(tk)
        tk += 0.01 + 0.0006 * r.normal()
    v = _normals(78, 60, 0.03, -0.8)
    grid = [100.0 + k / 100.0 for k in range(-3, 66)]
    add_resample("jittered", t, v, grid)
    tg = t[:20] + t[30:]
    vg = v[:20] + v[30:]
    add_resample("gap", tg, vg, grid)
    add_resample("empty", [], [], grid[:10])
    add_resample("single", [100.2], [0.5], [100.15, 100.2, 100.21, 100.224, 100.226, 100.3])
    add_resample("two", [1.0, 1.04], [2.0, -1.0], [0.99, 1.0, 1.01, 1.02, 1.04, 1.05, 1.066])
    ts = [5.0 + k / 100.0 for k in range(10)]
    add_resample("exactHits", ts, _normals(79, 10), ts)
    add_resample("emptyGrid", [1.0, 2.0], [1.0, 2.0], [])

    clean_cases = []
    raw = {"t": [3.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0], "x": [30.0, 10.0, 20.0, 11.0, 31.0, 5.0, 21.0],
           "y": [0.3, 0.1, 0.2, 0.11, 0.31, 0.05, 0.21], "z": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}
    out = sort_dedup(raw)
    clean_cases.append({"input": raw, "output": {k: list(out[k]) for k in raw}})
    empty = {"t": [], "x": [], "y": [], "z": []}
    clean_cases.append({"input": empty, "output": empty})
    return {"generator": generator_info("dsp_vectors.json"), "biquads": biquads, "filters": filters,
            "resample": resample, "sortDedup": clean_cases}


# --------------------------------------------------------------------------- pipeline
def round_recording(rec: dict, digits: int = 7) -> dict:
    """Round stream values (inputs only) to keep the fixture small."""
    out = dict(rec)
    for key in ("accel", "gyro", "motion"):
        out[key] = {k: [round(float(v), digits) for v in vals] for k, vals in rec[key].items()}
    return out


def build_calibration() -> dict:
    procs, reps = [], []
    for i in range(20):
        rec = round_recording(synth.synth_recording(9000 + i, hand=synth.TABLE, challenge_seed=19000 + i,
                                                    purpose="calibration", device=DEVICE, created_at=CREATED))
        p = signal.process(rec)
        procs.append(p)
        reps.append(quality.evaluate(p, mode="onTable"))
    cal = calibration.build(procs, reps, DEVICE["model"], "2026-11-24T09:00:00Z")
    assert cal is not None and cal["patternsUsed"] >= 12
    return cal


PERSON = synth.make_hand(31)
PERSON2 = synth.make_hand(32)


def pipeline_specs(cal: dict) -> list[dict]:
    O = synth.SynthOptions
    lab = synth.label("7A1C2E9B-3F4D-4E5A-9B6C-1D2E3F4A5B6C", "Amira", "right")
    lab2 = synth.label("0B9D4C1E-6A2F-4B3C-8D7E-9F0A1B2C3D4E", "Basem", "left", posture="standing")
    return [
        dict(issues=[], name="handheld", desc="normal handheld check, no calibration", hand=PERSON, opts=O(), cal=None,
             mode="handheld", label=lab),
        dict(issues=[], name="handheldCalibrated", desc="normal handheld check with a calibration (relative features)",
             hand=PERSON2, opts=O(latency=0.055), cal=cal, mode="handheld", label=lab2),
        dict(issues=[], name="onTable", desc="calibration pattern flat on a table, onTable mode", hand=synth.TABLE, opts=O(),
             cal=None, mode="onTable", label=None, purpose="calibration"),
        dict(issues=["sensorGaps"], name="gaps", desc="0.35 s hole in accel and gyro", hand=PERSON, opts=O(gap=(0.55, 0.35)), cal=None,
             mode="handheld", label=lab),
        dict(issues=[], name="jitteredUnsorted", desc="large timestamp jitter, samples shuffled", hand=PERSON,
             opts=O(jitter=0.0012, unsorted=True), cal=cal, mode="handheld", label=lab),
        dict(issues=[], name="duplicateTimestamps", desc="20% of samples repeated with the same timestamp", hand=PERSON2,
             opts=O(duplicates=0.2), cal=None, mode="handheld", label=lab2),
        dict(issues=["moving"], name="moving", desc="hand moving (rotation rate ~0.6 rad/s)", hand=PERSON, opts=O(motion=0.6),
             cal=None, mode="handheld", label=lab),
        dict(issues=["moving"], name="tilted", desc="grip angle changes by 25 degrees", hand=PERSON2, opts=O(tilt_deg=25.0), cal=cal,
             mode="handheld", label=lab2),
        dict(issues=["resting"], name="resting", desc="phone on a table during a check (calibrated threshold)", hand=synth.TABLE,
             opts=O(), cal=cal, mode="handheld", label=lab),
        dict(issues=["resting"], name="restingNoCalibration", desc="phone on a table during a check (absolute threshold)",
             hand=synth.TABLE, opts=O(), cal=None, mode="handheld", label=lab),
        dict(issues=["weakAlignment", "weakSignal"], name="weakSignal", desc="response 30x weaker (thick case)", hand=PERSON, opts=O(signal_scale=0.03),
             cal=None, mode="handheld", label=lab),
        dict(issues=["lowSampleRate"], name="lowSampleRate", desc="sensors at 50 Hz", hand=PERSON2, opts=O(rate=50.0), cal=None,
             mode="handheld", label=lab2),
        dict(issues=["sensorGaps"], name="gyroMissing", desc="gyro stream without samples", hand=PERSON, opts=O(drop_gyro=True), cal=None,
             mode="handheld", label=lab),
        dict(issues=["weakAlignment"], name="wrongChallenge", desc="recording claims a different pattern than the one played", hand=PERSON,
             opts=O(), cal=None, mode="handheld", label=lab, swap_challenge=True),
        dict(issues=[], name="lateOnset", desc="haptic onset 0.18 s after hapticStart", hand=PERSON2, opts=O(latency=0.18),
             cal=cal, mode="handheld", label=lab2),
        dict(issues=["weakAlignment"], name="replayFlatPeak",
             desc="a different pattern than the one played, ncc >= minNCC but a flat correlation peak (peakMargin < minPeakMargin)",
             hand=PERSON, opts=O(), cal=None, mode="handheld", label=lab, swap_challenge=True, attempts=400,
             accept=lambda e: e["ncc"] >= quality.DEFAULT_CONFIG.min_ncc),
    ]


def run_pipeline(rec: dict, cal: dict | None, mode: str):
    audit = TieAudit()
    p = signal.process(rec, audit=audit)
    rep = quality.evaluate(p, calibration=cal, mode=mode, audit=audit)
    fa = features.extract_absolute(p, audit=audit)
    fr = features.relative(fa, cal)
    expected = {
        "n": p.n, "lag": p.lag, "ncc": p.ncc, "peakMargin": p.peak_margin,
        "noiseAccel": p.noise_accel, "noiseGyro": p.noise_gyro,
        "gapFraction": p.gap_fraction, "sampleRate": p.sample_rate,
        "quietRange": list(p.quiet_range), "eventRange": list(p.event_range),
        "metrics": rep["metrics"], "issues": rep["issues"], "passed": rep["passed"],
        "featuresAbsolute": list(fa), "features": list(fr),
    }
    return expected, audit


def make_pipeline() -> tuple[dict, dict]:
    cal = build_calibration()
    cases = []
    for idx, spec in enumerate(pipeline_specs(cal)):
        for attempt in range(spec.get("attempts", 40)):
            seed = 1000 * (idx + 1) + attempt
            rec = synth.synth_recording(seed, hand=spec["hand"], opts=spec["opts"],
                                        challenge_seed=50000 + seed, purpose=spec.get("purpose", "check"),
                                        label=spec["label"], device=DEVICE, created_at=CREATED)
            if spec.get("swap_challenge"):
                rec["challenge"] = generate(70000 + seed)
            rec = round_recording(rec)
            expected, audit = run_pipeline(rec, spec["cal"], spec["mode"])
            if (margin_ok(audit) and expected["issues"] == spec["issues"]
                    and all(math.isfinite(v) for v in expected["featuresAbsolute"])
                    and spec.get("accept", lambda e: True)(expected)):
                break
        else:
            raise RuntimeError(f"no suitable draw for pipeline case {spec['name']}: {expected['issues']} {audit.worst()}")
        cases.append({"name": spec["name"], "description": spec["desc"], "recording": rec,
                      "calibration": spec["cal"], "mode": spec["mode"], "expected": expected})
    return {"generator": generator_info("pipeline_vectors.json"), "cases": cases}, cal


# --------------------------------------------------------------------------- model
def _mirror_left(h: synth.HandParams) -> synth.HandParams:
    """Systematic right/left difference shared by every synthetic person."""
    g = h.gravity
    return replace(h, gravity=(-g[0], g[1], g[2]),
                   accel_gain=(h.accel_gain[0] * 0.8, h.accel_gain[1] * 1.25, h.accel_gain[2]),
                   gyro_gain=(h.gyro_gain[0] * 1.3, h.gyro_gain[1], h.gyro_gain[2] * 0.75),
                   decay=h.decay * 1.15)


def _watch(h: synth.HandParams) -> synth.HandParams:
    return replace(h, decay=h.decay * 1.35, accel_gain=tuple(v * 0.8 for v in h.accel_gain),
                   gyro_gain=(h.gyro_gain[0] * 0.6, h.gyro_gain[1] * 0.9, h.gyro_gain[2] * 0.7))


def _hand_params(person_seed: int, hand: str, accessory: str) -> synth.HandParams:
    base = synth.make_hand(person_seed)
    h = base if hand == "right" else _mirror_left(base)
    h = synth.make_hand(person_seed * 10 + (1 if hand == "right" else 2), base=h, spread=0.3)
    return _watch(h) if accessory == "watch" else h


class FeatureSource:
    """Pipeline-derived feature vectors for synthetic people (cached)."""

    def __init__(self, cal: dict):
        self.cal = cal
        self.counter = 0

    def sample(self, params: synth.HandParams, seed: int) -> list[float]:
        h = synth.make_hand(seed, base=params, spread=0.06)
        rec = synth.synth_recording(seed, hand=h, challenge_seed=seed + 333_333)
        p = signal.process(rec)
        return [float(v) for v in features.extract(p, self.cal)]


def _decision_json(ident: Identifier, name: str, patterns: list[list[float]], audit: TieAudit) -> dict:
    d = ident.decide(patterns, audit=audit)
    return {
        "name": name,
        "patterns": patterns,
        "distances": d["distancesPerPattern"],
        "handLogOdds": d["handLogOddsPerPattern"],
        "outcome": d["outcome"],
        "bestDistance": d["bestDistance"],
        "marginName": d["marginName"],
        "marginHand": d["marginHand"],
        "meanDistances": [p["distance"] for p in d["prototypeDistances"]],
        "meanHandLogOdds": d["handLogOdds"],
        "bestPrototype": d["bestPrototype"],
        "personScores": d["personScores"],
        "adaptiveEligible": ident.adaptive_eligible(d),
    }


def _model_block(samples: list[dict], ident: Identifier, mcfg: ModelConfig, icfg: IdentifierConfig, queries: list[dict]) -> dict:
    m = ident.model
    hm = ident.hand_model
    return {
        "config": mcfg.to_json(),
        "identifierConfig": icfg.to_json(),
        "samples": [{"personID": s["prototype"][0], "hand": s["prototype"][1], "accessory": s["prototype"][2],
                     "source": s["source"], "features": s["features"]} for s in samples],
        "expected": {
            "prototypes": [list(p) for p in ident.prototypes],
            "mean": list(m.mean), "scale": list(m.scale),
            "shrinkage": m.shrinkage, "absThreshold": m.abs_threshold,
            "handShrinkage": hm.shrinkage if hm is not None else None,
            "handAbsThreshold": hm.abs_threshold if hm is not None else None,
            "shrinkageLW": None if math.isnan(m.shrinkage_lw) else m.shrinkage_lw,
            "mu": m.mu,
            "classCounts": m.class_counts,
            "classMeans": [list(r) for r in m.class_means],
            "looDistances": m.loo,
        },
        "queries": queries,
    }


def _select_queries(ident: Identifier, candidates: list[tuple[str, list[list[float]], str | None]]) -> list[dict]:
    """Keep candidates whose decision is tie-free; ``want`` (outcome type or
    unknown reason) filters candidate groups: the first passing candidate of
    each group name is kept."""
    out = []
    taken = set()
    for name, patterns, want in candidates:
        if name in taken:
            continue
        audit = TieAudit()
        q = _decision_json(ident, name, patterns, audit)
        kind = q["outcome"]["type"] if q["outcome"]["type"] != "unknown" else q["outcome"]["reason"]
        if callable(want):
            if not want(q):
                continue
        elif want is not None and kind != want:
            continue
        if not margin_ok(audit):
            continue
        out.append(q)
        taken.add(name)
    return out


PERSON_IDS = ["2F6B1D3A-8C4E-4F2A-9E1B-5C7D8E9F0A1B", "9C3E5A7B-1D2F-4A6C-8B9E-0F1A2B3C4D5E", "5D8F2B4C-6E1A-4C3D-9A7B-8E9F0A1B2C3D"]
STRANGER_SEEDS = [701, 702, 703, 704, 705, 706, 707, 708]


def _blend(a: list[float], b: list[float], w: float) -> list[float]:
    return [(1.0 - w) * x + w * y for x, y in zip(a, b)]


def _midpoints(ident: Identifier, kind: str) -> list[list[list[float]]]:
    """Three patterns on the line between two class means (standardised space).

    For classes a, b with ``Delta = ||L^-1 (m_a - m_b)||^2`` the point at
    weight ``w`` has ``D2_a = w^2 Delta`` and ``D2_b = (1 - w)^2 Delta``, so the
    margin between them is ``(1 - 2w) Delta``; ``w`` is chosen for a margin of
    3 (ambiguous: two people) or 1.5 (handUnclear: one person's two hands).
    """
    from auditor_ref.model import forward_sq_norm

    m = ident.model
    out = []
    protos = ident.prototypes
    for a, pa in enumerate(protos):
        for b, pb in enumerate(protos):
            if a == b or pa[2] != "none" or pb[2] != "none":
                continue
            if kind == "ambiguous" and pa[0] == pb[0]:
                continue
            if kind == "handUnclear" and not (pa[0] == pb[0] and pa[1] == "right" and pb[1] == "left"):
                continue
            delta = forward_sq_norm(m.chol, m.class_means[a] - m.class_means[b])
            target = 3.0 if kind == "ambiguous" else 1.5
            w = 0.5 - target / (2.0 * delta)
            if w <= 0.05:
                continue
            pats = []
            for dw in (-0.015, 0.0, 0.015):
                zc = (1.0 - (w + dw)) * m.class_means[a] + (w + dw) * m.class_means[b]
                pats.append([float(v) for v in m.mean + m.scale * zc])
            out.append(pats)
    return out


def _head_disagreement_candidates(ident: Identifier) -> list[list[float]]:
    """Points moved from a prototype mean along the hand head's direction."""
    if ident.hand_model is None:
        return []
    m = ident.model
    hm = ident.hand_model
    u = hm.class_means[1] - hm.class_means[0]           # right -> left (pooled)
    out = []
    for k, p in enumerate(ident.prototypes):
        if p[2] != "none":
            continue
        other = [j for j, q in enumerate(ident.prototypes) if q[0] == p[0] and q[1] != p[1] and q[2] == "none"]
        direction = u if p[1] == "right" else -u
        if other:
            v = m.class_means[other[0]] - m.class_means[k]
            direction = direction - (direction @ v) / (v @ v) * v
        for t in np.linspace(0.1, 4.0, 40):
            zc = m.class_means[k] + float(t) * direction
            out.append([float(x) for x in m.mean + m.scale * zc])
    return out


def make_model_set(src: FeatureSource, person_ids: list[str], person_seeds: list[int], with_watch: bool, tag: str) -> dict:
    mcfg = ModelConfig()
    icfg = IdentifierConfig()
    samples = []
    params = {}
    seed = 100_000 * (1 + len(person_ids)) + (7 if with_watch else 0)
    for pid, ps in zip(person_ids, person_seeds):
        for hand in ("right", "left"):
            params[(pid, hand, "none")] = _hand_params(ps, hand, "none")
            for _ in range(8):
                seed += 1
                samples.append({"prototype": (pid, hand, "none"), "source": "enrollment",
                                "features": src.sample(params[(pid, hand, "none")], seed)})
    if with_watch:
        pid = person_ids[0]
        params[(pid, "right", "watch")] = _hand_params(person_seeds[0], "right", "watch")
        for _ in range(5):
            seed += 1
            samples.append({"prototype": (pid, "right", "watch"), "source": "accessory",
                            "features": src.sample(params[(pid, "right", "watch")], seed)})
    # two adaptive samples on one prototype (they are ordinary samples for the model)
    key = (person_ids[-1], "left", "none")
    for _ in range(2):
        seed += 1
        samples.append({"prototype": key, "source": "adaptive", "features": src.sample(params[key], seed)})
    ident = Identifier(samples, mcfg, icfg)

    def fresh(k, s):
        return src.sample(params[k], s)

    cand: list[tuple[str, list[list[float]], str | None]] = []
    qs = 900_000 + seed
    keys = list(params)
    # clean checks of every prototype (one pattern)
    for i, k in enumerate(keys):
        for attempt in range(6):
            qs += 1
            cand.append((f"check-{i}-{k[1]}-{k[2]}", [fresh(k, qs)], "identified"))
    # multi-pattern checks
    for i, k in enumerate(keys[:3]):
        for attempt in range(6):
            qs += 3
            cand.append((f"multi-{i}", [fresh(k, qs), fresh(k, qs + 1)], None))
            cand.append((f"multi3-{i}", [fresh(k, qs), fresh(k, qs + 1), fresh(k, qs + 2)], None))
    # hand unclear: blend of the same person's right and left
    for pid in person_ids:
        r = (pid, "right", "none")
        l = (pid, "left", "none")
        for w in (0.5, 0.45, 0.55, 0.4, 0.6, 0.35, 0.65):
            qs += 6
            pats = [_blend(fresh(r, qs + j), fresh(l, qs + 3 + j), w) for j in range(3)]
            cand.append((f"handUnclear-{pid[:4]}", pats, "identifiedHandUnclear"))
            cand.append((f"handUnclearShort-{pid[:4]}", pats[:1], "needMorePatterns"))
    # hand head disagreement: same blends, looking for marginHand >= handMargin with the head disagreeing
    # ambiguous: blend of two people
    if len(person_ids) >= 2:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            if b >= len(person_ids):
                continue
            ka = (person_ids[a], "right", "none")
            kb = (person_ids[b], "right", "none")
            for w in (0.5, 0.45, 0.55, 0.4, 0.6):
                qs += 6
                pats = [_blend(fresh(ka, qs + j), fresh(kb, qs + 3 + j), w) for j in range(3)]
                cand.append(("ambiguous", pats, "ambiguous"))
    # strangers
    for s in STRANGER_SEEDS:
        stranger = _hand_params(s, "right", "none")
        qs += 3
        pats = [src.sample(stranger, qs + j) for j in range(3)]
        cand.append(("stranger", pats, "noMatch"))
        cand.append(("strangerShort", pats[:2], "needMorePatterns"))
    for pats in _midpoints(ident, "ambiguous"):
        cand.append(("ambiguousMidpoint", pats, "ambiguous"))
    for pats in _midpoints(ident, "handUnclear"):
        cand.append(("handUnclearMidpoint", pats, "identifiedHandUnclear"))

    def head_disagrees(q: dict) -> bool:
        hand = q["bestPrototype"][1] if q["bestPrototype"] else None
        head_ok = q["meanHandLogOdds"] > 0 if hand == "right" else q["meanHandLogOdds"] < 0
        return (hand is not None and not head_ok and q["marginHand"] is not None
                and q["marginHand"] >= icfg.hand_margin and q["marginName"] >= icfg.name_margin
                and q["bestDistance"] <= ident.model.abs_threshold)

    for x in _head_disagreement_candidates(ident):
        cand.append(("handHeadDisagrees", [x], head_disagrees))
    queries = _select_queries(ident, cand)
    hd = [q for q in queries if q["name"] == "handHeadDisagrees"]
    kinds = {q["outcome"]["type"] if q["outcome"]["type"] != "unknown" else q["outcome"]["reason"] for q in queries}
    print(f"  {tag}: {len(samples)} samples, {len(ident.prototypes)} prototypes, {len(queries)} queries, outcomes {sorted(kinds)}"
          f", hand-head disagreement {'found' if hd else 'not found'}")
    return _model_block(samples, ident, mcfg, icfg, queries)


def make_model(cal: dict) -> dict:
    src = FeatureSource(cal)
    main = make_model_set(src, PERSON_IDS, [41, 42, 43], True, "main")
    single = make_model_set(src, ["E4A1B2C3-D4E5-4F60-8172-93A4B5C6D7E8"], [44], False, "singlePerson")
    out = {"generator": generator_info("model_vectors.json"),
           "note": "queries are decided on all their patterns at once (m = len(patterns)); infinities are null"}
    out.update(main)
    out["singlePerson"] = single
    return out


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    print("rng_vectors.json")
    write_json(out / "rng_vectors.json", make_rng())
    print("challenge_vectors.json")
    write_json(out / "challenge_vectors.json", make_challenges())
    print("dsp_vectors.json")
    write_json(out / "dsp_vectors.json", make_dsp())
    print("pipeline_vectors.json")
    pipe, cal = make_pipeline()
    write_json(out / "pipeline_vectors.json", pipe)
    print("model_vectors.json")
    write_json(out / "model_vectors.json", make_model(cal))
    total = 0
    for name in ("rng_vectors.json", "challenge_vectors.json", "dsp_vectors.json", "pipeline_vectors.json", "model_vectors.json"):
        data = (out / name).read_bytes()
        total += len(data)
        print(f"  {name:24s} {len(data):9d} bytes  sha256 {hashlib.sha256(data).hexdigest()[:16]}")
    print(f"  total {total} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
