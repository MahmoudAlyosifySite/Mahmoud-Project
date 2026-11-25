"""The committed SPEC 11 fixtures agree with the reference implementation.

Recomputes every expected value from the fixture inputs. Tolerances are tight
(1e-9 relative) but allow last-bit differences of libm between machines.
Skipped when the fixture directory has not been generated yet.
"""

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from auditor_ref import dsp, features, quality, signal
from auditor_ref.challenge import generate
from auditor_ref.identifier import Identifier, IdentifierConfig
from auditor_ref.model import ModelConfig
from auditor_ref.recording import sort_dedup
from auditor_ref.rng import SplitMix64, Xoshiro256, seed_to_str, str_to_seed

RESEARCH = Path(__file__).resolve().parents[1]
FIXTURES = RESEARCH.parent / "AuditorCore" / "Tests" / "AuditorCoreTests" / "Fixtures"
NAMES = ("rng_vectors.json", "challenge_vectors.json", "dsp_vectors.json", "pipeline_vectors.json", "model_vectors.json")
SEED_RE = re.compile(r"^0x[0-9a-f]{16}$")


def _reject(name):
    raise ValueError(f"non-finite constant {name} in fixture")


def _load(name):
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"{name} not generated")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject)


def _inf(v):
    return math.inf if v is None else v


def _close(a, b, rel=1e-9):
    a = np.asarray([_inf(v) for v in np.ravel(np.asarray(a, dtype=object))], dtype=float)
    b = np.asarray([_inf(v) for v in np.ravel(np.asarray(b, dtype=object))], dtype=float)
    assert a.shape == b.shape
    fin = np.isfinite(a) & np.isfinite(b)
    assert np.array_equal(np.isinf(a), np.isinf(b))
    assert np.all(np.abs(a[fin] - b[fin]) <= rel * np.maximum(1.0, np.abs(b[fin]))), np.max(np.abs(a[fin] - b[fin]))


@pytest.mark.parametrize("name", NAMES)
def test_files_are_clean_json(name):
    data = _load(name)
    assert data["generator"]["script"] == "research/scripts/make_fixtures.py"
    assert (FIXTURES / name).stat().st_size < 3_000_000


def test_total_size():
    if not all((FIXTURES / n).exists() for n in NAMES):
        pytest.skip("fixtures not generated")
    assert sum((FIXTURES / n).stat().st_size for n in NAMES) < 3_000_000


def test_rng_vectors():
    data = _load("rng_vectors.json")
    seeds = [str_to_seed(c["seed"]) for c in data["cases"]]
    assert 0 in seeds and 2 ** 64 - 1 in seeds
    for c in data["cases"]:
        assert SEED_RE.match(c["seed"])
        seed = str_to_seed(c["seed"])
        sm = SplitMix64(seed)
        assert c["splitmix"] == [seed_to_str(sm.next()) for _ in range(5)]
        x = Xoshiro256(seed)
        assert c["xoshiro"] == [seed_to_str(x.next()) for _ in range(10)]
        u = Xoshiro256(seed)
        assert c["uniform"] == [u.uniform() for _ in range(5)]
        for e in c["ints"]:
            g = Xoshiro256(seed)
            assert e["values"] == [g.int_in_range(e["lo"], e["hi"]) for _ in range(10)]
        for e in c["uniformRange"]:
            g = Xoshiro256(seed)
            assert e["values"] == [g.uniform_range(e["a"], e["b"]) for _ in range(5)]


def test_challenge_vectors():
    data = _load("challenge_vectors.json")
    assert len(data["cases"]) >= 50
    seeds = {str_to_seed(c["seed"]) for c in data["cases"]}
    assert {0, 1, 2 ** 64 - 1} <= seeds
    for c in data["cases"]:
        assert generate(str_to_seed(c["seed"])) == c["spec"]


def test_dsp_vectors():
    data = _load("dsp_vectors.json")
    for b in data["biquads"]:
        _close(dsp.biquad(b["kind"], b["fc"], b["fs"], b["q"]), b["coeffs"], 1e-12)
    ns = set()
    for f in data["filters"]:
        c = dsp.biquad(f["kind"], f["fc"], f["fs"], f["q"])
        ns.add(len(f["x"]))
        _close(dsp.lfilter(c, f["x"]), f["lfilter"])
        _close(dsp.filtfilt_z(c, f["x"], f["pad"], f["dcGain"]), f["filtfilt"])
    assert {0, 1, 2} <= ns and any(0 < n < 30 for n in ns)
    for r in data["resample"]:
        vals, miss = dsp.resample(r["t"], r["v"], r["grid"], r["gapTolerance"])
        assert list(vals) == r["values"] and [bool(m) for m in miss] == r["missing"]
    for case in data["sortDedup"]:
        out = sort_dedup(case["input"])
        assert {k: list(v) for k, v in out.items()} == case["output"]


def test_pipeline_vectors():
    data = _load("pipeline_vectors.json")
    names = [c["name"] for c in data["cases"]]
    assert len(names) >= 8 and len(set(names)) == len(names)
    assert any(c["calibration"] for c in data["cases"]) and any(not c["calibration"] for c in data["cases"])
    assert {"handheld", "onTable"} == {c["mode"] for c in data["cases"]}
    for c in data["cases"]:
        rec, cal, e = c["recording"], c["calibration"], c["expected"]
        assert re.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$", rec["createdAt"])
        p = signal.process(rec)
        rep = quality.evaluate(p, calibration=cal, mode=c["mode"])
        assert (p.n, p.lag, list(p.quiet_range), list(p.event_range)) == (e["n"], e["lag"], e["quietRange"], e["eventRange"]), c["name"]
        assert rep["issues"] == e["issues"], c["name"]
        assert rep["metrics"]["lag"] == e["metrics"]["lag"]
        _close([p.ncc, p.peak_margin, p.noise_accel, p.noise_gyro, p.gap_fraction, p.sample_rate],
               [e["ncc"], e["peakMargin"], e["noiseAccel"], e["noiseGyro"], e["gapFraction"], e["sampleRate"]])
        keys = [k for k in quality.METRIC_KEYS if k != "lag"]
        _close([rep["metrics"][k] for k in keys], [e["metrics"][k] for k in keys])
        _close(features.extract_absolute(p), e["featuresAbsolute"])
        _close(features.extract(p, cal), e["features"])


def _check_model_block(blk):
    mcfg = ModelConfig(blk["config"]["shrinkageMin"], blk["config"]["looQuantile"], blk["config"]["absFactor"])
    ic = blk["identifierConfig"]
    icfg = IdentifierConfig(ic["nameMargin"], ic["handMargin"], ic["maxPatterns"], ic["adaptiveMaxPerPrototype"],
                            ic["adaptiveNameFactor"], ic["adaptiveAbsFactor"])
    samples = [{"prototype": (s["personID"], s["hand"], s["accessory"]), "features": s["features"], "source": s["source"]}
               for s in blk["samples"]]
    ident = Identifier(samples, mcfg, icfg)
    ex = blk["expected"]
    assert [list(p) for p in ident.prototypes] == ex["prototypes"]
    _close(ident.model.mean, ex["mean"])
    _close(ident.model.scale, ex["scale"])
    _close([ident.model.shrinkage, ident.model.abs_threshold], [ex["shrinkage"], ex["absThreshold"]])
    if ex["handShrinkage"] is None:
        assert ident.hand_model is None
    else:
        _close([ident.hand_model.shrinkage], [ex["handShrinkage"]])
    kinds = set()
    for q in blk["queries"]:
        d = ident.decide(q["patterns"])
        _close(d["distancesPerPattern"], q["distances"])
        _close(d["handLogOddsPerPattern"], q["handLogOdds"])
        _close([d["bestDistance"], d["marginName"], d["marginHand"]], [q["bestDistance"], q["marginName"], q["marginHand"]])
        assert d["outcome"] == q["outcome"], q["name"]
        assert ident.adaptive_eligible(d) == q["adaptiveEligible"]
        o = q["outcome"]
        kinds.add(o["type"] if o["type"] != "unknown" else o["reason"])
    return ident, kinds


def test_model_vectors():
    data = _load("model_vectors.json")
    ident, kinds = _check_model_block(data)
    persons = {p[0] for p in ident.prototypes}
    assert len(persons) >= 3 and {p[1] for p in ident.prototypes} == {"right", "left"}
    assert any(p[2] != "none" for p in ident.prototypes)
    assert {"identified", "identifiedHandUnclear", "needMorePatterns", "noMatch", "ambiguous"} <= kinds
    assert any(len(q["patterns"]) > 1 for q in data["queries"])
    single, kinds1 = _check_model_block(data["singlePerson"])
    assert len({p[0] for p in single.prototypes}) == 1
    assert {"identified", "noMatch"} <= kinds1


@pytest.mark.slow
def test_generator_is_deterministic(tmp_path):
    script = RESEARCH / "scripts" / "make_fixtures.py"
    for out in (tmp_path / "a", tmp_path / "b"):
        subprocess.run([sys.executable, str(script), "--out", str(out)], check=True, capture_output=True)
    for n in NAMES:
        assert (tmp_path / "a" / n).read_bytes() == (tmp_path / "b" / n).read_bytes()
