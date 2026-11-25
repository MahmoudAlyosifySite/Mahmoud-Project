#!/usr/bin/env python3
"""Write the v1 parity fixture of SPEC 14.5 (``v1_vectors.json``).

- ``tensor``: the encoder input of SPEC 14.1 for three recordings of
  ``pipeline_vectors.json`` (positive, negative and large lags), computed by
  ``research/v1/data.py`` (``tensor_from_processed`` on ``auditor_ref.signal.process``);
  all 2560 float32 values, row-major ``[channel][u]``.
- ``identifier``: the on-device decision of SPEC 14.3 on clustered synthetic unit
  embeddings (3 persons, both hands, one accessory prototype, strangers, multi-pattern
  queries). Prototypes and leave-one-out similarities come from
  ``v1.decision.Prototypes``, the right/left model from ``v1.evaluate_v1.pooled_hand_model``;
  the decision itself is the SPEC 14.3 rule in cosine-distance units, written out below.

Rules as in ``make_fixtures.py``: full-precision floats, infinities as ``null``, never
NaN, deterministic output; queries whose discrete outcomes are within a relative
margin of 1e-3 of flipping are re-drawn.

Usage: ``python scripts/make_v1_fixtures.py [--out DIR]``
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
RESEARCH = HERE.parents[1]
sys.path.insert(0, str(RESEARCH))

import numpy as np  # noqa: E402

from auditor_ref import signal  # noqa: E402
from v1 import data as v1data  # noqa: E402
from v1.decision import Prototypes  # noqa: E402
from v1.evaluate_v1 import pooled_hand_model  # noqa: E402

FIXTURES = RESEARCH.parent / "AuditorCore" / "Tests" / "AuditorCoreTests" / "Fixtures"
TENSOR_CASES = ("handheld", "wrongChallenge", "lateOnset")
DIM = 128
CONFIG = {"looQuantile": 0.95, "slack": 0.0, "nameMargin": 0.03, "handMargin": 0.0,
          "handShrinkage": 0.1, "maxPatterns": 3, "adaptiveMaxPerPrototype": 20}
ADAPTIVE_NAME_FACTOR = 2.0
ADAPTIVE_DISTANCE_MARGIN = 0.05
SCORE_TEMPERATURE = 0.05
TIE_MARGIN = 1e-3
PERSONS = ("3C9E2A71-0B4D-4E8F-9A1B-2C3D4E5F6071",
           "8A7B6C5D-4E3F-4A2B-9C1D-0E1F2A3B4C5D",
           "B1C2D3E4-F5A6-4B7C-8D9E-0F1A2B3C4D5E")


def clean(obj):
    if isinstance(obj, float):
        if math.isnan(obj):
            raise ValueError("NaN in fixture")
        return None if math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return clean(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


# ---------------------------------------------------------------------------
# Tensor cases (SPEC 14.1)
# ---------------------------------------------------------------------------

def tensor_cases() -> list[dict]:
    pipeline = json.loads((FIXTURES / "pipeline_vectors.json").read_text())
    by_name = {c["name"]: c for c in pipeline["cases"]}
    out = []
    for name in TENSOR_CASES:
        rec = by_name[name]["recording"]
        p = signal.process(rec)
        x = v1data.tensor_from_processed(p)
        assert x.shape == (10, 256) and x.dtype == np.float32
        out.append({"name": name, "lag": int(p.alignment.lag), "n": int(p.n), "recording": rec,
                    "tensor": [float(v) for v in x.ravel()]})
    return out


# ---------------------------------------------------------------------------
# Identifier cases (SPEC 14.3)
# ---------------------------------------------------------------------------

def unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def as_float32_unit(v: np.ndarray) -> list[float]:
    """Like a Core ML output: unit length, then rounded to float32."""
    return [float(t) for t in unit(v).astype(np.float32)]


class World:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        c0 = unit(rng.normal(size=DIM))
        c1 = unit(rng.normal(size=DIM))
        c2 = unit(c1 + 0.55 * unit(rng.normal(size=DIM)))   # persons 1 and 2 are close
        self.centers = [c0, c1, c2]
        self.hand_dirs = [unit(rng.normal(size=DIM)) for _ in PERSONS]
        self.watch_dir = unit(rng.normal(size=DIM))

    def draw(self, person: int, hand: str, noise: float, watch: bool = False, hand_weight: float = 0.45):
        sign = 1.0 if hand == "right" else -1.0
        v = self.centers[person] + hand_weight * sign * self.hand_dirs[person]
        if watch:
            v = v + 0.4 * self.watch_dir
        v = v + noise * self.rng.normal(size=DIM) / math.sqrt(DIM)
        return as_float32_unit(v)

    def stranger(self, noise: float):
        return as_float32_unit(unit(self.rng.normal(size=DIM)) + noise * self.rng.normal(size=DIM) / math.sqrt(DIM))


class Model:
    """SPEC 14.3 identifier state from the research code."""

    def __init__(self, samples: list[dict]):
        emb = np.array([s["embedding"] for s in samples], dtype=np.float64)
        keys = [(s["personID"], s["hand"], s["accessory"]) for s in samples]
        self.protos = Prototypes.fit(emb, keys)
        self.keys = [tuple(k) for k in self.protos.keys]
        self.sim_threshold = self.protos.abs_threshold(CONFIG["looQuantile"], CONFIG["slack"])
        self.abs_threshold = 1.0 - self.sim_threshold
        unit_emb = np.array([unit(e) for e in emb])
        hands = np.array([0 if s["hand"] == "right" else 1 for s in samples])
        self.hand = pooled_hand_model(unit_emb, hands, CONFIG["handShrinkage"])

    def similarities(self, pattern) -> list[float]:
        e = unit(np.asarray(pattern, dtype=np.float64))
        return [float(e @ p) for p in self.protos.matrix]

    def hand_log_odds(self, pattern) -> float:
        if self.hand is None:
            return 0.0
        w, b = self.hand
        return float(unit(np.asarray(pattern, dtype=np.float64)) @ w + b)

    def decide(self, patterns: list) -> tuple[dict, list[tuple[str, float, float, float]]]:
        """Decision dict (Swift ``Decision`` fields) and the discrete comparisons made."""
        m = len(patterns)
        sims = [self.similarities(x) for x in patterns]
        hands = [self.hand_log_odds(x) for x in patterns]
        k_count = len(self.keys)
        sbar = []
        for k in range(k_count):
            s = 0.0
            for j in range(m):
                s = s + sims[j][k]
            sbar.append(s / m)
        dbar = [1.0 - v for v in sbar]
        has_head = self.hand is not None
        hbar = (sum(hands) / m) if has_head else 0.0
        kstar = 0
        for k in range(1, k_count):
            if dbar[k] < dbar[kstar]:
                kstar = k
        person, hand, _ = self.keys[kstar]
        best = dbar[kstar]
        other_person = min((dbar[k] for k in range(k_count) if self.keys[k][0] != person), default=math.inf)
        other_hand = min((dbar[k] for k in range(k_count)
                          if self.keys[k][0] == person and self.keys[k][1] != hand), default=math.inf)
        margin_name = other_person - best
        margin_hand = other_hand - best
        abs_ok = best <= self.abs_threshold
        name_ok = margin_name >= CONFIG["nameMargin"]
        head_ok = (not has_head) or (hbar > 0.0 if hand == "right" else hbar < 0.0)
        hand_ok = margin_hand >= CONFIG["handMargin"] and head_ok
        if abs_ok and name_ok and hand_ok:
            outcome = {"type": "identified", "personID": person, "hand": hand, "reason": None}
        elif m < CONFIG["maxPatterns"]:
            outcome = {"type": "needMorePatterns", "personID": None, "hand": None, "reason": None}
        elif abs_ok and name_ok:
            outcome = {"type": "identifiedHandUnclear", "personID": person, "hand": None, "reason": None}
        elif not abs_ok:
            outcome = {"type": "unknown", "personID": None, "hand": None, "reason": "noMatch"}
        else:
            outcome = {"type": "unknown", "personID": None, "hand": None, "reason": "ambiguous"}
        eligible = (outcome["type"] == "identified"
                    and margin_name >= ADAPTIVE_NAME_FACTOR * CONFIG["nameMargin"]
                    and best <= self.abs_threshold - ADAPTIVE_DISTANCE_MARGIN)
        # Person scores: softmax of max similarity / 0.05, persons in prototype order.
        persons: list[str] = []
        smax: list[float] = []
        for k, key in enumerate(self.keys):
            if key[0] in persons:
                i = persons.index(key[0])
                smax[i] = max(smax[i], sbar[k])
            else:
                persons.append(key[0])
                smax.append(sbar[k])
        logits = [v / SCORE_TEMPERATURE for v in smax]
        top = max(logits)
        ex = [math.exp(v - top) for v in logits]
        total = 0.0
        for v in ex:
            total = total + v
        scores = [{"personID": p, "probability": ex[i] / total, "bestDistance": 1.0 - smax[i]}
                  for i, p in enumerate(persons)]
        order = sorted(range(len(scores)), key=lambda i: (-scores[i]["probability"], i))
        decision = {
            "similarities": sims, "handLogOdds": hands, "outcome": outcome,
            "bestDistance": best, "marginName": margin_name, "marginHand": margin_hand,
            "meanDistances": dbar, "meanHandLogOdds": hbar, "bestPrototype": list(self.keys[kstar]),
            "personScores": [scores[i] for i in order], "adaptiveEligible": eligible,
        }
        second = min((dbar[k] for k in range(k_count) if k != kstar), default=math.inf)
        checks = [("argmin", best, second, 1.0), ("abs", best, self.abs_threshold, 1.0),
                  ("name", margin_name, CONFIG["nameMargin"], 1.0),
                  ("adaptiveName", margin_name, ADAPTIVE_NAME_FACTOR * CONFIG["nameMargin"], 1.0),
                  ("adaptiveAbs", best, self.abs_threshold - ADAPTIVE_DISTANCE_MARGIN, 1.0)]
        if has_head:
            checks.append(("head", hbar, 0.0, 1.0))
        return decision, checks


def safe(checks) -> bool:
    for _, a, b, scale in checks:
        if math.isinf(a) or math.isinf(b):
            continue
        if abs(a - b) / max(abs(a), abs(b), scale) < TIE_MARGIN:
            return False
    return True


def identifier_block() -> dict:
    rng = np.random.default_rng(20261125)
    world = World(rng)
    samples = []
    for p, pid in enumerate(PERSONS):
        for hand in ("right", "left"):
            for _ in range(6):
                samples.append({"personID": pid, "hand": hand, "accessory": "none", "source": "enrollment",
                                "embedding": world.draw(p, hand, 0.7)})
    for _ in range(4):
        samples.append({"personID": PERSONS[0], "hand": "right", "accessory": "watch", "source": "accessory",
                        "embedding": world.draw(0, "right", 0.7, watch=True)})
    model = Model(samples)
    assert model.hand is not None

    # Candidate generators: (label, function returning a list of patterns).
    def owner(p, hand, m, noise, watch=False):
        return lambda: [world.draw(p, hand, noise, watch=watch) for _ in range(m)]

    def handless(p, m):
        return lambda: [world.draw(p, "right", 0.45, hand_weight=0.0) for _ in range(m)]

    def between(m):
        return lambda: [as_float32_unit(unit(np.asarray(world.draw(1, "right", 0.5)))
                                        + 0.8 * unit(np.asarray(world.draw(2, "right", 0.5)))) for _ in range(m)]

    generators = []
    for p in range(3):
        for hand in ("right", "left"):
            generators.append((f"owner-{p}-{hand}", owner(p, hand, 1, 0.6)))
            generators.append((f"noisy2-{p}-{hand}", owner(p, hand, 2, 1.1)))
            generators.append((f"noisy3-{p}-{hand}", owner(p, hand, 3, 0.85)))
    generators.append(("watch-0-right", owner(0, "right", 1, 0.6, watch=True)))
    for p in range(3):
        generators.append((f"handless3-{p}", handless(p, 3)))
    generators.append(("between3", between(3)))
    generators.append(("stranger1", lambda: [world.stranger(0.3)]))
    generators.append(("stranger3", lambda: [world.stranger(0.3) for _ in range(3)]))

    quotas = {"identified": 8, "needMorePatterns": 4, "identifiedHandUnclear": 2,
              "unknown/noMatch": 3, "unknown/ambiguous": 2}
    per_generator = 3
    counts = {k: 0 for k in quotas}
    used_by_gen: dict[str, int] = {}
    queries = []
    for attempt in range(60):
        for label, gen in generators:
            if used_by_gen.get(label, 0) >= per_generator:
                continue
            patterns = gen()
            decision, checks = model.decide(patterns)
            if not safe(checks):
                continue
            o = decision["outcome"]
            cat = o["type"] if o["type"] != "unknown" else "unknown/" + o["reason"]
            if counts[cat] >= quotas[cat] and attempt > 0:
                continue
            counts[cat] += 1
            used_by_gen[label] = used_by_gen.get(label, 0) + 1
            queries.append({"name": f"{label}-{used_by_gen[label]}", "patterns": patterns, **decision})
        if all(counts[k] >= quotas[k] for k in quotas):
            break
    missing = {k: quotas[k] - counts[k] for k in quotas if counts[k] < quotas[k]}
    if missing:
        raise SystemExit(f"could not fill the query quotas: {missing}")
    if not any(len(q["patterns"]) > 1 and q["outcome"]["type"] == "identified" for q in queries):
        raise SystemExit("no multi-pattern identified query")

    w, b = model.hand
    return {
        "config": CONFIG,
        "dimension": DIM,
        "samples": samples,
        "expected": {
            "prototypes": [list(k) for k in model.keys],
            "prototypeVectors": [[float(v) for v in row] for row in model.protos.matrix],
            "loo": [float(v) for v in model.protos.loo],
            "simThreshold": float(model.sim_threshold),
            "absThreshold": float(model.abs_threshold),
            "handWeights": [float(v) for v in w],
            "handBias": float(b),
        },
        "queries": queries,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=FIXTURES)
    args = ap.parse_args()
    doc = {
        "generator": {"script": "research/scripts/make_v1_fixtures.py", "spec": "docs/SPEC.md section 14",
                      "file": "v1_vectors.json"},
        "note": "tensor: float32 values row-major [channel][u]; identifier: decisions in cosine-distance "
                "units (SPEC 14.3), queries decided on all their patterns at once; infinities are null",
        "tensor": {"channels": 10, "length": 256, "patternStart": 25, "cases": tensor_cases()},
        "identifier": identifier_block(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "v1_vectors.json"
    path.write_text(json.dumps(clean(doc), allow_nan=False, separators=(",", ":"), ensure_ascii=False))
    ident = doc["identifier"]
    kinds: dict[str, int] = {}
    for q in ident["queries"]:
        o = q["outcome"]
        key = o["type"] + ("/" + o["reason"] if o["reason"] else "")
        kinds[key] = kinds.get(key, 0) + 1
    print(f"wrote {path} ({path.stat().st_size} bytes): {len(doc['tensor']['cases'])} tensor cases, "
          f"{len(ident['samples'])} samples, {len(ident['queries'])} queries {kinds}")


if __name__ == "__main__":
    main()
