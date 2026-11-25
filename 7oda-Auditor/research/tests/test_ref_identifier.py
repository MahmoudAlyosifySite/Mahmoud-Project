"""SPEC 9.1 and 9.3-9.6: prototypes, hand head, sequential decision, adaptive updates."""

import math

import numpy as np
import pytest

from auditor_ref.identifier import (Identifier, IdentifierConfig, hand_challenge_passed, prototype_key,
                                    request_hand, sort_prototypes)
from auditor_ref.model import GaussianModel
from auditor_ref.rng import Xoshiro256

D = 6
PROTOS = [("A", "right", "none"), ("A", "right", "watch"), ("A", "left", "none"),
          ("B", "right", "none"), ("B", "left", "none"), ("C", "right", "none")]


def _samples(protos=PROTOS, per=4, seed=0, d=D):
    rs = np.random.RandomState(seed)
    out = []
    for i, p in enumerate(protos):
        center = rs.normal(0, 3, size=d)
        for _ in range(per):
            out.append({"prototype": p, "features": list(center + rs.normal(0, 1, size=d)),
                        "source": "accessory" if p[2] != "none" else "enrollment"})
    return out


def _stub(table, hands=None, abs_threshold=50.0, protos=PROTOS, with_head=True):
    """Identifier whose distances / hand log-odds come from lookup tables keyed by pattern[0]."""
    ident = Identifier(_samples(protos))
    assert ident.prototypes == sorted(protos, key=prototype_key)
    ident.model.abs_threshold = abs_threshold
    ident.distances = lambda x: list(table[int(x[0])])
    if with_head:
        ident.hand_model = object()
        ident.hand_log_odds = lambda x: float(hands[int(x[0])])
    else:
        ident.hand_model = None
    return ident


def test_prototype_order():
    shuffled = [PROTOS[i] for i in (5, 2, 4, 0, 3, 1)] + [("A", "left", "other"), ("A", "left", "ring"), ("0", "left", "bracelet")]
    assert sort_prototypes(shuffled) == [("0", "left", "bracelet"), ("A", "right", "none"), ("A", "right", "watch"),
                                         ("A", "left", "none"), ("A", "left", "ring"), ("A", "left", "other"),
                                         ("B", "right", "none"), ("B", "left", "none"), ("C", "right", "none")]


def test_identified():
    ident = _stub({0: [5, 30, 20, 40, 45, 60]}, {0: 2.0})
    d = ident.decide([[0]])
    assert d["outcome"] == {"type": "identified", "personID": "A", "hand": "right", "reason": None}
    assert d["bestPrototype"] == ("A", "right", "none")
    assert (d["bestDistance"], d["marginName"], d["marginHand"], d["handLogOdds"]) == (5, 35, 15, 2.0)
    assert d["patternsUsed"] == 1 and d["absThreshold"] == 50.0
    assert [p["distance"] for p in d["prototypeDistances"]] == [5, 30, 20, 40, 45, 60]
    probs = [s["probability"] for s in d["personScores"]]
    assert [s["personID"] for s in d["personScores"]] == ["A", "B", "C"]
    assert sum(probs) == pytest.approx(1.0) and probs == sorted(probs, reverse=True)
    e = np.exp(-np.array([5, 40, 60]) / 2.0)
    np.testing.assert_allclose(probs, e / e.sum(), rtol=1e-12)


def test_accessory_prototype_wins_without_penalising_same_person():
    ident = _stub({0: [9, 4, 20, 40, 45, 60]}, {0: 1.0})
    d = ident.decide([[0]])
    assert d["outcome"]["type"] == "identified" and d["bestPrototype"] == ("A", "right", "watch")
    assert d["marginName"] == 36 and d["marginHand"] == 16   # A/right/none (9) is ignored by both margins


def test_hand_unclear_needs_more_then_hand_unclear():
    t = {0: [10, 30, 11.5, 40, 45, 60], 1: [10, 30, 11.5, 40, 45, 60], 2: [10, 30, 11.5, 40, 45, 60]}
    ident = _stub(t, {0: 1.0, 1: 1.0, 2: 1.0})
    assert ident.decide([[0]])["outcome"]["type"] == "needMorePatterns"
    assert ident.decide([[0], [1]])["outcome"]["type"] == "needMorePatterns"
    d = ident.decide([[0], [1], [2]])
    assert d["outcome"] == {"type": "identifiedHandUnclear", "personID": "A", "hand": None, "reason": None}
    assert d["marginHand"] == pytest.approx(1.5)


def test_ambiguous_and_no_match():
    ident = _stub({0: [10, 30, 25, 13, 45, 60]}, {0: 1.0})
    d = ident.decide([[0]] * 3)
    assert d["outcome"] == {"type": "unknown", "personID": None, "hand": None, "reason": "ambiguous"}
    assert d["marginName"] == 3
    ident = _stub({0: [80, 90, 95, 120, 125, 160]}, {0: 1.0})
    assert ident.decide([[0]] * 3)["outcome"]["reason"] == "noMatch"
    assert ident.decide([[0]] * 2)["outcome"]["type"] == "needMorePatterns"
    # both absOK and nameOK false -> noMatch takes precedence
    ident = _stub({0: [80, 90, 95, 82, 125, 160]}, {0: 1.0})
    assert ident.decide([[0]] * 3)["outcome"]["reason"] == "noMatch"


def test_sequential_means():
    t = {0: [5, 30, 20, 40, 45, 60], 1: [25, 30, 10, 40, 45, 60]}
    ident = _stub(t, {0: 3.0, 1: -1.0})
    d = ident.decide([[0], [1]])
    assert [p["distance"] for p in d["prototypeDistances"]] == [15, 30, 15, 40, 45, 60]
    assert d["bestPrototype"] == ("A", "right", "none")      # tie 15/15 -> lowest index
    assert d["handLogOdds"] == 1.0 and d["marginHand"] == 0.0
    assert d["distancesPerPattern"] == [t[0], t[1]] and d["handLogOddsPerPattern"] == [3.0, -1.0]


def test_hand_head_disagreement():
    ident = _stub({0: [5, 30, 20, 40, 45, 60]}, {0: -0.5})
    assert ident.decide([[0]])["outcome"]["type"] == "needMorePatterns"
    assert ident.decide([[0]] * 3)["outcome"]["type"] == "identifiedHandUnclear"
    # left prototype wins and the head agrees (h < 0)
    ident = _stub({0: [20, 30, 5, 40, 45, 60]}, {0: -0.5})
    assert ident.decide([[0]])["outcome"] == {"type": "identified", "personID": "A", "hand": "left", "reason": None}
    # h exactly 0 agrees with neither hand
    ident = _stub({0: [20, 30, 5, 40, 45, 60]}, {0: 0.0})
    assert ident.decide([[0]])["outcome"]["type"] == "needMorePatterns"
    # without a hand head the hand margin alone decides
    ident = _stub({0: [20, 30, 5, 40, 45, 60]}, with_head=False)
    d = ident.decide([[0]])
    assert d["outcome"]["type"] == "identified" and d["handLogOdds"] == 0.0


def test_single_person_infinite_margins():
    protos = [("A", "right", "none")]
    ident = _stub({0: [12.0], 1: [400.0]}, protos=protos, with_head=False)
    d = ident.decide([[0]])
    assert d["outcome"] == {"type": "identified", "personID": "A", "hand": "right", "reason": None}
    assert math.isinf(d["marginName"]) and math.isinf(d["marginHand"])
    assert d["personScores"] == [{"personID": "A", "probability": 1.0, "bestDistance": 12.0}]
    assert ident.decide([[1]] * 3)["outcome"]["reason"] == "noMatch"


def test_single_person_both_hands_real_model():
    protos = [("P", "right", "none"), ("P", "left", "none")]
    s = _samples(protos, per=6, seed=3)
    ident = Identifier(s)
    assert ident.hand_model is not None and len(ident.prototypes) == 2
    x = s[0]["features"]
    d = ident.decide([x])
    assert math.isinf(d["marginName"]) and d["bestPrototype"] == ("P", "right", "none")
    assert d["handLogOdds"] == pytest.approx((ident.hand_model.distances(x)[1] - ident.hand_model.distances(x)[0]) / 2)


def test_no_people_and_no_patterns():
    d = Identifier([]).decide([[0.0] * D])
    assert d["outcome"] == {"type": "unknown", "personID": None, "hand": None, "reason": "noPeople"}
    assert Identifier(_samples()).decide([])["outcome"]["type"] == "needMorePatterns"


def test_hand_head_requires_two_samples_per_hand():
    s = _samples([("A", "right", "none"), ("B", "right", "none")], per=3)
    assert Identifier(s).hand_model is None
    s2 = s + [{"prototype": ("A", "left", "none"), "features": [0.0] * D, "source": "enrollment"}]
    assert Identifier(s2).hand_model is None
    s3 = s2 + [{"prototype": ("B", "left", "none"), "features": [1.0] * D, "source": "enrollment"}]
    ident = Identifier(s3)
    assert ident.hand_model is not None and ident.hand_model.class_counts == [6, 2]


def test_real_model_end_to_end():
    s = _samples(per=8, seed=11)
    ident = Identifier(s)
    rs = np.random.RandomState(5)
    hits = 0
    for i in range(0, len(s), 8):
        p = s[i]["prototype"]
        center = np.mean([q["features"] for q in s if q["prototype"] == p], axis=0)
        d = ident.decide([list(center + rs.normal(0, 0.3, size=D))])
        hits += d["bestPrototype"] == p
        assert np.isfinite(d["bestDistance"])
    assert hits >= 5
    stranger = list(np.full(D, 40.0))
    assert ident.decide([stranger] * 3)["outcome"] == {"type": "unknown", "personID": None, "hand": None, "reason": "noMatch"}
    # distances go through the trained model
    m = GaussianModel.fit([q["features"] for q in s], [ident.prototypes.index(tuple(q["prototype"])) for q in s], 6)
    np.testing.assert_allclose(ident.distances(stranger), m.distances(stranger), rtol=1e-12)


def test_adaptive_eligibility():
    ident = _stub({0: [5, 30, 20, 40, 45, 60], 1: [5, 30, 20, 16, 45, 60], 2: [38, 60, 50, 70, 75, 90]}, {0: 1, 1: 1, 2: 1})
    d = ident.decide([[0]])
    assert ident.adaptive_eligible(d)
    new = ident.adaptive_samples(d, [[0.5] * D], now="2026-11-24T10:00:00Z")
    assert new == [{"prototype": ("A", "right", "none"), "features": [0.5] * D, "source": "adaptive",
                    "createdAt": "2026-11-24T10:00:00Z"}]
    d = ident.decide([[1]])                      # marginName 11 < 12
    assert d["outcome"]["type"] == "identified" and not ident.adaptive_eligible(d)
    d = ident.decide([[2]])                      # 38 > 0.75 * 50
    assert d["outcome"]["type"] == "identified" and not ident.adaptive_eligible(d)
    assert ident.adaptive_samples(d, [[0.0] * D]) == []
    d = _stub({0: [10, 30, 11, 40, 45, 60]}, {0: 1}).decide([[0]] * 3)
    assert d["outcome"]["type"] == "identifiedHandUnclear" and not ident.adaptive_eligible(d)


def test_adaptive_cap():
    base = _samples(per=2)
    proto = ("B", "left", "none")
    adaptive = [{"prototype": proto, "features": [float(i)] * D, "source": "adaptive",
                 "createdAt": "2026-11-24T10:%02d:00Z" % i} for i in range(25)]
    other = [{"prototype": PROTOS[0], "features": [9.0] * D, "source": "adaptive", "createdAt": "2026-11-24T09:00:00Z"}]
    out = Identifier.apply_adaptive_cap(base + adaptive[::-1] + other, 20)
    kept = [s for s in out if s["prototype"] == proto and s["source"] == "adaptive"]
    assert len(kept) == 20 and {s["features"][0] for s in kept} == set(float(i) for i in range(5, 25))
    assert all(s in out for s in base) and other[0] in out
    assert len(out) == len(base) + 21


def test_updated_retrains_and_caps():
    cfg = IdentifierConfig(adaptive_max_per_prototype=2)
    s = _samples(per=6, seed=2)
    ident = Identifier(s, config=cfg)
    p0 = s[0]["prototype"]
    x = list(np.mean([q["features"] for q in s if q["prototype"] == p0], axis=0))
    d = ident.decide([x])
    if not ident.adaptive_eligible(d):
        pytest.skip("synthetic draw not confident enough")
    new = ident
    for i in range(3):
        new = new.updated(d, [x], now="2026-11-24T10:0%d:00Z" % i)
    assert sum(q["source"] == "adaptive" for q in new.samples) == 2
    assert len(new.samples) == len(s) + 2


def test_hand_challenge():
    a = [request_hand(Xoshiro256(i)) for i in range(200)]
    assert set(a) == {"right", "left"}
    assert request_hand(Xoshiro256(1)) == ("right" if Xoshiro256(1).uniform() < 0.5 else "left")
    ok = {"outcome": {"type": "identified", "personID": "A", "hand": "left", "reason": None}}
    assert hand_challenge_passed(ok, "left") and not hand_challenge_passed(ok, "right")
    unclear = {"outcome": {"type": "identifiedHandUnclear", "personID": "A", "hand": None, "reason": None}}
    assert not hand_challenge_passed(unclear, "left")
