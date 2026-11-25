"""SPEC 14.3 decision rule: prototypes, thresholds, sequential outcomes."""
import numpy as np

from v1 import decision as DC


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _protos():
    rng = np.random.default_rng(0)
    centres = {("bob", "right", "none"): rng.normal(size=8), ("bob", "left", "none"): rng.normal(size=8),
               ("amy", "right", "none"): rng.normal(size=8), ("amy", "left", "none"): rng.normal(size=8),
               ("amy", "right", "watch"): rng.normal(size=8)}
    emb, keys = [], []
    for k, c in centres.items():
        for _ in range(5):
            emb.append(_unit(c + 0.1 * rng.normal(size=8)))
            keys.append(k)
    return DC.Prototypes.fit(np.array(emb), keys), centres


def test_prototype_order_and_normalization():
    p, _ = _protos()
    assert p.keys == [("amy", "right", "none"), ("amy", "right", "watch"), ("amy", "left", "none"),
                      ("bob", "right", "none"), ("bob", "left", "none")]
    np.testing.assert_allclose(np.linalg.norm(p.matrix, axis=1), 1.0)
    assert p.loo.size == 25 and np.all(p.loo > 0.5)
    assert p.abs_threshold(0.95, 0.1) == np.quantile(p.loo, 0.05) - 0.1


def test_outcomes():
    p, c = _protos()
    right = np.array([[3.0, -3.0, -5.0]])
    left = np.array([[-3.0, 3.0, -5.0]])
    e = _unit(c[("bob", "left", "none")])[None]
    out = DC.decide_one(e, left, p, abs_threshold=0.5, name_margin=0.05, hand_margin=0.05)
    assert out["outcome"] == "identified" and out["prototype"] == ("bob", "left", "none")
    # hand head disagrees -> one more pattern, then hand unclear
    out = DC.decide_one(e, right, p, 0.5, 0.05, 0.05)
    assert out["outcome"] == "needMorePatterns"
    out = DC.decide_one(np.repeat(e, 3, 0), np.repeat(right, 3, 0), p, 0.5, 0.05, 0.05)
    assert out["outcome"] == "identifiedHandUnclear"
    # a stranger far from everyone -> noMatch
    stranger = _unit(np.random.default_rng(5).normal(size=8))[None]
    out = DC.decide_one(np.repeat(stranger, 3, 0), np.repeat(right, 3, 0), p, 0.99, 0.05, 0.05)
    assert out["outcome"] == "unknown(noMatch)"
    # halfway between two people -> ambiguous
    mid = _unit(_unit(c[("bob", "left", "none")]) + _unit(c[("amy", "left", "none")]))[None]
    out = DC.decide_one(np.repeat(mid, 3, 0), np.repeat(left, 3, 0), p, -1.0, 0.5, 0.0)
    assert out["outcome"] == "unknown(ambiguous)"
    # the watch prototype still names amy, with her right hand
    w = _unit(c[("amy", "right", "watch")])[None]
    out = DC.decide_one(w, right, p, 0.5, 0.05, 0.05)
    assert out["outcome"] == "identified" and out["prototype"][:2] == ("amy", "right")
    two = DC.decide_one(w, np.array([[0.0, 0.0, 9.0]]), p, 0.5, 0.05, 0.05)
    assert two["twoHanded"]


def test_vectorized_decide_matches_single_check_rule():
    p, c = _protos()
    rng = np.random.default_rng(1)
    keys = list(c)
    n = 300
    emb = np.zeros((n, 3, 8))
    hand = np.zeros((n, 3, 3))
    for i in range(n):
        k = keys[rng.integers(len(keys))]
        emb[i] = _unit(c[k] + rng.normal(0, rng.uniform(0.2, 2.0), (3, 8)))
        hand[i] = rng.normal(0, 2, (3, 3))
    n_pat = rng.integers(1, 4, n)
    st = DC.SequentialStats.build(np.einsum("nmd,kd->nmk", emb, p.matrix), hand[:, :, 0] - hand[:, :, 1],
                                  p.person, p.hand, n_pat)
    thr, nm, hm = 0.6, 0.08, 0.04
    outcome, pp, ph, used = DC.decide(st, thr, nm, hm, 3)
    names = {DC.IDENTIFIED: "identified", DC.HAND_UNCLEAR: "identifiedHandUnclear",
             DC.UNKNOWN_NO_MATCH: "unknown(noMatch)", DC.UNKNOWN_AMBIGUOUS: "unknown(ambiguous)"}
    for i in range(n):
        for m in range(1, n_pat[i] + 1):
            # the phone stops at the first decisive pattern; the last available one is final
            out = DC.decide_one(emb[i, :m], hand[i, :m], p, thr, nm, hm, max_patterns=n_pat[i])
            if out["outcome"] != "needMorePatterns":
                break
        assert names[outcome[i]] == out["outcome"], i
        assert used[i] == m
        key = out["prototype"]
        assert p.persons[pp[i]] == key[0] and ph[i] == DC.HAND_ORDER[key[1]]


def test_metrics_and_tuning():
    p, c = _protos()
    rng = np.random.default_rng(2)
    keys = [k for k in c if k[2] == "none"]
    rows, tp, th, strangers = [], [], [], []
    for i in range(200):
        if i % 4 == 0:
            rows.append(_unit(rng.normal(size=(3, 8))))
            tp.append(-1)
            th.append(0)
            strangers.append(True)
        else:
            k = keys[i % len(keys)]
            rows.append(_unit(c[k] + rng.normal(0, 0.3, (3, 8))))
            tp.append(p.persons.index(k[0]))
            th.append(DC.HAND_ORDER[k[1]])
            strangers.append(False)
    emb = np.array(rows)
    tp, th, strangers = np.array(tp), np.array(th), np.array(strangers)
    hs = np.where(th == 0, 2.0, -2.0)[:, None].repeat(3, 1)
    st = DC.SequentialStats.build(np.einsum("nmd,kd->nmk", emb, p.matrix), hs, p.person, p.hand)
    res = DC.tune_thresholds(st, p.abs_threshold(), tp, th, strangers, np.arange(0, 0.5, 0.05),
                             np.arange(0, 0.2, 0.02), np.arange(0, 0.2, 0.04))
    met = res["tuning_metrics"]
    assert res["met_target"] and met["stranger_accept"] <= 0.02
    assert met["top1"] > 0.9 and met["name_and_hand"] <= met["top1"]
    assert 1.0 <= met["patterns_per_answer"] <= 3.0
    # several stranger-accept targets in one pass: the 2 % entry equals the scalar call,
    # looser targets never score lower on the tuning set
    multi = DC.tune_thresholds(st, p.abs_threshold(), tp, th, strangers, np.arange(0, 0.5, 0.05),
                               np.arange(0, 0.2, 0.02), np.arange(0, 0.2, 0.04), [0.02, 0.10])
    assert multi[0.02]["slack"] == res["slack"] and multi[0.02]["name_margin"] == res["name_margin"]
    assert multi[0.10]["tuning_metrics"]["name_and_hand"] >= met["name_and_hand"]
    assert multi[0.10]["tuning_metrics"]["stranger_accept"] <= 0.10
