"""The evaluation's vectorised decision replay agrees with auditor_ref.identifier outcome for outcome."""
import numpy as np

from evaluation.decision import NEED_MORE, FastModel, Thresholds, decide, make_identifier
from evaluation.protocol import pick


def _samples(rng, d=36):
    out, centers = [], {}
    for p in ("P01", "P02", "P03"):
        for h in ("right", "left"):
            centers[(p, h)] = rng.normal(0, 1.0, d)
            for _ in range(6):
                out.append({"prototype": (p, h, "none"), "features": (centers[(p, h)] + rng.normal(0, 0.6, d)).tolist()})
    return out, centers


def test_fast_decision_matches_reference():
    rng = np.random.default_rng(3)
    samples, centers = _samples(rng)
    keys = list(centers)
    for th in (Thresholds(), Thresholds(abs_factor=0.7, name_margin=2.0, hand_margin=1.0)):
        ident = make_identifier(samples, th)
        fm = FastModel(ident)
        for trial in range(60):
            c = centers[keys[trial % len(keys)]] if trial % 3 else rng.normal(0, 1.0, 36)
            pats = [c + rng.normal(0, 0.6 + 0.4 * (trial % 4), 36) for _ in range(3)]
            for m in (1, 2, 3):
                X = np.array(pats[:m])
                fast = decide(fm, fm.distances(X), fm.hand_log_odds(X), th)
                ref = ident.decide([x.tolist() for x in X])["outcome"]
                assert fast["kind"] == ref["type"]
                if fast["kind"] != NEED_MORE:
                    assert fast["person"] == ref["personID"] and fast["hand"] == ref["hand"]
                    assert fast["reason"] == ref["reason"]


def test_pick_respects_target_and_falls_back():
    rows = [{"absFactor": a, "nameMargin": 0.0, "stranger_accept": sa, "top1_name_acc": t1, "patterns_per_answer": 2.0}
            for a, sa, t1 in ((1.0, 0.01, 0.5), (2.0, 0.03, 0.9), (0.5, 0.0, 0.2))]
    best, met = pick(rows, 0.02)
    assert met and best["absFactor"] == 1.0
    best, met = pick(rows, -1.0)
    assert not met and best["absFactor"] == 0.5
