"""Open-set decision rules shared by the v1 evaluation and the phone contract.

v1 (SPEC 14.3) mirrors the v0 rule (SPEC 9.4) with similarities:

* prototype = L2-normalized mean embedding of a (person, hand, accessory);
* score = cosine similarity, fused over patterns by averaging;
* ``absThreshold`` = ``quantile_{1 - looQuantile}`` of the leave-one-out
  similarities minus a slack (tuned on held-out sessions);
* name margin = best score minus the best score of any other person;
  hand margin = best score minus the best score of the same person's other
  hand; both are similarity differences;
* the hand head replaces the pooled Gaussian hand model: it agrees with the
  best prototype when its fused right-vs-left logit difference has the sign
  of that prototype's hand.

The vectorized ``SequentialStats`` / ``decide`` pair evaluates the sequential
rule for many checks and many threshold settings at once. It is generic in
"scores" (higher is better), so v0 uses it too with ``score = -D^2`` and
``hand score = (D^2_left - D^2_right) / 2``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HAND_ORDER = {"right": 0, "left": 1}
ACCESSORY_ORDER = {"none": 0, "watch": 1, "ring": 2, "bracelet": 3, "other": 4}

IDENTIFIED, HAND_UNCLEAR, UNKNOWN_NO_MATCH, UNKNOWN_AMBIGUOUS = 0, 1, 2, 3
OUTCOME_NAMES = ("identified", "identifiedHandUnclear", "unknown(noMatch)", "unknown(ambiguous)")


def prototype_sort_key(key):
    person, hand, accessory = key
    return (str(person), HAND_ORDER.get(hand, 9), ACCESSORY_ORDER.get(accessory, 9))


def _normalize(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-12)


@dataclass
class Prototypes:
    keys: list               # [(personID, hand, accessory)] in SPEC 9.1 order
    matrix: np.ndarray       # [K, D] L2-normalized
    person: np.ndarray       # [K] person index into ``persons``
    hand: np.ndarray         # [K] 0 right, 1 left
    persons: list
    loo: np.ndarray          # leave-one-out similarities of the enrollment samples

    @staticmethod
    def fit(emb: np.ndarray, keys: list) -> "Prototypes":
        emb = _normalize(np.asarray(emb, dtype=np.float64))
        uniq = sorted(set(map(tuple, keys)), key=prototype_sort_key)
        index = {k: i for i, k in enumerate(uniq)}
        lab = np.array([index[tuple(k)] for k in keys])
        sums = np.zeros((len(uniq), emb.shape[1]))
        np.add.at(sums, lab, emb)
        counts = np.bincount(lab, minlength=len(uniq))
        matrix = _normalize(sums)
        loo = []
        for i in range(emb.shape[0]):
            k = lab[i]
            if counts[k] >= 2:
                loo.append(float(_normalize(sums[k] - emb[i]) @ emb[i]))
        persons = sorted({k[0] for k in uniq})
        pidx = {p: i for i, p in enumerate(persons)}
        return Prototypes(uniq, matrix, np.array([pidx[k[0]] for k in uniq]),
                          np.array([HAND_ORDER.get(k[1], 0) for k in uniq]), persons, np.array(loo))

    def scores(self, emb: np.ndarray) -> np.ndarray:
        return _normalize(np.asarray(emb, dtype=np.float64)) @ self.matrix.T

    def abs_threshold(self, loo_quantile: float = 0.95, slack: float = 0.0) -> float:
        if self.loo.size == 0:
            return -1.0 - slack
        return float(np.quantile(self.loo, 1.0 - loo_quantile)) - slack


@dataclass
class SequentialStats:
    """Per check and per number of patterns m = 1..M: best prototype, margins
    and hand-head agreement, from per-pattern scores."""

    best_k: np.ndarray        # [n, M]
    best: np.ndarray          # [n, M] fused best score
    margin_name: np.ndarray   # [n, M] (+inf without competitor)
    margin_hand: np.ndarray   # [n, M]
    head_ok: np.ndarray       # [n, M] bool
    available: np.ndarray     # [n, M] bool: pattern m exists for this check
    pred_person: np.ndarray   # [n, M] person of the best prototype (caller's person ids)
    pred_hand: np.ndarray     # [n, M] hand of the best prototype (0 right, 1 left)

    @staticmethod
    def build(scores: np.ndarray, hand_score: np.ndarray | None, proto_person: np.ndarray,
              proto_hand: np.ndarray, n_patterns: np.ndarray | None = None,
              person_ids: np.ndarray | None = None) -> "SequentialStats":
        """``scores`` [n, M, K] per pattern; ``hand_score`` [n, M] (> 0 means right).

        ``person_ids`` maps ``proto_person`` to the caller's global person ids
        (so stats of several enrollment groups can be concatenated)."""
        n, m_max, k = scores.shape
        if n_patterns is None:
            n_patterns = np.full(n, m_max)
        avail = np.arange(m_max)[None, :] < n_patterns[:, None]
        csum = np.cumsum(np.where(avail[:, :, None], scores, 0.0), axis=1)
        cnt = np.maximum(np.cumsum(avail, axis=1), 1)[:, :, None]
        fused = csum / cnt                                            # [n, M, K]
        best_k = np.argmax(fused, axis=2)                             # ties -> lowest index
        best = np.take_along_axis(fused, best_k[:, :, None], axis=2)[:, :, 0]
        bp = proto_person[best_k]                                     # [n, M]
        bh = proto_hand[best_k]
        same_person = proto_person[None, None, :] == bp[:, :, None]
        other_hand = same_person & (proto_hand[None, None, :] != bh[:, :, None])
        neg = -np.inf
        other_p = np.where(~same_person, fused, neg).max(axis=2)
        other_h = np.where(other_hand, fused, neg).max(axis=2)
        margin_name = best - other_p
        margin_hand = best - other_h
        if hand_score is None:
            head_ok = np.ones((n, m_max), dtype=bool)
        else:
            hs = np.cumsum(np.where(avail, hand_score, 0.0), axis=1) / cnt[:, :, 0]
            head_ok = np.where(bh == 0, hs > 0, hs < 0)
        pid = proto_person if person_ids is None else np.asarray(person_ids)[proto_person]
        return SequentialStats(best_k, best, margin_name, margin_hand, head_ok, avail,
                               pid[best_k], bh)

    @staticmethod
    def concat(parts) -> "SequentialStats":
        parts = list(parts)
        return SequentialStats(*(np.concatenate([getattr(p, f) for p in parts])
                                 for f in SequentialStats.__dataclass_fields__))


def decide(stats: SequentialStats, abs_thr, name_margin: float, hand_margin: float,
           max_patterns: int = 3):
    """Sequential rule of SPEC 9.4 / 14.3 for every check.

    ``abs_thr`` is a scalar or a per-check array (score units: accept when
    ``best >= abs_thr``). Returns ``(outcome, pred_person, pred_hand,
    patterns_used)``.
    """
    n, m_max = stats.best.shape
    m_lim = min(max_patterns, m_max)
    abs_thr = np.broadcast_to(np.asarray(abs_thr, dtype=np.float64), (n,))
    abs_ok = stats.best >= abs_thr[:, None]
    name_ok = stats.margin_name >= name_margin
    hand_ok = (stats.margin_hand >= hand_margin) & stats.head_ok
    ident = abs_ok & name_ok & hand_ok & stats.available
    outcome = np.full(n, -1)
    used = np.zeros(n, dtype=int)
    pick = np.zeros(n, dtype=int)
    for m in range(m_lim):
        # final step for a check: pattern budget reached or no further pattern recorded
        if m + 1 >= m_lim:
            last = np.ones(n, dtype=bool)
        else:
            last = ~stats.available[:, m + 1]
        todo = outcome < 0
        hit = todo & ident[:, m]
        outcome[hit] = IDENTIFIED
        used[hit] = m + 1
        pick[hit] = m
        fin = todo & ~hit & last
        if np.any(fin):
            unclear = fin & abs_ok[:, m] & name_ok[:, m]
            outcome[unclear] = HAND_UNCLEAR
            outcome[fin & ~unclear & ~abs_ok[:, m]] = UNKNOWN_NO_MATCH
            outcome[fin & ~unclear & abs_ok[:, m]] = UNKNOWN_AMBIGUOUS
            used[fin] = m + 1
            pick[fin] = m
    rows = np.arange(n)
    return outcome, stats.pred_person[rows, pick], stats.pred_hand[rows, pick], used


def check_metrics(outcome, pp, ph, used, true_person, true_hand, is_stranger) -> dict:
    """Metrics of the proposal for one set of checks.

    ``pp`` / ``ph``: predicted person and hand; ``true_person`` uses the same
    person ids (any value for strangers). Top-1 counts Unknown and hand-unclear answers with the right
    name as correct names (a name was given); hand accuracy counts only full
    identifications with the right hand.
    """
    named = (outcome == IDENTIFIED) | (outcome == HAND_UNCLEAR)
    enrolled = ~is_stranger
    ne = max(int(enrolled.sum()), 1)
    ns = max(int(is_stranger.sum()), 1)
    name_ok = named & (pp == true_person) & enrolled
    hand_ok = (outcome == IDENTIFIED) & (ph == true_hand) & enrolled
    both = name_ok & hand_ok
    return {
        "checks_enrolled": int(enrolled.sum()),
        "checks_stranger": int(is_stranger.sum()),
        "top1": float(name_ok.sum() / ne),
        "hand_acc": float(hand_ok.sum() / ne),
        "name_and_hand": float(both.sum() / ne),
        "wrong_name": float((named & (pp != true_person) & enrolled).sum() / ne),
        "false_unknown": float((~named & enrolled).sum() / ne),
        "hand_unclear": float(((outcome == HAND_UNCLEAR) & enrolled).sum() / ne),
        "stranger_accept": float((named & is_stranger).sum() / ns) if is_stranger.any() else float("nan"),
        "patterns_per_answer": float(used.mean()) if used.size else float("nan"),
    }


def closed_set_top1(stats: SequentialStats, true_person, true_hand, m: int = 1):
    """Nearest-prototype accuracy with ``m`` patterns and no Unknown (diagnostic)."""
    pp = stats.pred_person[:, m - 1]
    ph = stats.pred_hand[:, m - 1]
    return float(np.mean(pp == true_person)), float(np.mean((pp == true_person) & (ph == true_hand)))


def tune_thresholds(stats: SequentialStats, abs_base, true_person, true_hand, is_stranger,
                    slacks, name_margins, hand_margins,
                    max_stranger_accept: float = 0.02, max_patterns: int = 3,
                    abs_mode: str = "subtract"):
    """Grid search on a tuning set: maximize name+hand accuracy subject to the
    stranger-accept target; ties -> higher top-1, lower stranger accept, fewer
    patterns. If no setting meets the
    target, pick the lowest stranger accept. A sequence of targets returns a
    dict target -> choice (one pass over the grid).

    ``abs_mode='subtract'``: threshold = abs_base - slack (v1, similarities).
    ``abs_mode='scale'``: threshold = abs_base * slack (v0: abs_base = -LOO
    quantile distance, slack = absFactor).
    """
    scalar = np.isscalar(max_stranger_accept)
    targets = [float(max_stranger_accept)] if scalar else [float(t) for t in max_stranger_accept]
    best = {t: None for t in targets}
    fallback = None
    for s in slacks:
        thr = abs_base - s if abs_mode == "subtract" else abs_base * s
        for nm in name_margins:
            for hm in hand_margins:
                out, pp, ph, used = decide(stats, thr, nm, hm, max_patterns)
                met = check_metrics(out, pp, ph, used, true_person, true_hand, is_stranger)
                key = (met["name_and_hand"], met["top1"], -met["stranger_accept"],
                       -met["patterns_per_answer"])
                row = (s, nm, hm, met)
                for t in targets:
                    if met["stranger_accept"] <= t and (best[t] is None or key > best[t][0]):
                        best[t] = (key, row)
                fkey = (-met["stranger_accept"], met["name_and_hand"])
                if fallback is None or fkey > fallback[0]:
                    fallback = (fkey, row)

    def pack(b):
        chosen = b[1] if b is not None else fallback[1]
        return {"slack": chosen[0], "name_margin": chosen[1], "hand_margin": chosen[2],
                "tuning_metrics": chosen[3], "met_target": b is not None}
    if scalar:
        return pack(best[targets[0]])
    return {t: pack(best[t]) for t in targets}


def decide_one(pattern_emb: np.ndarray, hand_logits: np.ndarray, protos: Prototypes,
               abs_threshold: float, name_margin: float, hand_margin: float, max_patterns: int = 3):
    """The phone's rule for one check with the patterns played so far (14.3).

    ``pattern_emb`` [m, D], ``hand_logits`` [m, 3]. Returns a dict with the
    outcome name, prototype key and margins.
    """
    s = protos.scores(pattern_emb)[None]                  # [1, m, K]
    hs = (hand_logits[:, 0] - hand_logits[:, 1])[None]    # right-vs-left
    st = SequentialStats.build(s, hs, protos.person, protos.hand)
    m = pattern_emb.shape[0]
    abs_ok = st.best[0, m - 1] >= abs_threshold
    name_ok = st.margin_name[0, m - 1] >= name_margin
    hand_ok = st.margin_hand[0, m - 1] >= hand_margin and bool(st.head_ok[0, m - 1])
    k = int(st.best_k[0, m - 1])
    if abs_ok and name_ok and hand_ok:
        outcome = "identified"
    elif m < max_patterns:
        outcome = "needMorePatterns"
    elif abs_ok and name_ok:
        outcome = "identifiedHandUnclear"
    else:
        outcome = "unknown(noMatch)" if not abs_ok else "unknown(ambiguous)"
    two_handed = int(np.argmax(hand_logits.mean(axis=0))) == 2
    return {"outcome": outcome, "prototype": protos.keys[k], "best": float(st.best[0, m - 1]),
            "marginName": float(st.margin_name[0, m - 1]), "marginHand": float(st.margin_hand[0, m - 1]),
            "twoHanded": two_handed}
