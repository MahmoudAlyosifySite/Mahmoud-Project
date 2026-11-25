"""Fast, exact emulation of the SPEC 9.4 sequential decision on cached distances.

`auditor_ref.identifier.Identifier.decide` recomputes every distance on every call; the
evaluation sweeps thresholds over thousands of checks, so distances are computed once per
model (vectorised, same formulas) and the decision rule is replayed on the cached values.
`tests/test_eval_decision.py` checks outcome-for-outcome agreement with the reference.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from auditor_ref.identifier import Identifier, IdentifierConfig
from auditor_ref.model import ModelConfig

IDENTIFIED = "identified"
HAND_UNCLEAR = "identifiedHandUnclear"
NEED_MORE = "needMorePatterns"
UNKNOWN = "unknown"
QUALITY_FAIL = "qualityFail"          # every attempt of a slot failed the quality gate before any answer
OVERHEAD_S = 0.35                     # per played pattern (post-roll + processing), proposal / SPEC D1


@dataclass(frozen=True)
class Thresholds:
    abs_factor: float = 1.5
    name_margin: float = 6.0
    hand_margin: float = 3.0
    max_patterns: int = 3
    max_retries: int = 3              # per pattern slot, as in the app (SPEC 12)


class FastModel:
    """Vectorised distances of an Identifier's class model and hand head."""

    def __init__(self, ident: Identifier):
        self.ident = ident
        self.prototypes = list(ident.prototypes)
        m = ident.model
        self.K = len(self.prototypes)
        self.base_threshold = m.abs_threshold / ident.model_config.abs_factor
        self._mean, self._scale, self._cm = np.asarray(m.mean), np.asarray(m.scale), np.asarray(m.class_means)
        self._Linv = np.linalg.inv(np.asarray(m.chol))
        hm = ident.hand_model
        self.has_hand = hm is not None
        if self.has_hand:
            self._hmean, self._hscale, self._hcm = np.asarray(hm.mean), np.asarray(hm.scale), np.asarray(hm.class_means)
            self._hLinv = np.linalg.inv(np.asarray(hm.chol))
        persons = [p[0] for p in self.prototypes]
        hands = [p[1] for p in self.prototypes]
        self.person_of = persons
        self.hand_of = hands
        P = np.array(persons, dtype=object)
        H = np.array(hands, dtype=object)
        self.other_person = P[None, :] != P[:, None]                       # [k*, k]
        self.other_hand = (P[None, :] == P[:, None]) & (H[None, :] != H[:, None])

    @staticmethod
    def _d2(X, mean, scale, cm, Linv):
        z = (np.atleast_2d(X) - mean) / scale
        diff = z[:, None, :] - cm[None, :, :]
        y = diff @ Linv.T
        return np.einsum("nkd,nkd->nk", y, y)

    def distances(self, X) -> np.ndarray:
        return self._d2(X, self._mean, self._scale, self._cm, self._Linv)

    def hand_log_odds(self, X) -> np.ndarray:
        X = np.atleast_2d(X)
        if not self.has_hand:
            return np.zeros(len(X))
        d = self._d2(X, self._hmean, self._hscale, self._hcm, self._hLinv)
        return (d[:, 1] - d[:, 0]) / 2.0


def decide(fm: FastModel, dist_rows: np.ndarray, h_rows: np.ndarray, th: Thresholds, final: bool = False) -> dict:
    """SPEC 9.4 on m >= 1 patterns given their distance rows (m, K) and hand log-odds (m,).
    `final=True` applies the last-pattern branch even if m < maxPatterns (slot abandoned)."""
    m = len(dist_rows)
    dbar = dist_rows[0].copy()
    for j in range(1, m):
        dbar = dbar + dist_rows[j]
    dbar = dbar / m
    hbar = float(sum(float(v) for v in h_rows) / m) if fm.has_hand else 0.0
    k = int(np.argmin(dbar))
    best = float(dbar[k])
    op = dbar[fm.other_person[k]]
    oh = dbar[fm.other_hand[k]]
    margin_name = float(op.min()) - best if op.size else math.inf
    margin_hand = float(oh.min()) - best if oh.size else math.inf
    abs_ok = best <= th.abs_factor * fm.base_threshold
    name_ok = margin_name >= th.name_margin
    hand = fm.hand_of[k]
    head_ok = (not fm.has_hand) or (hbar > 0.0 if hand == "right" else hbar < 0.0)
    hand_ok = margin_hand >= th.hand_margin and head_ok
    person = fm.person_of[k]
    if abs_ok and name_ok and hand_ok:
        kind, reason = IDENTIFIED, None
    elif m < th.max_patterns and not final:
        kind, reason = NEED_MORE, None
    elif abs_ok and name_ok:
        kind, reason = HAND_UNCLEAR, None
    elif not abs_ok:
        kind, reason = UNKNOWN, "noMatch"
    else:
        kind, reason = UNKNOWN, "ambiguous"
    return {"kind": kind, "person": person if kind in (IDENTIFIED, HAND_UNCLEAR) else None,
            "hand": hand if kind == IDENTIFIED else None, "reason": reason, "bestDistance": best,
            "marginName": margin_name, "marginHand": margin_hand, "handLogOdds": hbar, "bestPerson": person,
            "bestHand": hand}


def run_check(fm: FastModel, stream, passed, dist, hlo, durations, th: Thresholds) -> dict:
    """Play the recordings of one check in order, like the app: a quality failure is retried
    (at most `max_retries` per slot), each passing pattern is added and the decision taken;
    stop at a final outcome. dist/hlo are indexed by item index (rows for passing items)."""
    pats, played, secs, fails = [], 0, 0.0, 0
    dec = None
    for idx in stream:
        played += 1
        secs += durations[idx] + OVERHEAD_S
        if not passed[idx]:
            fails += 1
            if fails > th.max_retries:
                break
            continue
        fails = 0
        pats.append(idx)
        dec = decide(fm, dist[pats], hlo[pats], th)
        if dec["kind"] != NEED_MORE:
            break
    if dec is None:
        dec = {"kind": QUALITY_FAIL, "person": None, "hand": None, "reason": "quality"}
    elif dec["kind"] == NEED_MORE:
        dec = decide(fm, dist[pats], hlo[pats], th, final=True)
    dec.update(patterns=len(pats), played=played, seconds=secs)
    return dec


def make_identifier(samples, th: Thresholds | None = None) -> Identifier:
    th = th or Thresholds()
    return Identifier(samples, ModelConfig(abs_factor=th.abs_factor),
                      IdentifierConfig(name_margin=th.name_margin, hand_margin=th.hand_margin,
                                       max_patterns=th.max_patterns))
