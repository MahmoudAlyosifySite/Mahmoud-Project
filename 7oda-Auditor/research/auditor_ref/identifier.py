"""Recognition (SPEC 9.1 and 9.3-9.6): prototypes, hand head, sequential
decision, adaptive template updates and the optional hand challenge.

Samples are dicts ``{"prototype": (personID, hand, accessory), "features":
[36], "source": "enrollment" | "accessory" | "adaptive"}`` with optional
``"createdAt"`` (ISO string) and ``"id"``.

``Identifier.decide(patterns)`` returns a dict with the fields of the Swift
``Decision`` (SPEC 10.1), using JSON-style keys::

    outcome {type, personID, hand, reason}, patternsUsed, bestPrototype,
    bestDistance, marginName, marginHand, handLogOdds, absThreshold,
    personScores [{personID, probability, bestDistance}] (sorted desc),
    prototypeDistances [{prototype, distance}] (prototype order)

plus ``distancesPerPattern`` and ``handLogOddsPerPattern``. Margins are
``math.inf`` when there is no competitor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ._numerics import TieAudit, mean, note
from .model import GaussianModel, ModelConfig
from .rng import Xoshiro256

HAND_ORDER = {"right": 0, "left": 1}
ACCESSORY_ORDER = {"none": 0, "watch": 1, "ring": 2, "bracelet": 3, "other": 4}
SOURCES = ("enrollment", "accessory", "adaptive")


@dataclass(frozen=True)
class IdentifierConfig:
    """Defaults of SPEC 9.4/9.5 (``IdentifierConfig`` of SPEC 10.1)."""

    name_margin: float = 6.0
    hand_margin: float = 3.0
    max_patterns: int = 3
    adaptive_max_per_prototype: int = 20
    adaptive_name_factor: float = 2.0
    adaptive_abs_factor: float = 0.75

    def to_json(self) -> dict:
        return {
            "nameMargin": self.name_margin,
            "handMargin": self.hand_margin,
            "maxPatterns": self.max_patterns,
            "adaptiveMaxPerPrototype": self.adaptive_max_per_prototype,
            "adaptiveNameFactor": self.adaptive_name_factor,
            "adaptiveAbsFactor": self.adaptive_abs_factor,
        }


DEFAULT_CONFIG = IdentifierConfig()


def prototype_key(p: Sequence[str]) -> tuple:
    """Sort key of SPEC 9.1: personID string, then right < left, then
    none < watch < ring < bracelet < other."""
    return (p[0], HAND_ORDER[p[1]], ACCESSORY_ORDER[p[2]])


def sort_prototypes(protos) -> list[tuple[str, str, str]]:
    return sorted({tuple(p) for p in protos}, key=prototype_key)


def _outcome(kind: str, person: str | None = None, hand: str | None = None, reason: str | None = None) -> dict:
    return {"type": kind, "personID": person, "hand": hand, "reason": reason}


class Identifier:
    """SPEC 9: prototypes, Gaussian model, hand head and decision."""

    def __init__(self, samples: Sequence[dict], model_config: ModelConfig | None = None,
                 config: IdentifierConfig | None = None) -> None:
        self.model_config = model_config or ModelConfig()
        self.config = config or DEFAULT_CONFIG
        self.samples = list(samples)
        self.prototypes = sort_prototypes(s["prototype"] for s in self.samples)
        self._index = {p: i for i, p in enumerate(self.prototypes)}
        self.model: GaussianModel | None = None
        self.hand_model: GaussianModel | None = None
        if self.samples:
            x = [list(map(float, s["features"])) for s in self.samples]
            labels = [self._index[tuple(s["prototype"])] for s in self.samples]
            self.model = GaussianModel.fit(x, labels, len(self.prototypes), self.model_config)
            hands = [s["prototype"][1] for s in self.samples]
            if hands.count("right") >= 2 and hands.count("left") >= 2:
                hand_labels = [HAND_ORDER[h] for h in hands]
                self.hand_model = GaussianModel.fit(x, hand_labels, 2, self.model_config)

    # --- scores ------------------------------------------------------------
    def distances(self, x) -> list[float]:
        return self.model.distances(x) if self.model is not None else []

    def hand_log_odds(self, x) -> float:
        """``h(x) = (D2_left(x) - D2_right(x)) / 2``; 0 without a hand head."""
        if self.hand_model is None:
            return 0.0
        d = self.hand_model.distances(x)
        return (d[1] - d[0]) / 2.0

    # --- SPEC 9.4 -------------------------------------------------------------
    def decide(self, patterns: Sequence[Sequence[float]], audit: TieAudit | None = None) -> dict:
        cfg = self.config
        m = len(patterns)
        base = {
            "patternsUsed": m,
            "bestPrototype": None,
            "bestDistance": math.inf,
            "marginName": math.inf,
            "marginHand": math.inf,
            "handLogOdds": 0.0,
            "absThreshold": self.model.abs_threshold if self.model is not None else 0.0,
            "personScores": [],
            "prototypeDistances": [],
            "distancesPerPattern": [],
            "handLogOddsPerPattern": [],
        }
        if not self.prototypes or self.model is None:
            return {"outcome": _outcome("unknown", reason="noPeople"), **base}
        if m == 0:
            return {"outcome": _outcome("needMorePatterns"), **base}

        per_pattern = [self.distances(x) for x in patterns]
        per_hand = [self.hand_log_odds(x) for x in patterns]
        k_count = len(self.prototypes)
        dbar = [mean([per_pattern[j][k] for j in range(m)]) for k in range(k_count)]
        hbar = mean(per_hand) if self.hand_model is not None else 0.0

        kstar = 0
        for k in range(1, k_count):
            if dbar[k] < dbar[kstar]:
                kstar = k
        if audit is not None and k_count >= 2:
            second = min(dbar[k] for k in range(k_count) if k != kstar)
            note(audit, "decision.argmin", dbar[kstar], second, 1.0)
        person, hand, _ = self.prototypes[kstar]
        best = dbar[kstar]
        other_person = min((dbar[k] for k in range(k_count) if self.prototypes[k][0] != person), default=math.inf)
        other_hand = min((dbar[k] for k in range(k_count)
                          if self.prototypes[k][0] == person and self.prototypes[k][1] != hand), default=math.inf)
        margin_name = other_person - best
        margin_hand = other_hand - best
        abs_thr = self.model.abs_threshold
        abs_ok = best <= abs_thr
        name_ok = margin_name >= cfg.name_margin
        if self.hand_model is None:
            head_ok = True
        else:
            head_ok = hbar > 0.0 if hand == "right" else hbar < 0.0
            note(audit, "decision.handHead", hbar, 0.0, 1.0)
        hand_ok = margin_hand >= cfg.hand_margin and head_ok
        note(audit, "decision.abs", best, abs_thr)
        note(audit, "decision.name", margin_name, cfg.name_margin)
        note(audit, "decision.hand", margin_hand, cfg.hand_margin)

        if abs_ok and name_ok and hand_ok:
            outcome = _outcome("identified", person, hand)
            note(audit, "adaptive.name", margin_name, cfg.adaptive_name_factor * cfg.name_margin)
            note(audit, "adaptive.abs", best, cfg.adaptive_abs_factor * abs_thr)
        elif m < cfg.max_patterns:
            outcome = _outcome("needMorePatterns")
        elif abs_ok and name_ok:
            outcome = _outcome("identifiedHandUnclear", person)
        elif not abs_ok:
            outcome = _outcome("unknown", reason="noMatch")
        else:
            outcome = _outcome("unknown", reason="ambiguous")

        return {
            "outcome": outcome,
            **base,
            "bestPrototype": self.prototypes[kstar],
            "bestDistance": best,
            "marginName": margin_name,
            "marginHand": margin_hand,
            "handLogOdds": hbar,
            "personScores": self._person_scores(dbar),
            "prototypeDistances": [{"prototype": p, "distance": dbar[k]} for k, p in enumerate(self.prototypes)],
            "distancesPerPattern": per_pattern,
            "handLogOddsPerPattern": per_hand,
        }

    def _person_scores(self, dbar: Sequence[float]) -> list[dict]:
        """``softmax_p(-min_{k in p} Dbar_k / 2)``, sorted by probability
        (descending), ties by personID order."""
        persons: list[str] = []
        for p in self.prototypes:
            if p[0] not in persons:
                persons.append(p[0])
        best = [min(dbar[k] for k, p in enumerate(self.prototypes) if p[0] == pid) for pid in persons]
        logits = [-b / 2.0 for b in best]
        top = max(logits)
        ex = [math.exp(v - top) for v in logits]
        total = 0.0
        for v in ex:
            total = total + v
        scores = [{"personID": pid, "probability": ex[i] / total, "bestDistance": best[i]} for i, pid in enumerate(persons)]
        order = sorted(range(len(scores)), key=lambda i: (-scores[i]["probability"], i))
        return [scores[i] for i in order]

    # --- SPEC 9.5 ---------------------------------------------------------------
    def adaptive_eligible(self, decision: dict) -> bool:
        cfg = self.config
        if decision["outcome"]["type"] != "identified" or self.model is None:
            return False
        return (decision["marginName"] >= cfg.adaptive_name_factor * cfg.name_margin
                and decision["bestDistance"] <= cfg.adaptive_abs_factor * decision["absThreshold"])

    def adaptive_samples(self, decision: dict, patterns: Sequence[Sequence[float]], now: str | None = None) -> list[dict]:
        """New ``adaptive`` samples for prototype k* (``[]`` if not eligible)."""
        if not self.adaptive_eligible(decision):
            return []
        proto = tuple(decision["bestPrototype"])
        out = []
        for x in patterns:
            s = {"prototype": proto, "features": [float(v) for v in x], "source": "adaptive"}
            if now is not None:
                s["createdAt"] = now
            out.append(s)
        return out

    @staticmethod
    def apply_adaptive_cap(samples: Sequence[dict], cap: int) -> list[dict]:
        """Keep at most ``cap`` adaptive samples per prototype, dropping the
        oldest (by ``createdAt``, then list position). Enrollment and accessory
        samples are never dropped. Order of the kept samples is preserved."""
        drop: set[int] = set()
        groups: dict[tuple, list[int]] = {}
        for i, s in enumerate(samples):
            if s.get("source") == "adaptive":
                groups.setdefault(tuple(s["prototype"]), []).append(i)
        for idx in groups.values():
            if len(idx) <= cap:
                continue
            ordered = sorted(idx, key=lambda i: (samples[i].get("createdAt") or "", i))
            drop.update(ordered[: len(idx) - cap])
        return [s for i, s in enumerate(samples) if i not in drop]

    def updated(self, decision: dict, patterns: Sequence[Sequence[float]], now: str | None = None) -> "Identifier":
        """Identifier retrained after adding the adaptive samples (SPEC 9.5)."""
        new = self.adaptive_samples(decision, patterns, now)
        if not new:
            return self
        samples = self.apply_adaptive_cap(self.samples + new, self.config.adaptive_max_per_prototype)
        return Identifier(samples, self.model_config, self.config)


# --- SPEC 9.6 -------------------------------------------------------------------
def request_hand(rng: Xoshiro256) -> str:
    """Random requested hand: ``uniform() < 0.5 -> right``."""
    return "right" if rng.uniform() < 0.5 else "left"


def hand_challenge_passed(decision: dict, requested: str) -> bool:
    out = decision["outcome"]
    return out["type"] == "identified" and out["hand"] == requested
