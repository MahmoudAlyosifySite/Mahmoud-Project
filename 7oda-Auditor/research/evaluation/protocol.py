"""Evaluation protocol: calibration -> enrollment -> Identifier -> sequential checks -> metrics.

Works on processed items (evaluation.processing.item_from) from simulated studies or app
exports. A check is an ordered stream of recordings from one grip; the app plays them until
the SPEC 9.4 decision is final (quality failures are retried, at most 3 per slot).
"""
from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from .decision import (HAND_UNCLEAR, IDENTIFIED, QUALITY_FAIL, UNKNOWN, FastModel, Thresholds, make_identifier,
                       run_check)
from .processing import FEATURE_NAMES, MASKS, relative

# The grid reaches below absFactor 0.5 and above nameMargin 30 because strict stranger targets push the
# optimum to the strict corner; a tuned value on the grid edge is flagged in the report.
ABS_GRID = (0.25, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0)
NAME_GRID = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0)
OPERATING_TARGETS = (0.01, 0.02, 0.05, 0.10, 0.20)
ACCEPTED = (IDENTIFIED, HAND_UNCLEAR)


@dataclass
class Check:
    stream: list
    person: str | None
    hand: str | None
    accessory: str
    session: int
    role: str
    victim: str | None = None


@dataclass
class Dataset:
    items: list
    calibration: dict | None
    F: np.ndarray                      # (n, d) features (calibration-relative for the adopted feature set)
    F_ng: np.ndarray | None            # accelerometer-only features (gyro zeroed), NaN if not computed
    passed: np.ndarray
    passed_ng: np.ndarray | None
    dur: np.ndarray
    enroll: dict = field(default_factory=dict)        # set name -> {(person, hand, round): [idx...]}
    acc_add: dict = field(default_factory=dict)       # (person, hand, accessory) -> [idx...]
    checks: dict = field(default_factory=dict)        # id -> Check
    enrolled: set = field(default_factory=set)
    per_round: int = 5
    names: tuple = tuple(FEATURE_NAMES)


def _arrays(items, calibration, key="f_abs"):
    """Feature matrices of `items`. `key` "f_abs" is the adopted feature set (SPEC 7), made
    calibration-relative; any other key (e.g. "f_abs_v1", a comparison set) is used as stored (absolute
    features: on one phone the calibration offset does not change identification, DECISIONS D5)."""
    if key == "f_abs":
        F = np.array([relative(it["f_abs"], calibration) for it in items])
    else:
        F = np.array([np.asarray(it[key], float) for it in items])
    d = F.shape[1]
    if key == "f_abs" and any("f_abs_nogyro" in it for it in items):
        F_ng = np.array([relative(it["f_abs_nogyro"], calibration) if "f_abs_nogyro" in it else np.full(d, np.nan)
                         for it in items])
        p_ng = np.array([it.get("passed_nogyro", False) for it in items])
    else:
        F_ng, p_ng = None, None
    return F, F_ng, np.array([it["passed"] for it in items]), p_ng, np.array([it["duration"] for it in items])


def dataset_from_study(items, calibration, per_round=5, key="f_abs", names=None) -> Dataset:
    F, F_ng, passed, p_ng, dur = _arrays(items, calibration, key)
    if names is None:
        names = FEATURE_NAMES if F.shape[1] == len(FEATURE_NAMES) else [f"f{j}" for j in range(F.shape[1])]
    ds = Dataset(items=items, calibration=calibration, F=F, F_ng=F_ng, passed=passed, passed_ng=p_ng, dur=dur,
                 per_round=per_round, names=tuple(names))
    streams = defaultdict(list)
    for i, it in enumerate(items):
        st = it.get("study") or {}
        role = st.get("role")
        if role in ("enroll", "enroll2"):
            ds.enroll.setdefault(role, defaultdict(list))[(it["person"], it["hand"], st["round"])].append((st["seq"], i))
        elif role == "accessoryAdd":
            ds.acc_add.setdefault((it["person"], it["hand"], it["accessory"]), []).append((st["seq"], i))
        elif role in ("test", "accessoryTest", "mimic", "replay"):
            streams[st["check"]].append((st["seq"], i))
    for name, d in ds.enroll.items():
        ds.enroll[name] = {k: [i for _, i in sorted(v)] for k, v in d.items()}
    ds.acc_add = {k: [i for _, i in sorted(v)] for k, v in ds.acc_add.items()}
    for c, lst in streams.items():
        idx = [i for _, i in sorted(lst)]
        it = items[idx[0]]
        st = it["study"]
        victim = None if st.get("victim") is None else f"P{st['victim'] + 1:02d}"
        ds.checks[c] = Check(stream=idx, person=it["person"], hand=it["hand"], accessory=it["accessory"],
                             session=int(str(it["session"]).rsplit("S", 1)[-1]), role=st["role"], victim=victim)
    ds.enrolled = {k[0] for k in ds.enroll.get("enroll", {})}
    return ds


# --------------------------------------------------------------------------- models

def enrollment_samples(ds: Dataset, sets=("enroll",), mask=None, feature="F", hands=("right", "left"),
                       accessory_add=False, per_round=None):
    F = getattr(ds, feature)
    ok = ds.passed_ng if feature == "F_ng" else ds.passed
    mask = list(range(F.shape[1])) if mask is None else list(mask)
    per_round = per_round or ds.per_round
    samples = []
    for name in sets:
        for (person, hand, rnd), idx in sorted(ds.enroll.get(name, {}).items(), key=lambda kv: str(kv[0])):
            if hand not in hands:
                continue
            good = [i for i in idx if ok[i]][:per_round]
            for i in good:
                samples.append({"prototype": (person, hand, "none"), "features": F[i, mask].tolist(),
                                "source": "enrollment", "index": i})
    if accessory_add:
        for (person, hand, acc), idx in sorted(ds.acc_add.items()):
            if hand not in hands:
                continue
            for i in [i for i in idx if ok[i]][:per_round]:
                samples.append({"prototype": (person, hand, acc), "features": F[i, mask].tolist(),
                                "source": "accessory", "index": i})
    return samples


class Scorer:
    """A fitted model plus cached per-item distances for one feature view."""

    def __init__(self, ds: Dataset, samples, mask=None, feature="F"):
        self.ds = ds
        self.mask = list(range(ds.F.shape[1])) if mask is None else list(mask)
        self.feature = feature
        self.ident = make_identifier(samples)
        self.fm = FastModel(self.ident)
        self.passed = ds.passed_ng if feature == "F_ng" else ds.passed
        n = len(ds.items)
        self.dist = np.full((n, self.fm.K), np.nan)
        self.hlo = np.zeros(n)
        self._have = np.zeros(n, bool)

    def ensure(self, idx):
        idx = np.asarray([i for i in idx if not self._have[i] and self.passed[i]], int)
        if idx.size:
            X = getattr(self.ds, self.feature)[idx][:, self.mask]
            self.dist[idx] = self.fm.distances(X)
            self.hlo[idx] = self.fm.hand_log_odds(X)
            self._have[idx] = True

    def run(self, check_ids, th: Thresholds):
        out = []
        for c in check_ids:
            ch = self.ds.checks[c]
            self.ensure(ch.stream)
            r = run_check(self.fm, ch.stream, self.passed, self.dist, self.hlo, self.ds.dur, th)
            r.update(check=c, truth=ch.person, truth_hand=ch.hand, role=ch.role, session=ch.session,
                     accessory=ch.accessory, victim=ch.victim)
            out.append(r)
        return out


# --------------------------------------------------------------------------- metrics

def summarize(results, enrolled: set) -> dict:
    E = [r for r in results if r["truth"] in enrolled and r["role"] != "replay"]
    S = [r for r in results if r["truth"] not in enrolled]

    def rate(rs, pred):
        return float(np.mean([pred(r) for r in rs])) if rs else float("nan")

    acc = lambda r: r["kind"] in ACCEPTED
    ident_correct = [r for r in E if r["kind"] == IDENTIFIED and r["person"] == r["truth"]]
    out = {
        "n_enrolled_checks": len(E), "n_stranger_checks": len(S),
        "top1_name_acc": rate(E, lambda r: acc(r) and r["person"] == r["truth"]),
        "name_and_hand_acc": rate(E, lambda r: r["kind"] == IDENTIFIED and r["person"] == r["truth"]
                                  and r["hand"] == r["truth_hand"]),
        "hand_acc_given_named": (float(np.mean([r["hand"] == r["truth_hand"] for r in ident_correct]))
                                 if ident_correct else float("nan")),
        "misidentified": rate(E, lambda r: acc(r) and r["person"] != r["truth"]),
        "false_unknown": rate(E, lambda r: r["kind"] == UNKNOWN),
        "hand_unclear": rate(E, lambda r: r["kind"] == HAND_UNCLEAR),
        "quality_fail": rate(E, lambda r: r["kind"] == QUALITY_FAIL),
        "stranger_accept": rate(S, acc),
        "patterns_per_answer": rate(E + S, lambda r: r["patterns"]),
        "recordings_per_answer": rate(E + S, lambda r: r["played"]),
        "seconds_per_answer": rate(E + S, lambda r: r["seconds"]),
        "seconds_p90": float(np.percentile([r["seconds"] for r in E + S], 90)) if E + S else float("nan"),
        "unknown_reasons": dict(Counter(r["reason"] for r in E if r["kind"] == UNKNOWN)),
        "patterns_hist": dict(sorted(Counter(int(r["patterns"]) for r in E + S).items())),
    }
    return out


def confusion(results, enrolled: set) -> dict:
    labels = sorted(enrolled)
    cols = labels + ["Unknown", "qualityFail"]
    rows = labels + ["stranger"]
    M = np.zeros((len(rows), len(cols)), int)
    for r in results:
        if r["role"] == "replay":
            continue
        ri = rows.index(r["truth"]) if r["truth"] in enrolled else len(rows) - 1
        if r["kind"] in ACCEPTED:
            ci = cols.index(r["person"])
        elif r["kind"] == QUALITY_FAIL:
            ci = len(cols) - 1
        else:
            ci = len(cols) - 2
        M[ri, ci] += 1
    return {"rows": rows, "cols": cols, "counts": M.tolist()}


def closed_set(scorer: Scorer, check_ids, max_patterns=3) -> dict:
    """Forced-choice identification without the Unknown option (enrolled people's checks, replays excluded):
    the nearest prototype on the first passing pattern, and on the mean distance of the first `max_patterns`
    passing patterns. Separates "the features do not identify" from "the thresholds reject"."""
    fm, ds = scorer.fm, scorer.ds
    P, H = np.array(fm.person_of, dtype=object), np.array(fm.hand_of, dtype=object)
    first, mean_m, both = [], [], []
    for c in check_ids:
        ch = ds.checks[c]
        if ch.person not in ds.enrolled or ch.role == "replay":
            continue
        scorer.ensure(ch.stream)
        good = [i for i in ch.stream if scorer.passed[i]][:max_patterns]
        if not good:
            continue
        k1 = int(np.argmin(scorer.dist[good[0]]))
        k = int(np.argmin(scorer.dist[good].mean(0)))
        first.append(P[k1] == ch.person)
        mean_m.append(P[k] == ch.person)
        both.append(P[k] == ch.person and H[k] == ch.hand)
    f = lambda v: float(np.mean(v)) if v else float("nan")
    return {"n": len(mean_m), "patterns": max_patterns, "top1_first_pattern": f(first), "top1_mean_patterns": f(mean_m),
            "name_and_hand_mean_patterns": f(both), "chance": 1.0 / max(1, len(ds.enrolled))}


# --------------------------------------------------------------------------- threshold tuning

def grid_rows(scorer: Scorer, check_ids, base: Thresholds, abs_grid=ABS_GRID, name_grid=NAME_GRID):
    rows = []
    for a, nm in itertools.product(abs_grid, name_grid):
        th = replace(base, abs_factor=a, name_margin=nm)
        s = summarize(scorer.run(check_ids, th), scorer.ds.enrolled)
        rows.append({"absFactor": a, "nameMargin": nm, "stranger_accept": s["stranger_accept"],
                     "false_unknown": s["false_unknown"], "top1_name_acc": s["top1_name_acc"],
                     "misidentified": s["misidentified"], "patterns_per_answer": s["patterns_per_answer"]})
    return rows


def pick(rows, target):
    """Among settings with stranger accept <= target, the one with the best top-1 name accuracy (ties: lower
    stranger accept, fewer patterns). If none qualifies, the setting with the lowest stranger accept."""
    ok = [r for r in rows if not math.isnan(r["stranger_accept"]) and r["stranger_accept"] <= target]
    if ok:
        return max(ok, key=lambda r: (r["top1_name_acc"], -r["stranger_accept"], -r["patterns_per_answer"])), True
    return min(rows, key=lambda r: (r["stranger_accept"], -r["top1_name_acc"])), False


def tune(scorer: Scorer, check_ids, base: Thresholds, target=0.02, abs_grid=ABS_GRID, name_grid=NAME_GRID):
    """Grid search of absFactor x nameMargin on held-out checks: among settings with stranger
    accept <= target, maximise top-1 name accuracy (ties: lower stranger accept, fewer patterns)."""
    rows = grid_rows(scorer, check_ids, base, abs_grid, name_grid)
    best, met = pick(rows, target)
    return replace(base, abs_factor=best["absFactor"], name_margin=best["nameMargin"]), rows, met


def on_grid_edge(th: Thresholds, abs_grid=ABS_GRID, name_grid=NAME_GRID) -> list:
    edge = []
    if th.abs_factor in (min(abs_grid), max(abs_grid)):
        edge.append("absFactor")
    if th.name_margin == max(name_grid):
        edge.append("nameMargin")
    return edge


def pareto(rows):
    """Accept vs false-unknown frontier of a tuning grid."""
    pts = sorted(rows, key=lambda r: (r["stranger_accept"], r["false_unknown"]))
    out, best_fu = [], math.inf
    for r in pts:
        if r["false_unknown"] < best_fu - 1e-12:
            out.append(r)
            best_fu = r["false_unknown"]
    return out


# --------------------------------------------------------------------------- full evaluation

def checks_where(ds: Dataset, role=None, sessions=None, hands=None):
    return [c for c, ch in sorted(ds.checks.items())
            if (role is None or ch.role in (role if isinstance(role, tuple) else (role,)))
            and (sessions is None or ch.session in sessions) and (hands is None or ch.hand in hands)]


def variant(ds, name, tune_ids, test_ids, base, sets=("enroll",), mask=None, feature="F", accessory_add=False,
            target=0.02, keep_grid=False):
    samples = enrollment_samples(ds, sets, mask, feature, accessory_add=accessory_add)
    sc = Scorer(ds, samples, mask, feature)
    th, grid, met = tune(sc, tune_ids, base, target)
    res_default = sc.run(test_ids, base)
    res_tuned = sc.run(test_ids, th)
    out = {"name": name, "n_samples": len(samples), "tuned": asdict(th), "target_met_on_tuning": met,
           "tuned_on_grid_edge": on_grid_edge(th),
           "default": summarize(res_default, ds.enrolled), "test": summarize(res_tuned, ds.enrolled),
           "closed_set": closed_set(sc, test_ids, th.max_patterns)}
    if keep_grid:
        out["grid"] = grid
        out["pareto"] = pareto(grid)
    return out, sc, th, res_tuned


def full_evaluation(ds: Dataset, tune_session: int, test_sessions, log=print, target=0.02, on_part=None) -> dict:
    """Main result, operating points, diagnostics and ablations. `on_part(report)` is called after each part
    so long runs can write partial results."""
    on_part = on_part or (lambda r: None)
    base = Thresholds()
    tune_ids = checks_where(ds, "test", [tune_session])
    test_ids = checks_where(ds, "test", list(test_sessions))
    report = {"tune_session": tune_session, "test_sessions": list(test_sessions), "stranger_target": target}

    main, sc, th, res = variant(ds, "v0 all features, 1 sitting, 3 patterns", tune_ids, test_ids, base,
                                keep_grid=True, target=target)
    report["main"] = main
    report["confusion"] = confusion(res, ds.enrolled)
    report["per_session"] = {str(s): summarize(sc.run(checks_where(ds, "test", [s]), th), ds.enrolled)
                             for s in sorted({tune_session, *test_sessions})}
    log(f"main: top1 {main['test']['top1_name_acc']:.3f} stranger {main['test']['stranger_accept']:.3f} "
        f"(tuned abs {th.abs_factor}, name {th.name_margin}); closed-set top1 "
        f"{main['closed_set']['top1_mean_patterns']:.3f}")
    report["closed_set_per_session"] = {str(s): closed_set(sc, checks_where(ds, "test", [s]))
                                        for s in sorted({tune_session, *test_sessions})}
    ops = []
    for t in OPERATING_TARGETS:
        row, met = pick(main["grid"], t)
        th_t = replace(base, abs_factor=row["absFactor"], name_margin=row["nameMargin"])
        s_t = summarize(sc.run(test_ids, th_t), ds.enrolled)
        ops.append({"target": t, "met_on_tuning": met, "absFactor": row["absFactor"], "nameMargin": row["nameMargin"],
                    "tuning_stranger_accept": row["stranger_accept"], "tuning_top1": row["top1_name_acc"],
                    "test": {k: s_t[k] for k in ("top1_name_acc", "stranger_accept", "false_unknown", "misidentified",
                                                 "patterns_per_answer")}})
    report["operating_points"] = ops
    report["diagnostics"] = separability(ds, sc, test_sessions)
    on_part(report)

    abl = {}
    abl["patterns"] = [variant(ds, f"max {k} pattern(s)", tune_ids, test_ids, replace(base, max_patterns=k),
                               target=target)[0] for k in (1, 2, 3)]
    abl["features"] = [variant(ds, n, tune_ids, test_ids, base, mask=m, target=target)[0]
                       for n, m in MASKS.items() if n != "all"]
    if ds.F_ng is not None:
        abl["features"].append(variant(ds, "accelerometer only (gyro stream zeroed)", tune_ids, test_ids, base,
                                       feature="F_ng", target=target)[0])
    if "enroll2" in ds.enroll:
        abl["sittings"] = [main | {"name": "1 sitting (15/hand)"},
                           variant(ds, "2 sittings (15 + 5/hand)", tune_ids, test_ids, base,
                                   sets=("enroll", "enroll2"), target=target)[0]]
    report["ablations"] = abl
    on_part(report)
    log("ablations: patterns, features, sittings done")

    acc_ids = checks_where(ds, "accessoryTest")
    if acc_ids and ds.acc_add:
        before = summarize(sc.run(acc_ids, th), ds.enrolled)
        sc_acc = Scorer(ds, enrollment_samples(ds, accessory_add=True))
        after = summarize(sc_acc.run(acc_ids, th), ds.enrolled)
        bare_after = summarize(sc_acc.run(test_ids, th), ds.enrolled)
        abl["accessory"] = {"before_quick_add": before, "after_quick_add": after,
                            "bare_checks_after_quick_add": bare_after,
                            "by_accessory": {a: {"before": summarize(sc.run([c for c in acc_ids if ds.checks[c].accessory == a], th), ds.enrolled)["top1_name_acc"],
                                                 "after": summarize(sc_acc.run([c for c in acc_ids if ds.checks[c].accessory == a], th), ds.enrolled)["top1_name_acc"]}
                                             for a in sorted({ds.checks[c].accessory for c in acc_ids})}}

    cross = {}
    for enrolled_hand, other in (("right", "left"), ("left", "right")):
        s1 = Scorer(ds, enrollment_samples(ds, hands=(enrolled_hand,)))
        ids = checks_where(ds, "test", list(test_sessions), hands=(other,))
        ids = [c for c in ids if ds.checks[c].person in ds.enrolled]
        rs = s1.run(ids, th)
        cross[f"enroll {enrolled_hand}, test {other}"] = {
            "n": len(rs), "unknown": float(np.mean([r["kind"] == UNKNOWN for r in rs])),
            "named_self": float(np.mean([r["kind"] in ACCEPTED and r["person"] == r["truth"] for r in rs])),
            "named_someone_else": float(np.mean([r["kind"] in ACCEPTED and r["person"] != r["truth"] for r in rs])),
            "quality_fail": float(np.mean([r["kind"] == QUALITY_FAIL for r in rs]))}
        # same-hand control with the same one-hand model
        ctrl = [c for c in checks_where(ds, "test", list(test_sessions), hands=(enrolled_hand,))
                if ds.checks[c].person in ds.enrolled]
        cross[f"enroll {enrolled_hand}, test {other}"]["same_hand_control_top1"] = summarize(
            s1.run(ctrl, th), ds.enrolled)["top1_name_acc"]
    abl["cross_hand"] = cross

    rep_ids = checks_where(ds, "replay")
    if rep_ids:
        idx = [i for c in rep_ids for i in ds.checks[c].stream]
        issues = Counter(x for i in idx for x in ds.items[i]["issues"])
        rs = sc.run(rep_ids, th)
        abl["replay"] = {"n_patterns": len(idx), "pattern_rejected_by_quality": float(np.mean(~ds.passed[idx])),
                         "rejection_issues": dict(issues), "n_checks": len(rs),
                         "checks_quality_fail": float(np.mean([r["kind"] == QUALITY_FAIL for r in rs])),
                         "checks_accepted_as_victim": float(np.mean([r["kind"] in ACCEPTED and r["person"] == r["truth"]
                                                                     for r in rs])),
                         "checks_accepted_as_anyone": float(np.mean([r["kind"] in ACCEPTED for r in rs])),
                         "ncc_median": float(np.median([ds.items[i]["metrics"]["ncc"] for i in idx]))}
    mim_ids = checks_where(ds, "mimic")
    if mim_ids:
        rs = sc.run(mim_ids, th)
        base_ids = [c for c in checks_where(ds, "test", [min(test_sessions, key=lambda s: abs(s - 3))])
                    if ds.checks[c].person not in ds.enrolled]
        abl["mimicry"] = {"n_checks": len(rs),
                          "accepted_as_victim": float(np.mean([r["kind"] in ACCEPTED and r["person"] == r["victim"]
                                                               for r in rs])),
                          "accepted_as_anyone": float(np.mean([r["kind"] in ACCEPTED for r in rs])),
                          "baseline_stranger_accept_same_session": summarize(sc.run(base_ids, th), ds.enrolled)[
                              "stranger_accept"],
                          "baseline_named_victim_rate": 1.0 / max(1, len(ds.enrolled))}
    report["ablations"] = abl
    report["quality"] = quality_summary(ds)
    on_part(report)
    return report


def quality_summary(ds: Dataset) -> dict:
    out = {}
    for role in ("calibration", "enroll", "test", "accessoryTest", "mimic"):
        idx = [i for i, it in enumerate(ds.items) if (it.get("study") or {}).get("role") == role]
        if not idx:
            continue
        out[role] = {"n": len(idx), "pass_rate": float(np.mean(ds.passed[idx])),
                     "issues": dict(Counter(x for i in idx for x in ds.items[i]["issues"])),
                     "snr_median": float(np.median([ds.items[i]["metrics"]["snr"] for i in idx])),
                     "ncc_median": float(np.median([ds.items[i]["metrics"]["ncc"] for i in idx]))}
    return out


def separability(ds: Dataset, sc: Scorer, test_sessions, targets=(0.02, 0.10)) -> dict:
    """Diagnostics of where identity information is (test checks of enrolled people, first 3 passing patterns):
    per-feature variance split into between person-hand, session (day), grip (check) and trial parts with Fisher
    ratios; discriminant dimensions (eigenvalues of W^-1 B, W = cross-session within-class scatter); a same-day
    forced-choice reference; and an optimistic bound for the threshold rule (abs + name thresholds chosen on
    the test checks themselves, single shot on the 3-pattern mean)."""
    rows = []
    for c in checks_where(ds, "test", list(test_sessions)):
        ch = ds.checks[c]
        if ch.person in ds.enrolled:
            rows += [(i, f"{ch.person}/{ch.hand}", ch.session, c) for i in [i for i in ch.stream if ds.passed[i]][:3]]
    if len(rows) < 20:
        return {}
    X = ds.F[[r[0] for r in rows]]
    cls = np.array([r[1] for r in rows])

    def gmean(keys):
        u, inv = np.unique(np.asarray(keys), return_inverse=True)
        M = np.zeros((len(u), X.shape[1]))
        np.add.at(M, inv, X)
        return (M / np.bincount(inv)[:, None])[inv]

    mP, mS, mC = gmean(cls), gmean([f"{r[1]}/{r[2]}" for r in rows]), gmean([r[3] for r in rows])
    vP, vS = mP.var(0), ((mS - mP) ** 2).mean(0)
    vG, vT = ((mC - mS) ** 2).mean(0), ((X - mC) ** 2).mean(0)
    feats = [{"index": j, "name": ds.names[j], "between": float(vP[j]), "session": float(vS[j]),
              "grip": float(vG[j]), "trial": float(vT[j]),
              "fisher_same_session": float(vP[j] / (vG[j] + vT[j] + 1e-30)),
              "fisher_cross_session": float(vP[j] / (vS[j] + vG[j] + vT[j] + 1e-30))} for j in range(X.shape[1])]
    sd = X.std(0)
    Z = (X - X.mean(0)) / np.where(sd > 0, sd, 1.0)
    u, inv = np.unique(cls, return_inverse=True)
    M = np.array([Z[inv == k].mean(0) for k in range(len(u))])
    # whiten with W restricted to its well-conditioned subspace (the energy shares of each event sum to 1,
    # so W is singular along that direction and the plain W^-1 B ratio there is 0/0)
    wv, wV = np.linalg.eigh(np.cov((Z - M[inv]).T))
    keep = wv > 1e-6 * wv.max()
    T = wV[:, keep] / np.sqrt(wv[keep])
    ev = np.sort(np.linalg.eigvalsh(T.T @ np.cov(M.T) @ T))[::-1]
    out = {"features": feats, "discriminant_eigenvalues": [float(v) for v in ev[:10]],
           "n_dims_ratio_gt_1": int((ev > 1).sum())}
    # same-day reference: enrollment rounds 0-1 (sitting, standing) -> round 2 (fresh grip), one pattern
    en = ds.enroll.get("enroll", {})
    tr = [(i, p, h) for (p, h, r), idx in en.items() if r in (0, 1) for i in idx if ds.passed[i]]
    te = [(i, p, h) for (p, h, r), idx in en.items() if r == 2 for i in idx if ds.passed[i]]
    if tr and te:
        fm2 = FastModel(make_identifier([{"prototype": (p, h, "none"), "features": ds.F[i].tolist()} for i, p, h in tr]))
        k = fm2.distances(ds.F[[t[0] for t in te]]).argmin(1)
        out["same_day_top1_first_pattern"] = float(np.mean(np.array(fm2.person_of)[k] == np.array([t[1] for t in te])))
    # optimistic bound of the abs + name-margin rule on the test checks
    P = np.array(sc.fm.person_of, dtype=object)
    g, s_ = [], []
    for c in checks_where(ds, "test", list(test_sessions)):
        ch = ds.checks[c]
        sc.ensure(ch.stream)
        good = [i for i in ch.stream if ds.passed[i]][:3]
        if not good:
            continue
        db = sc.dist[good].mean(0)
        k = int(np.argmin(db))
        rec = (db[k], db[P != P[k]].min() - db[k], P[k] == ch.person)
        (g if ch.person in ds.enrolled else s_).append(rec)
    G, S = np.array(g, float), np.array(s_, float)
    if len(G) and len(S):
        allv = np.r_[G, S]
        qa = np.unique(np.quantile(allv[:, 0], np.linspace(0.01, 1.0, 80)))
        qm = np.unique(np.quantile(allv[:, 1], np.linspace(0.0, 0.99, 80)))
        bound = {}
        for t in targets:
            best = 0.0
            for a in qa:
                sa = S[:, 0] <= a
                ga = (G[:, 0] <= a) & (G[:, 2] > 0)
                for m in qm:
                    if np.mean(sa & (S[:, 1] >= m)) <= t:
                        best = max(best, float(np.mean(ga & (G[:, 1] >= m))))
            bound[f"{t:g}"] = best
        out["oracle_top1_at_stranger_accept"] = bound
        out["genuine_best_d2_median"] = float(np.median(G[:, 0]))
        out["stranger_best_d2_median"] = float(np.median(S[:, 0]))
        out["genuine_margin_median"] = float(np.median(G[:, 1]))
        out["stranger_margin_median"] = float(np.median(S[:, 1]))
    return out
