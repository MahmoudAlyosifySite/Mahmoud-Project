"""Analyse a real app export (JSON Lines of SPEC section 4 recordings) with the evaluation protocol.

    cd research && python -m evaluation.analyze_export recordings.jsonl [--out DIR] [--target 0.02]

Steps (auditor_ref only, so the numbers are what the app's v0 pipeline would give):

1. `calibration` recordings -> one on-table calibration (SPEC 8; none if fewer than 12 pass).
2. Every other labelled recording -> quality report (`handheld`) and calibration-relative features.
   `signalCheck` recordings and unlabelled recordings are not used.
3. `enrollment` recordings (passing quality) -> prototypes (person, hand, none); `accessory`
   recordings -> (person, hand, accessory). The model is the SPEC 9 Identifier.
4. `test` and labelled `check` recordings -> checks. Recordings of one (session, person, hand,
   accessory) are ordered by `hapticStart`; a gap of more than `--check-gap` seconds starts a new
   run, and each run is consumed like the app does: patterns are added until the SPEC 9.4
   decision is final (quality failures retried, at most 3 per slot), then the next check starts.
5. Metrics. Sessions are numbered per person: 0 for sessions in which the person enrolled, then
   1, 2, ... in time order. Enrolled people's checks in sessions >= 1 are cross-session; checks of
   people without enrollment are strangers. Reported at the SPEC defaults and, when the export has
   test sessions 1 and >= 2 and strangers, with absFactor/nameMargin tuned on test session 1 for
   the stranger-accept target and reported on the later sessions. Also: forced choice (no Unknown), per-session results, confusion matrix.

Writes `<stem>_analysis.md` and `<stem>_analysis.json` (next to the input unless `--out` is given).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from auditor_ref import features

from .decision import Thresholds, run_check
from .processing import build_calibration, item_from, relative
from .protocol import (ABS_GRID, NAME_GRID, Check, Dataset, Scorer, closed_set, confusion, enrollment_samples, pick,
                       summarize)
from .run_sim_study import clean

TEST_PURPOSES = ("test", "check")
POSTURE_ROUND = {"sitting": 0, "standing": 1, "freshGrip": 2}
ALL_PER_ROUND = 10 ** 9          # use every passing enrollment pattern


def read_jsonl(path) -> tuple[list[dict], int]:
    """Recordings of a JSON Lines export; returns (recordings, number of unreadable lines)."""
    recs, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(rec, dict) and isinstance(rec.get("challenge"), dict) and "accel" in rec:
                recs.append(rec)
            else:
                bad += 1
    return recs, bad


def _label(rec) -> dict | None:
    lab = rec.get("label")
    if not isinstance(lab, dict) or not lab.get("personID") or lab.get("hand") not in ("right", "left"):
        return None
    return lab


def build_dataset(recs: list[dict], check_gap: float = 8.0, device: str = "export"):
    """Processed items and a protocol Dataset (checks are the gap-separated runs; see `run_streams`)."""
    cal_recs = [r for r in recs if r.get("purpose") == "calibration"]
    calibration, _ = build_calibration(cal_recs, device=device) if cal_recs else (None, [])
    used = [r for r in recs if r.get("purpose") in ("enrollment", "accessory", *TEST_PURPOSES) and _label(r)]
    # createdAt (UTC, whole seconds) orders sessions; hapticStart (uptime, restarts with the phone) orders within
    used.sort(key=lambda r: (str(r.get("createdAt") or ""), float(r.get("hapticStart", 0.0))))
    # Session number per person: 0 = a session in which the person enrolled (or added an accessory), then
    # 1, 2, ... for the person's other sessions in time order. This works whether the app's sessionID is shared
    # by everyone in a session or specific to one person.
    enroll_sessions, order = defaultdict(set), defaultdict(list)
    for r in used:
        lab = _label(r)
        sid = str(lab.get("sessionID"))
        if r.get("purpose") in ("enrollment", "accessory"):
            enroll_sessions[lab["personID"]].add(sid)
        if sid not in order[lab["personID"]]:
            order[lab["personID"]].append(sid)
    rank = {}
    for person, sids in order.items():
        k = 0
        for sid in sids:
            if sid in enroll_sessions[person]:
                rank[(person, sid)] = 0
            else:
                k += 1
                rank[(person, sid)] = k
    items = []
    for r in used:
        it, _, _ = item_from(r, calibration)
        it["session_number"] = rank[(it["person"], str(it["session"]))]
        items.append(it)
    F = np.array([relative(it["f_abs"], calibration) for it in items]) if items else np.zeros((0, len(features.FEATURE_NAMES)))
    ds = Dataset(items=items, calibration=calibration, F=F, F_ng=None,
                 passed=np.array([it["passed"] for it in items], bool), passed_ng=None,
                 dur=np.array([it["duration"] for it in items], float), per_round=ALL_PER_ROUND,
                 names=tuple(features.FEATURE_NAMES))
    enroll, acc, runs = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, it in enumerate(items):
        if it["purpose"] == "enrollment":
            enroll[(it["person"], it["hand"], POSTURE_ROUND.get(it["posture"], 0))].append(i)
        elif it["purpose"] == "accessory":
            acc[(it["person"], it["hand"], it["accessory"] or "other")].append(i)
        else:
            runs[(it["session_number"], str(it["session"]), it["person"], it["hand"], it["accessory"] or "none")].append(i)
    ds.enroll = {"enroll": dict(enroll)} if enroll else {}
    ds.acc_add = dict(acc)
    ds.enrolled = {k[0] for k in enroll}
    cid = 0
    for key in sorted(runs, key=lambda k: (k[0], k[1], str(k[2]), k[3], k[4])):
        idx = runs[key]
        seg = [idx[0]]
        for prev, cur in zip(idx, idx[1:]):
            if items[cur]["hapticStart"] - items[prev]["hapticStart"] > check_gap:
                ds.checks[cid] = _check(seg, key)
                cid += 1
                seg = []
            seg.append(cur)
        ds.checks[cid] = _check(seg, key)
        cid += 1
    return ds, calibration, {"session_numbers": {f"{p} {sid}": v for (p, sid), v in sorted(rank.items())}}


def _check(seg, key) -> Check:
    number, _, person, hand, accessory = key
    role = "test" if number > 0 else "sameSession"
    return Check(stream=list(seg), person=person, hand=hand, accessory=accessory, session=number, role=role)


def run_streams(sc: Scorer, run_ids, th: Thresholds) -> list[dict]:
    """Consume each run like the app: one check per final decision, the next check starts after it."""
    ds = sc.ds
    out = []
    for c in run_ids:
        ch = ds.checks[c]
        sc.ensure(ch.stream)
        rest = list(ch.stream)
        while rest:
            r = run_check(sc.fm, rest, sc.passed, sc.dist, sc.hlo, ds.dur, th)
            used = rest[:r["played"]]
            rest = rest[r["played"]:]
            r.update(run=c, stream=used, truth=ch.person, truth_hand=ch.hand, role=ch.role, session=ch.session,
                     accessory=ch.accessory, victim=None)
            out.append(r)
    return out


def checks_dataset(ds: Dataset, results) -> Dataset:
    """A copy of `ds` whose checks are the consumed checks (for forced choice on the same checks)."""
    d2 = replace(ds, checks={})
    for k, r in enumerate(results):
        d2.checks[k] = Check(stream=r["stream"], person=r["truth"], hand=r["truth_hand"], accessory=r["accessory"],
                             session=r["session"], role="test")
    return d2


def tune(sc: Scorer, run_ids, base: Thresholds, target: float):
    rows = []
    for a in ABS_GRID:
        for nm in NAME_GRID:
            th = replace(base, abs_factor=a, name_margin=nm)
            s = summarize(run_streams(sc, run_ids, th), sc.ds.enrolled)
            rows.append({"absFactor": a, "nameMargin": nm, "stranger_accept": s["stranger_accept"],
                         "false_unknown": s["false_unknown"], "top1_name_acc": s["top1_name_acc"],
                         "patterns_per_answer": s["patterns_per_answer"]})
    best, met = pick(rows, target)
    return replace(base, abs_factor=best["absFactor"], name_margin=best["nameMargin"]), met


def analyze(recs: list[dict], th: Thresholds | None = None, target: float = 0.02, check_gap: float = 8.0) -> dict:
    th = th or Thresholds()
    ds, calibration, meta = build_dataset(recs, check_gap)
    purposes = Counter(r.get("purpose") for r in recs)
    rep = {"n_recordings": len(recs), "purposes": dict(purposes), "calibration": None if calibration is None else
           {k: calibration[k] for k in ("patternsUsed", "quietGyroRMS", "snr", "ncc", "sampleRate")},
           "feature_version": features.FEATURE_VERSION, "thresholds": asdict(th), "stranger_target": target,
           "n_enrolled_people": len(ds.enrolled), "enrolled": sorted(ds.enrolled), **meta}
    roles = Counter((ds.items[i].get("purpose")) for i in range(len(ds.items)))
    rep["quality"] = {p: {"n": n, "pass_rate": float(np.mean([it["passed"] for it in ds.items if it["purpose"] == p])),
                          "issues": dict(Counter(x for it in ds.items if it["purpose"] == p for x in it["issues"]))}
                      for p, n in sorted(roles.items())}
    if not ds.enrolled:
        rep["error"] = "no enrollment recordings with a label: nothing to identify against"
        return rep
    sc = Scorer(ds, enrollment_samples(ds, accessory_add=bool(ds.acc_add)))
    cross_ids = [c for c, ch in sorted(ds.checks.items()) if ch.role == "test"]
    same_ids = [c for c, ch in sorted(ds.checks.items()) if ch.role == "sameSession"]
    res = run_streams(sc, cross_ids, th)
    rep["cross_session"] = summarize(res, ds.enrolled)
    rep["confusion"] = confusion(res, ds.enrolled)
    d2 = checks_dataset(ds, res)
    sc2 = Scorer(d2, enrollment_samples(d2, accessory_add=bool(ds.acc_add)))
    rep["forced_choice"] = {"1_pattern": closed_set(sc2, sorted(d2.checks), 1)["top1_first_pattern"],
                            "3_patterns": closed_set(sc2, sorted(d2.checks), 3)["top1_mean_patterns"],
                            "name_and_hand_3_patterns": closed_set(sc2, sorted(d2.checks), 3)["name_and_hand_mean_patterns"]}
    if same_ids:
        rep["same_session"] = summarize(run_streams(sc, same_ids, th), ds.enrolled)
    sessions = sorted({ds.checks[c].session for c in cross_ids})
    rep["per_session"] = {str(s): summarize(run_streams(sc, [c for c in cross_ids if ds.checks[c].session == s], th),
                                            ds.enrolled) for s in sessions}
    has_strangers = any(ds.checks[c].person not in ds.enrolled for c in cross_ids)
    if len(sessions) >= 2 and has_strangers:
        tune_ids = [c for c in cross_ids if ds.checks[c].session == sessions[0]]
        test_ids = [c for c in cross_ids if ds.checks[c].session != sessions[0]]
        th_t, met = tune(sc, tune_ids, th, target)
        rep["tuned"] = {"tune_session": sessions[0], "test_sessions": sessions[1:], "absFactor": th_t.abs_factor,
                        "nameMargin": th_t.name_margin, "target_met_on_tuning": met,
                        "default": summarize(run_streams(sc, test_ids, th), ds.enrolled),
                        "test": summarize(run_streams(sc, test_ids, th_t), ds.enrolled)}
    return rep


def _p(v):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:.1f} %"


def render(rep: dict, title: str) -> str:
    out = [f"# {title}", "", f"{rep['n_recordings']} recordings ({', '.join(f'{k}: {v}' for k, v in sorted(rep['purposes'].items(), key=lambda kv: str(kv[0])))}); "
           f"feature version {rep['feature_version']}; calibration: "
           + ("none" if rep["calibration"] is None else f"{rep['calibration']['patternsUsed']} patterns") + ".", ""]
    out += ["## Quality", "", "| purpose | recordings | pass rate | issues |", "|---|---|---|---|"]
    for p, q in rep["quality"].items():
        out.append(f"| {p} | {q['n']} | {_p(q['pass_rate'])} | {', '.join(f'{k} {v}' for k, v in q['issues'].items()) or '—'} |")
    if "error" in rep:
        return "\n".join(out + ["", f"**{rep['error']}**", ""]) + "\n"
    th = rep["thresholds"]
    out += ["", f"## Cross-session identification (SPEC defaults: absFactor {th['abs_factor']:g}, nameMargin "
                f"{th['name_margin']:g}, handMargin {th['hand_margin']:g}, maxPatterns {th['max_patterns']})", ""]
    rows = [("cross-session", rep["cross_session"])]
    if "same_session" in rep:
        rows.append(("same session as enrollment", rep["same_session"]))
    rows += [(f"session {s}", v) for s, v in rep["per_session"].items()]
    if "tuned" in rep:
        t = rep["tuned"]
        rows += [(f"sessions {t['test_sessions']} at defaults", t["default"]),
                 (f"sessions {t['test_sessions']} tuned on session {t['tune_session']} "
                  f"({t['absFactor']:g} / {t['nameMargin']:g}{'' if t['target_met_on_tuning'] else ', target not met'})",
                  t["test"])]
    out += ["| checks | enrolled / stranger checks | top-1 name | hand (given name) | name + hand | stranger accept "
            "| false Unknown | misidentified | patterns per answer | seconds per answer |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in rows:
        out.append(f"| {name} | {s['n_enrolled_checks']} / {s['n_stranger_checks']} | {_p(s['top1_name_acc'])} | "
                   f"{_p(s['hand_acc_given_named'])} | {_p(s['name_and_hand_acc'])} | {_p(s['stranger_accept'])} | "
                   f"{_p(s['false_unknown'])} | {_p(s['misidentified'])} | {s['patterns_per_answer']:.2f} | "
                   f"{s['seconds_per_answer']:.2f} |")
    fc = rep["forced_choice"]
    out += ["", f"Forced choice (nearest prototype, no Unknown) on the same checks: {_p(fc['1_pattern'])} from one "
                f"pattern, {_p(fc['3_patterns'])} from the mean of up to 3 (name + hand {_p(fc['name_and_hand_3_patterns'])}).",
            "", "## Confusion matrix (cross-session checks)", ""]
    cm = rep["confusion"]
    out += ["| truth \\ answer | " + " | ".join(cm["cols"]) + " |", "|---" * (len(cm["cols"]) + 1) + "|"]
    for r, row in zip(cm["rows"], cm["counts"]):
        out.append(f"| {r} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", help="recordings.jsonl exported by the app")
    ap.add_argument("--out", default=None, help="output directory (default: next to the export)")
    ap.add_argument("--target", type=float, default=0.02, help="stranger-accept target for threshold tuning")
    ap.add_argument("--check-gap", type=float, default=8.0, help="seconds between recordings that start a new run")
    ap.add_argument("--abs-factor", type=float, default=Thresholds.abs_factor)
    ap.add_argument("--name-margin", type=float, default=Thresholds.name_margin)
    ap.add_argument("--hand-margin", type=float, default=Thresholds.hand_margin)
    ap.add_argument("--max-patterns", type=int, default=Thresholds.max_patterns)
    args = ap.parse_args(argv)
    src = Path(args.export)
    recs, bad = read_jsonl(src)
    th = Thresholds(abs_factor=args.abs_factor, name_margin=args.name_margin, hand_margin=args.hand_margin,
                    max_patterns=args.max_patterns)
    rep = analyze(recs, th, args.target, args.check_gap)
    rep["unreadable_lines"] = bad
    out = Path(args.out) if args.out else src.parent
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{src.stem}_analysis.json").write_text(json.dumps(clean(rep), indent=1))
    (out / f"{src.stem}_analysis.md").write_text(render(rep, f"Export analysis: {src.name}"))
    s = rep.get("cross_session")
    print(f"{len(recs)} recordings; " + (f"cross-session top-1 {s['top1_name_acc']:.3f}, stranger accept "
                                         f"{s['stranger_accept']:.3f}" if s else rep.get("error", "")))
    print(f"wrote {out / (src.stem + '_analysis.md')}")


if __name__ == "__main__":
    main()
