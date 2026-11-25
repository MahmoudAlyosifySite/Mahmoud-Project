"""Feature-set comparison on the simulated study -> research/results/feature_comparison.{json,md}.

    cd research && python -m evaluation.compare_features --mode both --workers 16

Simulates the main study once per IMU mode (same seeds as evaluation.run_sim_study; the
trials do not depend on which features are computed) with feature sets v1 and v2 on every
item, then evaluates each set, and two combinations, with the same protocol: enrollment in
session 1; forced choice (no Unknown) from 1 pattern and from the mean of 3; threshold
tuning on session 2 for stranger-accept targets of 2, 5 and 10 %, results on sessions 3-4;
per-feature cross-session Fisher ratios and discriminant dimensions (separability).

Feature vectors are compared as absolute values: on one phone the calibration offset is a
constant and does not change identification (DECISIONS D5).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from auditor_ref import features as adopted
from auditor_ref import features_v1, features_v2

from .decision import Thresholds
from .protocol import (Scorer, checks_where, closed_set, dataset_from_study, enrollment_samples, grid_rows, pick,
                       separability, summarize)
from .run_sim_study import RESULTS, clean, load_or_run
from .study import StudyConfig

TARGETS = (0.02, 0.05, 0.10)
V1_NAMES = tuple(features_v1.FEATURE_NAMES)
V2_NAMES = tuple(features_v2.FEATURE_NAMES)
V2_TAIL = [i for i, n in enumerate(V2_NAMES) if n.startswith("tap_tail_")]


def _key(version: int) -> str:
    return "f_abs" if adopted.FEATURE_VERSION == version else f"f_abs_v{version}"


def feature_sets() -> dict:
    """name -> (list of (item key, column indices), feature names)."""
    k1, k2 = _key(1), _key(2)
    no_tail = [i for i in range(len(V2_NAMES)) if i not in V2_TAIL]
    return {
        "v1 (d=36)": ([(k1, list(range(36)))], list(V1_NAMES)),
        "v2 (d=39)": ([(k2, list(range(39)))], list(V2_NAMES)),
        "v2 without free-response features (d=29)": ([(k2, no_tail)], [V2_NAMES[i] for i in no_tail]),
        "v1 + v2 free-response features (d=46)": ([(k1, list(range(36))), (k2, V2_TAIL)],
                                                   list(V1_NAMES) + [V2_NAMES[i] for i in V2_TAIL]),
    }


def evaluate_set(items, calibration, parts, names, tune_ids, test_ids, test_sessions) -> dict:
    for it in items:
        it["_cmp"] = np.concatenate([np.asarray(it[k], float)[idx] if k in it else np.full(len(idx), np.nan)
                                     for k, idx in parts])
    ds = dataset_from_study(items, calibration, key="_cmp", names=names)
    sc = Scorer(ds, enrollment_samples(ds))
    out = {"d": len(names)}
    c1 = closed_set(sc, test_ids, max_patterns=1)
    c3 = closed_set(sc, test_ids, max_patterns=3)
    out["forced_choice"] = {"n": c3["n"], "top1_1_pattern": c1["top1_first_pattern"],
                            "top1_3_patterns": c3["top1_mean_patterns"],
                            "name_and_hand_3_patterns": c3["name_and_hand_mean_patterns"]}
    rows = grid_rows(sc, tune_ids, Thresholds())
    ops = []
    for t in TARGETS:
        row, met = pick(rows, t)
        th = replace(Thresholds(), abs_factor=row["absFactor"], name_margin=row["nameMargin"])
        s = summarize(sc.run(test_ids, th), ds.enrolled)
        ops.append({"target": t, "met_on_tuning": met, "absFactor": row["absFactor"], "nameMargin": row["nameMargin"],
                    **{k: s[k] for k in ("top1_name_acc", "name_and_hand_acc", "stranger_accept", "false_unknown",
                                         "misidentified", "patterns_per_answer")}})
    out["operating_points"] = ops
    # the SPEC decision rule with the grid point chosen on the test checks themselves: removes the noise of
    # transferring a tuned grid point from session 2 (optimistic, the same for every feature set)
    test_rows = grid_rows(sc, test_ids, Thresholds())
    out["grid_oracle"] = []
    for t in TARGETS:
        row, met = pick(test_rows, t)
        out["grid_oracle"].append({"target": t, "met": met, **{k: row[k] for k in ("absFactor", "nameMargin",
                                   "top1_name_acc", "stranger_accept", "false_unknown", "patterns_per_answer")}})
    diag = separability(ds, sc, test_sessions)
    fis = {f["name"]: f["fisher_cross_session"] for f in diag["features"]}
    out["fisher_cross_session"] = fis
    out["n_fisher_below_0.3"] = int(sum(v < 0.3 for v in fis.values()))
    out["discriminant_eigenvalues"] = diag["discriminant_eigenvalues"][:6]
    out["n_dims_ratio_gt_1"] = diag["n_dims_ratio_gt_1"]
    out["oracle_top1_at_stranger_accept"] = diag.get("oracle_top1_at_stranger_accept")
    for it in items:
        del it["_cmp"]
    return out


def compare_mode(mode: str, args, log) -> dict:
    t0 = time.time()
    extra = tuple(f"v{v}" for v in (1, 2) if v != adopted.FEATURE_VERSION)
    cfg = StudyConfig(mode=mode, seed=args.seed, phone_seed=args.phone_seed, workers=args.workers,
                      nogyro=False, extra_features=extra)
    data = load_or_run(cfg, Path(args.cache), f"{mode}_cmp_s{args.seed}_p{args.phone_seed}", log)
    items, cal = data["items"], data["calibration"]
    probe = dataset_from_study(items, cal)
    tune_ids = checks_where(probe, "test", [cfg.tune_session])
    test_sessions = [s for s in cfg.test_sessions if s != cfg.tune_session]
    test_ids = checks_where(probe, "test", test_sessions)
    res = {"mode": mode, "config": asdict(cfg), "tune_session": cfg.tune_session, "test_sessions": test_sessions,
           "sets": {}}
    for name, (parts, names) in feature_sets().items():
        res["sets"][name] = evaluate_set(items, cal, parts, names, tune_ids, test_ids, test_sessions)
        r = res["sets"][name]
        log(f"[{mode}] {name}: forced choice {r['forced_choice']['top1_1_pattern']:.3f}/"
            f"{r['forced_choice']['top1_3_patterns']:.3f}, open-set top-1 "
            + ", ".join(f"{o['target']:.0%}: {o['top1_name_acc']:.3f}" for o in r["operating_points"])
            + "; grid oracle " + ", ".join(f"{o['top1_name_acc']:.3f}" for o in r["grid_oracle"]))
    res["timing_s"] = round(time.time() - t0, 1)
    return res


def _pct(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.1f} %"


def render(results: dict, args) -> str:
    out = ["# Feature sets v1 vs v2 on the simulated study", "",
           f"Generated by `python -m evaluation.compare_features` (study seed {args.seed}, phone {args.phone_seed}; "
           "the default seeds give the same trials as `run_sim_study`): 20 enrolled people + 5 strangers, enrollment in session 1 (15 patterns per hand), thresholds "
           "tuned on session 2, results on sessions 3-4. Simulated numbers check the pipeline; they are not "
           "evidence about real hands (DECISIONS D9).", ""]
    for mode, res in results.items():
        out += [f"## IMU mode `{mode}`", "",
                "| Feature set | d | forced choice, 1 pattern | forced choice, 3 patterns | features with Fisher < 0.3 "
                "| discriminant dims > 1 | top-1 @2 / 5 / 10 % (tuned on session 2) | top-1 @2 / 5 / 10 % (grid point "
                "chosen on the test checks) |",
                "|---|---|---|---|---|---|---|---|"]
        for name, r in res["sets"].items():
            ops = {o["target"]: o for o in r["operating_points"]}
            gor = {o["target"]: o for o in r["grid_oracle"]}
            out.append(f"| {name} | {r['d']} | {_pct(r['forced_choice']['top1_1_pattern'])} | "
                       f"{_pct(r['forced_choice']['top1_3_patterns'])} | {r['n_fisher_below_0.3']} | "
                       f"{r['n_dims_ratio_gt_1']} | " + " / ".join(_pct(ops[t]["top1_name_acc"]) for t in TARGETS)
                       + " | " + " / ".join(_pct(gor[t]["top1_name_acc"]) for t in TARGETS) + " |")
        out += ["", "Operating points (tuned on session 2 for the stranger-accept target; test sessions 3-4):", "",
                "| Feature set | target | absFactor / nameMargin | top-1 | name + hand | stranger accept | false Unknown "
                "| misidentified | patterns per answer |", "|---|---|---|---|---|---|---|---|---|"]
        for name, r in res["sets"].items():
            for o in r["operating_points"]:
                out.append(f"| {name} | {o['target']:.0%}{'' if o['met_on_tuning'] else ' (not met)'} | "
                           f"{o['absFactor']:g} / {o['nameMargin']:g} | {_pct(o['top1_name_acc'])} | "
                           f"{_pct(o['name_and_hand_acc'])} | {_pct(o['stranger_accept'])} | "
                           f"{_pct(o['false_unknown'])} | {_pct(o['misidentified'])} | "
                           f"{o['patterns_per_answer']:.2f} |")
        out += ["", "Cross-session Fisher ratios (between person-hand / within: day + grip + trial):", ""]
        for name, r in res["sets"].items():
            if name.startswith(("v1 (", "v2 (")):
                fis = r["fisher_cross_session"]
                out.append(f"- {name}: " + ", ".join(f"{k} {v:.2f}" for k, v in fis.items()))
                out.append(f"  - discriminant eigenvalues: {', '.join(f'{v:.1f}' for v in r['discriminant_eigenvalues'])}")
        out.append("")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="both", choices=("none", "lpf", "both"))
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--phone-seed", type=int, default=7)
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--cache", default=str(Path.home() / "7oda-work" / "sim"))
    ap.add_argument("--tag", default="", help="suffix of the output file names")
    args = ap.parse_args(argv)
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    results = {}
    for mode in (("none", "lpf") if args.mode == "both" else (args.mode,)):
        results[mode] = compare_mode(mode, args, log)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        stem = "feature_comparison" + (f"_{args.tag}" if args.tag else "")
        (out / f"{stem}.json").write_text(json.dumps(clean(results), indent=1))
        (out / f"{stem}.md").write_text(render(results, args))


if __name__ == "__main__":
    main()
