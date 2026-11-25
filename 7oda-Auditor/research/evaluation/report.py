"""Markdown rendering of evaluation reports."""
from __future__ import annotations

import math

HEAD = [("top1_name_acc", "top-1 name"), ("hand_acc_given_named", "hand (given name)"),
        ("name_and_hand_acc", "name+hand"), ("stranger_accept", "stranger accept"),
        ("false_unknown", "false Unknown"), ("misidentified", "wrong name"), ("hand_unclear", "hand unclear"),
        ("quality_fail", "quality fail"), ("patterns_per_answer", "patterns/answer"),
        ("seconds_per_answer", "s/answer")]


def _f(v, pct=True):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if pct:
        return f"{100 * v:.1f}%"
    return f"{v:.2f}"


def _row(name, s, extra=""):
    cells = [_f(s[k], k not in ("patterns_per_answer", "seconds_per_answer")) for k, _ in HEAD]
    return f"| {name} | " + " | ".join(cells) + f" |{extra}"


def _table(rows, first="setting"):
    out = [f"| {first} | " + " | ".join(h for _, h in HEAD) + " |", "|---" * (len(HEAD) + 1) + "|"]
    out += rows
    return "\n".join(out)


def _th(t):
    return f"absFactor {t['abs_factor']:g}, nameMargin {t['name_margin']:g}"


def render(rep: dict, title: str) -> str:
    L = [f"# {title}", ""]
    cfg = rep.get("config", {})
    if cfg:
        L.append(f"IMU mode `{cfg.get('mode')}`, {cfg.get('n_people')} enrolled + {cfg.get('n_strangers')} strangers, "
                 f"{cfg.get('checks_per_session')} checks per person per session, phone seed {cfg.get('phone_seed')}, "
                 f"study seed {cfg.get('seed')}. Enrollment: session {cfg.get('enroll_session')}; thresholds tuned on "
                 f"session {rep['tune_session']}; test sessions {rep['test_sessions']}.")
        L.append("")
        L.append("Simulated numbers check the pipeline's mechanics; they are not evidence about real hands "
                 "(DECISIONS D9).")
        L.append("")
        L.append("How to read: *top-1 name* counts checks that were answered with the right name; Unknown answers "
                 "count as misses. With thresholds tuned for a low stranger accept rate, a model that ranks the "
                 "right person first can still answer Unknown most of the time, so the forced-choice (closed-set) "
                 "accuracy and the operating points below show how much of the gap is the features and how much "
                 "is the Unknown option.")
        L.append("")
    m = rep["main"]
    L += ["## Headline (test sessions, sequential checks up to 3 patterns)", "",
          _table([_row(f"tuned ({_th(m['tuned'])})", m["test"]), _row("SPEC defaults (1.5, 6)", m["default"])]), "",
          f"Tuning target stranger accept <= {100 * rep['stranger_target']:.0f}% on the tuning session: "
          f"{'met' if m['target_met_on_tuning'] else 'NOT met (closest setting used)'}. "
          f"Checks: {m['test']['n_enrolled_checks']} enrolled, {m['test']['n_stranger_checks']} stranger."
          + (f" Tuned value on the grid edge: {', '.join(m['tuned_on_grid_edge'])}." if m.get("tuned_on_grid_edge")
             else ""), ""]
    cs = m.get("closed_set")
    if cs:
        L += ["## Forced choice (no Unknown option)", "",
              "Nearest prototype for enrolled people's checks: first passing pattern, and the mean distance of the "
              f"first {cs['patterns']} passing patterns. Chance = {_f(cs['chance'])}.", "",
              "| sessions | checks | top-1, 1 pattern | top-1, mean of patterns | name+hand |", "|---|---|---|---|---|",
              f"| test {rep['test_sessions']} | {cs['n']} | {_f(cs['top1_first_pattern'])} | "
              f"{_f(cs['top1_mean_patterns'])} | {_f(cs['name_and_hand_mean_patterns'])} |"]
        for k, v in rep.get("closed_set_per_session", {}).items():
            L.append(f"| session {k} | {v['n']} | {_f(v['top1_first_pattern'])} | {_f(v['top1_mean_patterns'])} | "
                     f"{_f(v['name_and_hand_mean_patterns'])} |")
        dg = rep.get("diagnostics", {})
        if "same_day_top1_first_pattern" in dg:
            L += ["", f"Same-day reference (session 1: enroll on the sitting and standing rounds, test the fresh-grip "
                      f"round, 1 pattern): {_f(dg['same_day_top1_first_pattern'])}."]
        L.append("")
    ops = rep.get("operating_points")
    if ops:
        L += ["## Operating points (thresholds chosen on the tuning session for each stranger target)", "",
              "| target | absFactor | nameMargin | tuning: stranger / top-1 | test: top-1 name | test: stranger accept "
              "| test: false Unknown | test: wrong name |", "|---|---|---|---|---|---|---|---|"]
        for o in ops:
            t = o["test"]
            L.append(f"| {_f(o['target'])}{'' if o['met_on_tuning'] else ' (not met)'} | {o['absFactor']:g} | "
                     f"{o['nameMargin']:g} | {_f(o['tuning_stranger_accept'])} / {_f(o['tuning_top1'])} | "
                     f"{_f(t['top1_name_acc'])} | {_f(t['stranger_accept'])} | {_f(t['false_unknown'])} | "
                     f"{_f(t['misidentified'])} |")
        L.append("")
    dg = rep.get("diagnostics", {})
    if dg.get("features"):
        fe = sorted(dg["features"], key=lambda r: -r["fisher_cross_session"])
        weak = sum(r["fisher_cross_session"] < 0.3 for r in fe)
        L += ["## Where the identity information is (test sessions, enrolled people)", "",
              "Variance of each feature split into between person-hand, session (day), grip (check) and trial "
              "parts; Fisher = between / within (same-session: grip + trial; cross-session: + session).", "",
              "| feature | between | session | grip | trial | Fisher same-session | Fisher cross-session |",
              "|---|---|---|---|---|---|---|"]
        for r in fe[:10]:
            L.append(f"| {r['index']} {r['name']} | {r['between']:.3g} | {r['session']:.3g} | {r['grip']:.3g} | "
                     f"{r['trial']:.3g} | {r['fisher_same_session']:.2f} | {r['fisher_cross_session']:.2f} |")
        ev = ", ".join(f"{v:.2f}" for v in dg["discriminant_eigenvalues"][:8])
        L += ["", f"{weak} of {len(fe)} features have a cross-session Fisher ratio below 0.3 (full table in the JSON "
                  f"file, `diagnostics.features`). Discriminant eigenvalues (W^-1 B): {ev}; "
                  f"{dg['n_dims_ratio_gt_1']} directions with ratio > 1."]
        if "oracle_top1_at_stranger_accept" in dg:
            ob = dg["oracle_top1_at_stranger_accept"]
            L += ["", "Optimistic bound for the threshold rule (abs and name thresholds chosen on the test checks "
                      "themselves, single decision on the 3-pattern mean): top-1 name "
                  + ", ".join(f"{_f(v)} at stranger accept <= {_f(float(k))}" for k, v in ob.items())
                  + f". Median best distance: genuine {dg['genuine_best_d2_median']:.1f}, stranger "
                    f"{dg['stranger_best_d2_median']:.1f}; median name margin: genuine "
                    f"{dg['genuine_margin_median']:.1f}, stranger {dg['stranger_margin_median']:.1f}."]
        L.append("")
    L += ["## Per session (tuned thresholds)", "",
          _table([_row(f"session {s}", v) for s, v in rep["per_session"].items()], "session"), ""]
    L += ["## Threshold trade-off on the tuning session (Pareto frontier)", "",
          "| absFactor | nameMargin | stranger accept | false Unknown | top-1 name | wrong name |", "|---|---|---|---|---|---|"]
    for r in m.get("pareto", []):
        L.append(f"| {r['absFactor']:g} | {r['nameMargin']:g} | {_f(r['stranger_accept'])} | {_f(r['false_unknown'])} | "
                 f"{_f(r['top1_name_acc'])} | {_f(r['misidentified'])} |")
    L.append("")
    ab = rep.get("ablations", {})
    for key, title2 in (("patterns", "Patterns per check (re-tuned each)"),
                        ("features", "Feature sets (re-tuned each)"),
                        ("sittings", "Enrollment sittings (re-tuned each)")):
        if key in ab:
            rows = [_row(f"{v['name']} ({_th(v['tuned'])})", v["test"]) for v in ab[key]]
            if key == "features":
                rows.insert(0, _row(f"all 36 features ({_th(m['tuned'])})", m["test"]))
            L += [f"## {title2}", "", _table(rows), ""]
    if "accessory" in ab:
        a = ab["accessory"]
        L += ["## Accessories (enrolled bare; checks with the person's own accessory)", "",
              _table([_row("before quick add", a["before_quick_add"]), _row("after quick add", a["after_quick_add"]),
                      _row("bare-hand checks after quick add", a["bare_checks_after_quick_add"])]), "",
              "| accessory | top-1 before | top-1 after |", "|---|---|---|"]
        for k, v in a["by_accessory"].items():
            L.append(f"| {k} | {_f(v['before'])} | {_f(v['after'])} |")
        L.append("")
    if "cross_hand" in ab:
        L += ["## Cross-hand test (enroll one hand, check with the other: the right answer is Unknown)", "",
              "| setting | checks | Unknown | named self | named someone else | quality fail | same-hand control top-1 |",
              "|---|---|---|---|---|---|---|"]
        for k, v in ab["cross_hand"].items():
            L.append(f"| {k} | {v['n']} | {_f(v['unknown'])} | {_f(v['named_self'])} | {_f(v['named_someone_else'])} | "
                     f"{_f(v['quality_fail'])} | {_f(v['same_hand_control_top1'])} |")
        L.append("")
    if "replay" in ab:
        r = ab["replay"]
        iss = ", ".join(f"{k} {v}" for k, v in sorted(r["rejection_issues"].items(), key=lambda kv: -kv[1]))
        L += ["## Replay attack (victim's recorded responses fed to fresh challenges, padded to length)", "",
              f"- patterns rejected by the quality gate: {_f(r['pattern_rejected_by_quality'])} of {r['n_patterns']} "
              f"(median NCC {r['ncc_median']:.2f}; issues: {iss})",
              f"- checks ending in a quality failure: {_f(r['checks_quality_fail'])} of {r['n_checks']}",
              f"- checks accepted as the victim: {_f(r['checks_accepted_as_victim'])}; as anyone: "
              f"{_f(r['checks_accepted_as_anyone'])}", ""]
    if "mimicry" in ab:
        r = ab["mimicry"]
        L += ["## Mimicry proxy (stranger copies the victim's grip angles)", "",
              f"- {r['n_checks']} checks: accepted as the victim {_f(r['accepted_as_victim'])}, accepted as anyone "
              f"{_f(r['accepted_as_anyone'])}",
              f"- baseline (ordinary stranger checks, same session): accepted as anyone "
              f"{_f(r['baseline_stranger_accept_same_session'])}; chance of naming one given victim "
              f"{_f(r['baseline_named_victim_rate'])} x accept rate", ""]
    if "sensitivity" in rep:
        L += ["## Sensitivity to day-to-day variability (main setting, re-tuned)", "",
              _table([_row(f"day_scale {k}", v["test"]) for k, v in rep["sensitivity"].items()], "day-to-day scale"),
              "", "Forced choice (no Unknown option, mean of up to 3 patterns): "
              + ", ".join(f"day_scale {k} {_f(v['closed_set']['top1_mean_patterns'])}"
                          for k, v in rep["sensitivity"].items() if v.get("closed_set")) + ".",
              "", "day_scale 0 means every session reproduces the enrollment day's grip and tissue state "
              "(only fresh-grip and trial variation remain); 1 is the default simulator.", ""]
    q = rep.get("quality", {})
    if q:
        L += ["## Quality gate on simulated recordings", "", "| role | n | pass | median SNR | median NCC | issues |",
              "|---|---|---|---|---|---|"]
        for k, v in q.items():
            iss = ", ".join(f"{a} {b}" for a, b in sorted(v["issues"].items(), key=lambda kv: -kv[1]))
            L.append(f"| {k} | {v['n']} | {_f(v['pass_rate'])} | {v['snr_median']:.1f} | {v['ncc_median']:.2f} | {iss} |")
        L.append("")
    conf = rep.get("confusion")
    if conf:
        pairs = []
        for i, r in enumerate(conf["rows"]):
            for j, c in enumerate(conf["cols"]):
                if r != c and c not in ("Unknown", "qualityFail") and conf["counts"][i][j] > 0:
                    pairs.append((conf["counts"][i][j], r, c))
        pairs.sort(reverse=True)
        L += ["## Most frequent confusions (test sessions, tuned)", "",
              ", ".join(f"{r} -> {c} ({n})" for n, r, c in pairs[:12]) or "none", "",
              "Full confusion matrix in the JSON file (`confusion`).", ""]
    return "\n".join(L)
