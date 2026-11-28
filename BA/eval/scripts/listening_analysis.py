"""Analysis of the multi-listener blind listening test (listening_test_protocol.md, pre-specified analysis).

    python3 eval/scripts/listening_analysis.py --qc-only     # step 1: quality control, before any system score is computed
    python3 eval/scripts/listening_analysis.py               # step 2: the pre-specified analysis

Input: the private export of the D1 database (paper/human_eval/listening_test/scripts/export_d1.py,
~/ba-work/listening/export/latest.json) and the blinding key (listening_test/design/design.json, git-ignored).
Output: eval/results/listening_multi.json and .md with aggregates only (no answer of an individual listener).

Rules fixed before the system scores were looked at (protocol: "listeners failing both checks are excluded, and the
exclusion rule is applied before looking at system scores"; the design has 4 real-speech clips and 2 repeats):
  - only finished sessions of eligible listeners; sessions of people involved in the project are excluded (EXCLUDE);
  - real-speech check: passed if at least 2 of the 4 real-speech clips are rated >= 4 on both scales;
  - consistency check: passed if at least 1 of the 2 repeated clips is rated within 1 point of its first rating on
    both scales;
  - a listener failing both checks is excluded.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

BA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DESIGN = os.path.join(BA, "paper", "human_eval", "listening_test", "design", "design.json")
EXPORT = os.path.expanduser("~/ba-work/listening/export/latest.json")
RES = os.path.join(BA, "eval", "results")
EXCLUDE = {"b77b2d56-0313-41c0-8916-4726bbd624e6": "the author, testing the site (knew the hypothesis)"}
SYSTEMS = ["ba", "voxcpm2", "cb-v3", "namaa"]
CONTRASTS = [("ba", "voxcpm2"), ("ba", "cb-v3"), ("voxcpm2", "cb-v3")]
N_BOOT = 10000
# trial ids: p = practice (not analysed), g = rated generated clip, a = real speech (anchor), r = repeat of a g trial
# (consistency check only, not analysed), s = speaker-similarity pair
TRIAL_KIND = {"p": "practice", "g": "generated", "a": "real", "r": "repeat", "s": "similarity"}


def load():
    d = json.load(open(DESIGN))
    x = json.load(open(EXPORT))
    clips = d["clips"]
    lists = {L["id"]: L for L in d["lists"]}
    sessions = {}
    dropped = {"unfinished": 0, "ineligible": 0, "excluded": 0}
    for s in x["sessions"]:
        if not s["finished"]:
            dropped["unfinished"] += 1
        elif not s["eligible"]:
            dropped["ineligible"] += 1
        elif s["id"] in EXCLUDE:
            dropped["excluded"] += 1
        else:
            sessions[s["id"]] = s
    rows = []
    for a in x["answers"]:
        s = sessions.get(a["session_id"])
        if s is None:
            continue
        c = clips[a["clip"]]
        # the role of a rating comes from its trial, not from its clip: one audio file can serve as a rated clip in one
        # list and as one side of a similarity pair in another, and design.json records a single kind per file
        kind = TRIAL_KIND[a["trial_id"][0]]
        rows.append({"listener": a["session_id"], "list": s["list_id"], "trial": a["trial_id"], "clip": a["clip"],
                     "system": c["system"], "kind": kind, "text": c.get("text_id"), "voice": c.get("voice_id", "real"),
                     "egy": a["egy"], "nat": a["nat"], "words_wrong": a["words_wrong"], "sim": a["sim"]})
    return d, x, clips, lists, sessions, rows, dropped


def quality_control(sessions, rows, lists):
    out = {}
    for sid, s in sessions.items():
        mine = {r["trial"]: r for r in rows if r["listener"] == sid}
        real = [r for t, r in mine.items() if t.startswith("a")]
        real_ok = sum(1 for r in real if (r["egy"] or 0) >= 4 and (r["nat"] or 0) >= 4)
        cons = []
        for rep in lists[s["list_id"]]["repeats"]:
            r2, r1 = mine.get(rep["id"]), mine.get(rep["of"])
            if r1 and r2 and None not in (r1["egy"], r2["egy"], r1["nat"], r2["nat"]):
                cons.append(abs(r1["egy"] - r2["egy"]) <= 1 and abs(r1["nat"] - r2["nat"]) <= 1)
        demo = json.loads(s["demo"] or "{}")
        out[sid] = {"real_ok": real_ok, "real_n": len(real), "real_pass": real_ok >= 2,
                    "consistent": sum(cons), "repeats_n": len(cons), "cons_pass": any(cons),
                    "keep": real_ok >= 2 or any(cons), "headphones": demo.get("headphones"),
                    "governorate": demo.get("governorate"), "age": demo.get("age"), "gender": demo.get("gender")}
    return out


def boot_means(rows, key, systems, n_boot=N_BOOT, seed=0):
    """Mean per system with a two-way (listener x sentence) bootstrap percentile interval."""
    rng = np.random.default_rng(seed)
    listeners = sorted({r["listener"] for r in rows})
    texts = sorted({r["text"] for r in rows})
    li = {l: i for i, l in enumerate(listeners)}
    ti = {t: i for i, t in enumerate(texts)}
    out = {}
    for s in systems:
        R = [r for r in rows if r["system"] == s and r[key] is not None]
        if not R:
            continue
        L = np.array([li[r["listener"]] for r in R])
        T = np.array([ti[r["text"]] for r in R])
        y = np.array([float(r[key]) for r in R])
        reps = np.empty(n_boot)
        for b in range(n_boot):
            wl = np.bincount(rng.integers(0, len(listeners), len(listeners)), minlength=len(listeners))
            wt = np.bincount(rng.integers(0, len(texts), len(texts)), minlength=len(texts))
            w = wl[L] * wt[T]
            reps[b] = (w * y).sum() / w.sum() if w.sum() else np.nan
        out[s] = {"mean": float(y.mean()), "ci": [float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))],
                  "n": int(len(y))}
    return out


def krippendorff_ordinal(rows, key):
    """Krippendorff's alpha with the ordinal metric; units = clips, raters = listeners, missing allowed."""
    units = {}
    for r in rows:
        if r[key] is not None and r["kind"] in ("generated", "real"):
            units.setdefault(r["clip"], []).append(int(r[key]))
    units = {u: v for u, v in units.items() if len(v) >= 2}
    cats = list(range(1, 6))
    o = np.zeros((5, 5))
    for v in units.values():
        m = len(v)
        cnt = np.bincount(np.array(v) - 1, minlength=5)
        for c in range(5):
            for k in range(5):
                o[c, k] += (cnt[c] * (cnt[k] - (c == k))) / (m - 1)
    n_c = o.sum(1)
    n = n_c.sum()
    delta = np.zeros((5, 5))
    for c in range(5):
        for k in range(5):
            lo, hi = min(c, k), max(c, k)
            delta[c, k] = (n_c[lo:hi + 1].sum() - (n_c[lo] + n_c[hi]) / 2) ** 2
    d_o = (o * delta).sum() / n
    d_e = (np.outer(n_c, n_c) * delta).sum() / (n * (n - 1))
    return {"alpha": float(1 - d_o / d_e) if d_e else float("nan"), "units": len(units), "pairable_values": int(n)}


def clmm(rows):
    """Cumulative-link mixed model in R (package ordinal): rating ~ system + (1|listener) + (1|voice) + (1|sentence)."""
    script = os.path.join(BA, "eval", "scripts", "listening_clmm.R")
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "ratings.csv")
        with open(csv, "w") as f:
            f.write("listener,system,voice,sentence,egy,nat\n")
            for r in rows:
                if r["kind"] == "generated" and r["egy"] is not None and r["nat"] is not None:
                    f.write(f"{r['listener']},{r['system']},{r['voice']},{r['text']},{r['egy']},{r['nat']}\n")
        out = os.path.join(tmp, "clmm.json")
        env = dict(os.environ, R_LIBS_USER=os.path.expanduser("~/R/library"))
        subprocess.run(["Rscript", script, csv, out], check=True, env=env)
        return json.load(open(out))


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj.tolist()


def automatic_measures():
    """Per-system automatic measures of the same systems (held-out Egyptian texts), for the correlation."""
    bk = json.load(open(os.path.join(RES, "bakeoff.json")))
    d2 = json.load(open(os.path.join(RES, "dialect_classifier2.json")))
    en = json.load(open(os.path.join(RES, "engine.json")))
    out = {}
    for s in ("voxcpm2", "cb-v3", "namaa"):
        em = bk["systems"][s]["conditions"]["egy_main"]
        out[s] = {"c1": em["dialect_egyptian_share"], "c2": d2["systems"][s]["egy_main"]["share_c2"]["mean"],
                  "cer": em["cer_mean"]}
    b = en["ba"]["egy_main"]
    out["ba"] = {"c1": b["dialect_egyptian"], "c2": d2["systems"]["ba"]["egy_main"]["share_c2"]["mean"], "cer": b["cer"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc-only", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="shuffle the system labels within each listener and write nothing: tests the code without "
                         "showing any real system score")
    a = ap.parse_args()
    d, x, clips, lists, sessions, rows, dropped = load()
    qc = quality_control(sessions, rows, lists)
    keep = [s for s, q in qc.items() if q["keep"]]
    print(f"export {x['exported']}: {len(sessions)} complete sessions kept for QC; dropped {dropped}")
    print(f"QC: real-speech check passed by {sum(q['real_pass'] for q in qc.values())}, consistency by "
          f"{sum(q['cons_pass'] for q in qc.values())}; excluded (failed both): {len(qc) - len(keep)}")
    if a.qc_only:
        return
    R = [r for r in rows if r["listener"] in keep]
    if a.selftest:
        rng = np.random.default_rng(1)
        for l in keep:
            G = [r for r in R if r["listener"] == l and r["kind"] == "generated"]
            labs = rng.permutation([r["system"] for r in G])
            for r, s_ in zip(G, labs):
                r["system"] = s_
    gen = [r for r in R if r["kind"] == "generated"]
    real = [r for r in R if r["kind"] == "real"]
    res = {"export": x["exported"], "design": d.get("version"), "source_of_voices": d.get("source"),
           "listeners": len(keep), "excluded_qc": len(qc) - len(keep), "dropped": dropped,
           "excluded_by_rule": list(EXCLUDE.values()),
           "lists_completed": {str(k): v for k, v in sorted(__import__("collections").Counter(
               sessions[s]["list_id"] for s in keep).items())},
           "headphones": __import__("collections").Counter(qc[s]["headphones"] for s in keep),
           "governorates": __import__("collections").Counter(qc[s]["governorate"] for s in keep)}
    for key in ("egy", "nat"):
        res[f"means_{key}"] = boot_means(gen, key, SYSTEMS)
        res[f"means_{key}"].update(boot_means([dict(r, system="real") for r in real], key, ["real"]))
    ww = {}
    for s in SYSTEMS:
        v = [r["words_wrong"] for r in gen if r["system"] == s and r["words_wrong"] is not None]
        ww[s] = {"share": float(np.mean(v)), "n": len(v)}
    v = [r["words_wrong"] for r in real if r["words_wrong"] is not None]
    ww["real"] = {"share": float(np.mean(v)) if v else None, "n": len(v)}
    res["words_wrong"] = ww
    sims = [r for r in R if r["kind"] == "similarity" and r["sim"] is not None]
    res["similarity"] = boot_means([dict(r, text=r["clip"]) for r in sims], "sim", SYSTEMS)
    res["alpha"] = {k: krippendorff_ordinal(R, k) for k in ("egy", "nat")}
    cm = clmm(R)
    ps = [c["p"] for c in cm["contrasts"]]
    for c, padj in zip(cm["contrasts"], holm(ps)):
        c["p_holm"] = padj
    res["clmm"] = cm
    auto = automatic_measures()
    from scipy.stats import spearmanr, pearsonr
    m = res["means_egy"]
    res["automatic"] = auto
    res["correlation_egy"] = {}
    for k in ("c1", "c2", "cer"):
        xs = [m[s]["mean"] for s in SYSTEMS]
        ys = [auto[s][k] for s in SYSTEMS]
        res["correlation_egy"][k] = {"spearman": float(spearmanr(xs, ys).statistic), "pearson": float(pearsonr(xs, ys).statistic),
                                     "n_systems": len(SYSTEMS)}
    # sensitivity (the protocol asked for headphones; the page also accepted earbuds and loudspeakers)
    res["by_device"] = {}
    for tag, devs in (("headphones_or_earbuds", ("headphones", "earbuds")), ("loudspeaker", ("speaker",))):
        ls = [s for s in keep if qc[s]["headphones"] in devs]
        if len(ls) >= 3:
            H = [r for r in gen if r["listener"] in ls]
            res["by_device"][tag] = {"listeners": len(ls), "means_egy": boot_means(H, "egy", SYSTEMS, n_boot=2000),
                                     "means_nat": boot_means(H, "nat", SYSTEMS, n_boot=2000)}
    res["headphones"] = dict(res["headphones"])
    res["governorates"] = dict(res["governorates"])
    if a.selftest:
        print("selftest finished: CLMM converged", {k: v["converged"] for k, v in cm["fits"].items()},
              "| contrasts", len(cm["contrasts"]), "| alpha units", res["alpha"]["egy"]["units"])
        return
    json.dump(res, open(os.path.join(RES, "listening_multi.json"), "w"), indent=1, ensure_ascii=False)
    lines = [f"# Multi-listener listening test ({res['listeners']} listeners, export {res['export']})", ""]
    for key in ("egy", "nat"):
        lines.append(f"{key}: " + "; ".join(f"{s} {v['mean']:.2f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]"
                                           for s, v in res[f"means_{key}"].items()))
    for c in cm["contrasts"]:
        lines.append(f"CLMM {c['scale']} {c['a']} - {c['b']}: log-odds {c['est']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}], "
                     f"p {c['p']:.4f}, Holm {c['p_holm']:.4f}")
    lines.append("alpha: " + ", ".join(f"{k} {v['alpha']:.2f} ({v['units']} clips)" for k, v in res["alpha"].items()))
    open(os.path.join(RES, "listening_multi.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
