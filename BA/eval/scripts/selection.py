"""Experiment 5: what each selection metric picks (venv /tmp/ba/venvs/eval, CPU). -> BA/eval/results/selection.json, .md

Question: if a practitioner chose the base model by one of the standard automatic metrics (lowest CER, highest
ECAPA speaker similarity, highest DNSMOS), how often would the chosen system fail the dialect bar?

Data: the Phase 1 bake-off, Egyptian main set (32 non-digit texts x 3 Egyptian voices, one seed per utterance),
for the nine configurations; classifier 1 labels as stored, classifier 2 from dialect_classifier2.py's cache.
A second block repeats the analysis on the held-out three-seed runs (24 texts x 3 voices, uncertainty.py's data)
for the systems that have three seeds.

Candidate sets:
  eligible  the licence-eligible bases of G1 (VoxCPM2, Chatterbox V2, Chatterbox V3), each at its published
            defaults; this is the decision G1 actually made.
  all       all nine configurations (bases, fine-tunes, non-commercial references, VoxCPM2 prompt mode and the
            diagnostic PyPI loader of Chatterbox V3).
Dialect bar: 60% of the share on the lahgtna colloquial calibration clips, per classifier (G3's rule).

Method: one shared two-way bootstrap (texts and voices resampled independently with replacement, the same draw
for every system, B = 10,000, paper_common's seed). In each resample every system gets a weighted mean of each
metric; each selection rule picks its winner; we record the winner and whether its share under classifier 1 and
under classifier 2 in that resample is below the bar. Reported: the share of resamples in which each system is
picked, the share in which the pick fails the bar, and the bootstrap distribution of Spearman's rho between CER
and the Egyptian share across systems.

Run: /tmp/ba/venvs/eval/bin/python BA/eval/scripts/selection.py
"""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from paper_common import (ENGINE_ROOT, RESULTS, jl, bakeoff_rows, engine_rows, held_out, attach_dialect, B_DEFAULT,
                          RNG_SEED, two_way_bootstrap, ci, dump)

ALL = ["cb-v3", "cb-v3-pypi", "cb-v2", "voxcpm2", "namaa", "lahgtna", "omnivoice", "voicetut", "voxcpm2-prompt"]
ELIGIBLE = ["voxcpm2", "cb-v2", "cb-v3"]
RULES = {"cer": ("cer_best", -1), "ecapa": ("sim_ecapa", +1), "dnsmos": ("mos_p808", +1)}
METRICS = ["cer_best", "sim_ecapa", "mos_p808", "c1", "c2"]


def calib_bar():
    """60% of each classifier's Egyptian share on the lahgtna colloquial calibration clips."""
    d2 = json.load(open(f"{RESULTS}/dialect_classifier2.json"))
    real = d2["calibration_real_speech"]["lahgtna_eg"]
    s1, s2 = real["share_c1"], real["share_c2"]
    return {"c1": 0.6 * s1, "c2": 0.6 * s2, "real_c1": s1, "real_c2": s2}


def cell_table(rows_by_sys):
    """-> cells (sorted list of (text, voice)), {sys: {metric: array over cells}} (seeds averaged per cell)."""
    cells = None
    for rows in rows_by_sys.values():
        cs = {(r["text_id"], r["voice_id"]) for r in rows}
        cells = cs if cells is None else cells & cs
    cells = sorted(cells)
    pos = {c: i for i, c in enumerate(cells)}
    out = {}
    for s, rows in rows_by_sys.items():
        acc = {m: [[] for _ in cells] for m in METRICS}
        for r in rows:
            i = pos.get((r["text_id"], r["voice_id"]))
            if i is None:
                continue
            for m in METRICS:
                v = r.get(m)
                if v is not None:
                    acc[m][i].append(float(v))
        out[s] = {m: np.array([np.mean(v) if v else np.nan for v in acc[m]]) for m in METRICS}
    return cells, out


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else 0.0


def analyse(rows_by_sys, sets, bar, B=B_DEFAULT, seed=RNG_SEED):
    cells, T = cell_table(rows_by_sys)
    texts = sorted({c[0] for c in cells}); voices = sorted({c[1] for c in cells})
    ti = np.array([texts.index(c[0]) for c in cells]); vi = np.array([voices.index(c[1]) for c in cells])
    rng = np.random.default_rng(seed)
    Ct = rng.multinomial(len(texts), np.full(len(texts), 1 / len(texts)), size=B).astype(float)
    Cv = rng.multinomial(len(voices), np.full(len(voices), 1 / len(voices)), size=B).astype(float)
    W = Ct[:, ti] * Cv[:, vi]                                   # B x cells
    systems = list(rows_by_sys)
    M = {}                                                      # metric -> B x systems
    point = {}
    for m in METRICS:
        cols, pts = [], []
        for s in systems:
            x = T[s][m]
            ok = ~np.isnan(x)
            w = W[:, ok]
            cols.append((w @ x[ok]) / np.maximum(w.sum(1), 1e-12))
            pts.append(float(np.nanmean(x)))
        M[m] = np.stack(cols, 1)
        point[m] = dict(zip(systems, [round(p, 4) for p in pts]))
    out = {"n_cells": len(cells), "n_texts": len(texts), "n_voices": len(voices), "B": B, "point": point, "rules": {}}
    for set_name, members in sets.items():
        idx = [systems.index(s) for s in members if s in systems]
        names = [systems[i] for i in idx]
        res = {"members": names}
        for rule, (m, sign) in RULES.items():
            sub = sign * M[m][:, idx]
            win = np.argmax(sub, 1)
            picked = collections.Counter(names[w] for w in win)
            c1w = M["c1"][np.arange(B), np.array(idx)[win]]
            c2w = M["c2"][np.arange(B), np.array(idx)[win]]
            point_vals = {s: point[m][s] for s in names}
            point_pick = (min if sign < 0 else max)(point_vals, key=point_vals.get)
            res[rule] = dict(
                point_pick=point_pick,
                point_pick_c1=point["c1"][point_pick], point_pick_c2=point["c2"][point_pick],
                picked_share={k: round(v / B, 4) for k, v in picked.most_common()},
                fails_bar_c1=round(float(np.mean(c1w < bar["c1"])), 4),
                fails_bar_c2=round(float(np.mean(c2w < bar["c2"])), 4),
                fails_bar_both=round(float(np.mean((c1w < bar["c1"]) & (c2w < bar["c2"]))), 4))
        # the dialect-aware rule (G1 amended): lowest CER among systems whose c1 share clears the bar
        sub = np.where(M["c1"][:, idx] >= bar["c1"], -M["cer_best"][:, idx], -np.inf)
        has = np.isfinite(sub).any(1)
        win = np.argmax(sub, 1)
        picked = collections.Counter(names[w] if h else "none" for w, h in zip(win, has))
        res["amended"] = dict(picked_share={k: round(v / B, 4) for k, v in picked.most_common()})
        if len(idx) >= 4:
            rho1 = np.array([spearman(M["cer_best"][b, idx], M["c1"][b, idx]) for b in range(B)])
            rho2 = np.array([spearman(M["cer_best"][b, idx], M["c2"][b, idx]) for b in range(B)])
            res["spearman_cer_c1"] = dict(point=round(spearman(np.array([point["cer_best"][s] for s in names]),
                                                               np.array([point["c1"][s] for s in names])), 3),
                                          lo=round(float(np.percentile(rho1, 2.5)), 3), hi=round(float(np.percentile(rho1, 97.5)), 3))
            res["spearman_cer_c2"] = dict(point=round(spearman(np.array([point["cer_best"][s] for s in names]),
                                                               np.array([point["c2"][s] for s in names])), 3),
                                          lo=round(float(np.percentile(rho2, 2.5)), 3), hi=round(float(np.percentile(rho2, 97.5)), 3))
        out["rules"][set_name] = res
    return out


def main():
    bar = calib_bar()
    bk = {s: [r for r in attach_dialect(bakeoff_rows(s)) if r["cond"] == "egy_main"] for s in ALL}
    one_seed = analyse(bk, {"eligible": ELIGIBLE, "all": ALL}, bar)
    # held-out, three seeds (uncertainty.py's data) for the systems that have them
    seeded = {"voxcpm2": ("voxcpm2", "p_seeds_voxcpm2"), "cb-v3": ("cb-v3", "p_seeds_cbv3")}
    rows3 = {}
    for s, (b, exp) in seeded.items():
        r0 = [r for r in held_out(attach_dialect(bakeoff_rows(b))) if r["cond"] == "egy_main"]
        r12 = [r for r in held_out(attach_dialect(engine_rows(exp, s))) if r["cond"] == "egy_main"]
        rows3[s] = r0 + r12
    three_seed = analyse(rows3, {"voxcpm2_vs_cbv3": list(seeded)}, bar)
    # dialect share by the dialect of the reference clip (both classifiers' labels of the reference itself)
    d2 = json.load(open(f"{RESULTS}/dialect_classifier2.json"))
    ref = {r["voice_id"]: (r["c1"] == "Egyptian", r["c2"] == "EGY") for r in d2["reference_clips"] if r["clip"] == "ref"}
    by_ref = {}
    for s in ["cb-v3", "cb-v2", "voxcpm2", "voxcpm2-prompt", "namaa", "omnivoice", "voicetut", "lahgtna"]:
        rows = bk[s]
        d = {}
        for name, sel in [("ref_egyptian_both", lambda v: ref[v][0] and ref[v][1]),
                          ("ref_msa_both", lambda v: not ref[v][0] and not ref[v][1])]:
            rs = [r for r in rows if sel(r["voice_id"])]
            d[name] = {"n": len(rs), "voices": sorted({r["voice_id"] for r in rs}),
                       "c1": two_way_bootstrap([r["c1"] for r in rs], [r["text_id"] for r in rs], [r["voice_id"] for r in rs]),
                       "c2": two_way_bootstrap([r["c2"] for r in rs], [r["text_id"] for r in rs], [r["voice_id"] for r in rs]),
                       "cer_best": two_way_bootstrap([r["cer_best"] for r in rs], [r["text_id"] for r in rs], [r["voice_id"] for r in rs])}
        by_ref[s] = d
    out = {"design": __doc__, "bar": {k: round(v, 4) for k, v in bar.items()}, "bakeoff_one_seed": one_seed,
           "heldout_three_seeds": three_seed, "by_reference_dialect": by_ref, "reference_labels": ref,
           "command": "/tmp/ba/venvs/eval/bin/python BA/eval/scripts/selection.py"}
    dump(out, f"{RESULTS}/selection.json")

    L = ["# What each selection metric picks", "",
         "Generated by `BA/eval/scripts/selection.py` from `selection.json`. Shared two-way bootstrap over texts and voices "
         f"(B = {B_DEFAULT}). Dialect bar = 60% of the share on real colloquial speech (lahgtna): classifier 1 "
         f"{bar['c1']:.3f}, classifier 2 {bar['c2']:.3f}.", ""]
    for block, name in [(one_seed, "Bake-off, Egyptian main set, one seed"), (three_seed, "Held-out, three seeds")]:
        L += [f"## {name} ({block['n_texts']} texts x {block['n_voices']} voices)", "",
              "| set | rule | pick (point) | its share clf 1 / clf 2 | picked in resamples | pick fails bar clf 1 | clf 2 | both |",
              "|---|---|---|---|---|---|---|---|"]
        for set_name, res in block["rules"].items():
            for rule in RULES:
                r = res[rule]
                ps = ", ".join(f"{k} {v:.3f}" for k, v in r["picked_share"].items())
                L.append(f"| {set_name} | {rule} | {r['point_pick']} | {r['point_pick_c1']:.3f} / {r['point_pick_c2']:.3f} | {ps} | "
                         f"{r['fails_bar_c1']:.3f} | {r['fails_bar_c2']:.3f} | {r['fails_bar_both']:.3f} |")
            ps = ", ".join(f"{k} {v:.3f}" for k, v in res["amended"]["picked_share"].items())
            L.append(f"| {set_name} | CER after dialect bar (amended G1) | - | - | {ps} | - | - | - |")
        for set_name, res in block["rules"].items():
            for k in ("spearman_cer_c1", "spearman_cer_c2"):
                if k in res:
                    L.append(f"\n{set_name}: {k} = {res[k]['point']} [95% bootstrap {res[k]['lo']}, {res[k]['hi']}]")
        L.append("")
    L += ["## Egyptian share by the dialect of the reference clip (bake-off, one seed)", "",
          "References labelled Egyptian by both classifiers vs MSA by both (labels of the reference clip itself).", "",
          "| system | Egyptian refs: n, clf 1, clf 2, CER | MSA refs: n, clf 1, clf 2, CER |", "|---|---|---|"]
    for s, d in by_ref.items():
        a, b = d["ref_egyptian_both"], d["ref_msa_both"]
        L.append(f"| {s} | {a['n']}, {ci(a['c1'])}, {ci(a['c2'])}, {ci(a['cer_best'])} | {b['n']}, {ci(b['c1'])}, {ci(b['c2'])}, {ci(b['cer_best'])} |")
    with open(f"{RESULTS}/selection.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
