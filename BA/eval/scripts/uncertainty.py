"""Experiment 2: seeds and uncertainty (venv /tmp/ba/venvs/eval, CPU). -> BA/eval/results/uncertainty.json, .md

A. Three seeds. Egyptian held-out main (24 texts x 3 voices = 72) and digit (6 x 3 = 18) sentences for BA, raw
   VoxCPM2, Chatterbox V3 (and NAMAA when its run exists). Seed index 0 = the existing run (engine/full for BA,
   the Phase 1 bake-off for the others); indices 1 and 2 = synth_paper.py seeds_* (seed = crc32("<utt_id>#s<k>")).
   Per system and condition: the mean of each seed with its two-way bootstrap CI, the SD across seed means, and
   the 3-seed mean with a two-way bootstrap CI (texts and voices resampled, a cell's seeds kept together).
   Paired tests on the shared (text, voice) cells, each system's cell value averaged over its seeds:
   BA - raw VoxCPM2, VoxCPM2 - Chatterbox V3 (the dissociation) and BA - Chatterbox V3. Holm adjustment over the
   six primary tests (egy_main: BA-raw CER, ECAPA, share clf 1; VoxCPM2-CBV3 CER, share clf 1, share clf 2).
B. One-seed bake-off headline (Egyptian main set, 32 texts x 3 voices; digit set 8 x 3) with two-way bootstrap CIs
   for all nine systems.
C. BA vs raw VoxCPM2, one seed, every held-out condition of engine.md, with CIs and paired tests.

Metrics: CER (cer_best: best of spelled/digit reference, = CER on non-digit text) or WER for English/French,
ECAPA and WavLM similarity, Egyptian share under classifier 1 (stored labels) and classifier 2
(dialect_classifier2.py cache), DNSMOS P.808, failure share (Phase 1 rule). Bootstrap: paper_common.py.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from paper_common import (ENGINE_ROOT, RESULTS, TUNE_IDS, jl, bakeoff_rows, engine_rows, held_out, attach_dialect, add_canonical_cer,
                          two_way_bootstrap, paired, holm, fmt, ci, dump)

SEEDED = {"ba": ("engine", "full", "p_seeds_ba"), "voxcpm2": ("bakeoff", "voxcpm2", "p_seeds_voxcpm2"),
          "cb-v3": ("bakeoff", "cb-v3", "p_seeds_cbv3"), "namaa": ("bakeoff", "namaa", "p_seeds_namaa")}
METRICS = [("cer_best", "CER"), ("sim_ecapa", "ECAPA"), ("sim_wavlm", "WavLM"), ("c1", "Egyptian share clf 1"),
           ("c2", "Egyptian share clf 2"), ("mos_p808", "DNSMOS"), ("failed", "failure share")]
BINARY = {"c1", "c2", "failed"}
BAKEOFF = ["cb-v3", "cb-v3-pypi", "cb-v2", "voxcpm2", "namaa", "lahgtna", "omnivoice", "voicetut", "voxcpm2-prompt"]
CONDS = ["egy_main", "egy_digits"]


def seeded_rows(system):
    kind, base, exp = SEEDED[system]
    r0 = engine_rows(base, "ba") if kind == "engine" else bakeoff_rows(base)
    rows = [r for r in held_out(r0) if r["cond"] in CONDS]
    if os.path.exists(f"{ENGINE_ROOT}/{exp}/scores.jsonl"):
        rows += [r for r in engine_rows(exp, system) if r["cond"] in CONDS]
    return attach_dialect(rows)


def boot(rows, key):
    return two_way_bootstrap([r.get(key) for r in rows], [r["text_id"] for r in rows], [r["voice_id"] for r in rows])


def seed_summary(rows):
    out = {}
    seeds = sorted({r["seed_index"] for r in rows})
    for key, _ in METRICS:
        per = {k: boot([r for r in rows if r["seed_index"] == k], key) for k in seeds}
        means = [p["mean"] for p in per.values() if p["mean"] is not None]
        out[key] = dict(three_seed=boot(rows, key), per_seed={str(k): v for k, v in per.items()},
                        sd_across_seeds=round(float(np.std(means, ddof=1)), 4) if len(means) > 1 else None)
    return dict(n_rows=len(rows), seeds=seeds, n_per_seed={str(k): sum(1 for r in rows if r["seed_index"] == k) for k in seeds}, metrics=out)


def main():
    out = dict(design=__doc__.split("Metrics:")[0].strip(), held_out_rule="tuning texts excluded: " + " ".join(sorted(TUNE_IDS)))
    # ---------------- A
    rows = {s: seeded_rows(s) for s in SEEDED}
    rows = {s: r for s, r in rows.items() if r}
    A = {}
    for s, rr in rows.items():
        A[s] = {c: seed_summary([r for r in rr if r["cond"] == c]) for c in CONDS}
    out["three_seeds"] = A
    comps = [("ba", "voxcpm2"), ("voxcpm2", "cb-v3"), ("ba", "cb-v3")]
    if "namaa" in rows and len({r["seed_index"] for r in rows["namaa"]}) > 1:
        comps.append(("voxcpm2", "namaa"))
    P = {}
    for a, b in comps:
        if a not in rows or b not in rows:
            continue
        for c in CONDS:
            ra = [r for r in rows[a] if r["cond"] == c]
            rb = [r for r in rows[b] if r["cond"] == c]
            for key, _ in METRICS:
                res = paired(ra, rb, key, binary=key in BINARY)
                if res:
                    res["seeds_a"] = sorted({r["seed_index"] for r in ra})
                    res["seeds_b"] = sorted({r["seed_index"] for r in rb})
                    P[f"{a}_minus_{b}|{c}|{key}"] = res
    prim = ["ba_minus_voxcpm2|egy_main|cer_best", "ba_minus_voxcpm2|egy_main|sim_ecapa", "ba_minus_voxcpm2|egy_main|c1",
            "voxcpm2_minus_cb-v3|egy_main|cer_best", "voxcpm2_minus_cb-v3|egy_main|c1", "voxcpm2_minus_cb-v3|egy_main|c2"]
    adj = holm({k: P[k]["p_bootstrap"] for k in prim if k in P})
    for k, v in adj.items():
        P[k]["p_bootstrap_holm_primary6"] = v
    out["paired_three_seeds"] = P
    out["primary_tests"] = prim
    # ---------------- B
    B = {}
    for s in BAKEOFF:
        rr = attach_dialect([r for r in bakeoff_rows(s) if r["cond"] in CONDS])
        B[s] = {c: {key: boot([r for r in rr if r["cond"] == c], key) for key, _ in METRICS} for c in CONDS}
        B[s]["egy_main"]["cer_spelled_ref"] = boot([r for r in rr if r["cond"] == "egy_main"], "cer")
    out["bakeoff_one_seed"] = B
    # ---------------- C
    from engine_report import COND_ORDER
    ba = add_canonical_cer(attach_dialect(held_out(engine_rows("full", "ba"))))
    raw = add_canonical_cer(attach_dialect(held_out(bakeoff_rows("voxcpm2"))))
    C = {}
    for c in COND_ORDER:
        b_, r_ = [r for r in ba if r["cond"] == c], [r for r in raw if r["cond"] == c]
        if not b_:
            continue
        keys = ["err", "sim_ecapa", "sim_wavlm", "mos_p808", "failed"] + (["c1", "c2"] if b_[0]["lang"] == "arz" and c != "long_arz" else []) \
            + (["cer_canonical"] if c == "cs" else [])
        C[c] = dict(n=len(b_), metric_err="CER" if b_[0]["lang"] == "arz" else "WER",
                    ba={k: boot(b_, k) for k in keys}, raw={k: boot(r_, k) for k in keys},
                    paired={k: paired(b_, r_, k, binary=k in BINARY) for k in keys} if c not in ("long_arz", "long_en", "long_fr") else None)
    out["engine_vs_raw_one_seed"] = C
    out["command"] = "/tmp/ba/venvs/eval/bin/python BA/eval/scripts/uncertainty.py"
    dump(out, f"{RESULTS}/uncertainty.json")
    write_md(out)


def pline(name, p):
    return (f"| {name} | {p['n_cells']} | {fmt(p['mean_a'])} | {fmt(p['mean_b'])} | {fmt(p['diff'])} [{fmt(p['ci_lo'])}, {fmt(p['ci_hi'])}] | "
            f"{p['p_bootstrap']} | {p.get('p_bootstrap_holm_primary6', '-')} | {p['p_wilcoxon']} | {p['d_z']} | {p['rank_biserial']} | {p.get('cohens_h', '-')} |")


def write_md(o):
    L = ["# Seeds and uncertainty\n",
         "Generated by `BA/eval/scripts/uncertainty.py` from `uncertainty.json`. Intervals: 95% two-way bootstrap (texts and voices "
         "resampled independently, 10,000 resamples; a cell's seeds stay together). Held-out texts only for sections A and C.\n",
         "## A. Three seeds (Egyptian held-out main: 24 texts x 3 voices; digits: 6 x 3)\n",
         "| system | condition | seeds (n each) | CER [CI] | ECAPA [CI] | Egyptian share clf 1 [CI] | Egyptian share clf 2 [CI] | DNSMOS | failures | SD of seed means (CER / ECAPA / clf 1) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for s, d in o["three_seeds"].items():
        for c, v in d.items():
            m = v["metrics"]
            L.append(f"| {s} | {c} | {v['n_per_seed']} | {ci(m['cer_best']['three_seed'])} | {ci(m['sim_ecapa']['three_seed'])} | "
                     f"{ci(m['c1']['three_seed'])} | {ci(m['c2']['three_seed'])} | {fmt(m['mos_p808']['three_seed']['mean'])} | "
                     f"{fmt(m['failed']['three_seed']['mean'])} | {fmt(m['cer_best']['sd_across_seeds'])} / {fmt(m['sim_ecapa']['sd_across_seeds'])} / {fmt(m['c1']['sd_across_seeds'])} |")
    L.append("\nPer-seed means (CER / ECAPA / clf 1 share):\n")
    L.append("| system | condition | seed 0 | seed 1 | seed 2 |\n|---|---|---|---|---|")
    for s, d in o["three_seeds"].items():
        for c, v in d.items():
            m = v["metrics"]
            cells = []
            for k in ("0", "1", "2"):
                if k in m["cer_best"]["per_seed"]:
                    cells.append(f"{fmt(m['cer_best']['per_seed'][k]['mean'])} / {fmt(m['sim_ecapa']['per_seed'][k]['mean'])} / {fmt(m['c1']['per_seed'][k]['mean'])}")
                else:
                    cells.append("-")
            L.append(f"| {s} | {c} | " + " | ".join(cells) + " |")
    L.append("\n### Paired tests (A minus B on shared text-voice cells, 3 seeds averaged per cell)\n")
    L.append("Bootstrap p = two-sided two-way bootstrap; Holm = adjusted over the six primary tests; Wilcoxon on cell differences "
             "(ignores text/voice clustering); d_z = mean difference / SD of differences; r = matched-pairs rank-biserial; h = Cohen's h (shares).\n")
    L.append("| comparison | n cells | A | B | A - B [95% CI] | p boot | p Holm | p Wilcoxon | d_z | r | h |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, p in o["paired_three_seeds"].items():
        if k.split("|")[2] in ("sim_wavlm",):
            continue
        L.append(pline(k, p))
    L.append("\n## B. One-seed bake-off with CIs (Egyptian main set: 32 texts x 3 voices; digits 8 x 3)\n")
    L.append("| system | CER [CI] | ECAPA [CI] | WavLM [CI] | Egyptian clf 1 [CI] | Egyptian clf 2 [CI] | DNSMOS [CI] | digits CER (best ref) [CI] |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s, d in o["bakeoff_one_seed"].items():
        m = d["egy_main"]
        L.append(f"| {s} | {ci(m['cer_best'])} | {ci(m['sim_ecapa'])} | {ci(m['sim_wavlm'])} | {ci(m['c1'])} | {ci(m['c2'])} | {ci(m['mos_p808'])} | {ci(d['egy_digits']['cer_best'])} |")
    L.append("\n## C. BA vs raw VoxCPM2, one seed, all held-out conditions\n")
    L.append("| condition | n | metric | BA [CI] | raw [CI] | BA - raw [CI] | p boot | p Wilcoxon |\n|---|---|---|---|---|---|---|---|")
    for c, v in o["engine_vs_raw_one_seed"].items():
        for k in v["ba"]:
            name = v["metric_err"] if k == "err" else k
            p = (v["paired"] or {}).get(k)
            L.append(f"| {c} | {v['n']} | {name} | {ci(v['ba'][k])} | {ci(v['raw'][k])} | " +
                     (f"{fmt(p['diff'])} [{fmt(p['ci_lo'])}, {fmt(p['ci_hi'])}] | {p['p_bootstrap']} | {p['p_wilcoxon']} |" if p else "- | - | - |"))
    obs = f"{RESULTS}/uncertainty_observations.md"
    if os.path.exists(obs):
        L.append("\n## D. Interpretation\n")
        L.append(open(obs).read())
    open(f"{RESULTS}/uncertainty.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
