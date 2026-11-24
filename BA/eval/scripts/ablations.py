"""Experiment 3: ablations of the BA engine (venv /tmp/ba/venvs/eval, CPU). -> BA/eval/results/ablations.json, .md

Variants (synth_paper.py documents how each is made), all on the held-out Egyptian main (72), digit (18) and
code-switch (18) utterances and held-out English (42), same texts, voices and seeds (seed index 0):
  full         BA as shipped (engine/full)
  no_normaliser, no_loanwords, no_transcript (reference-only), raw_reference (no reference preparation)
  raw_voxcpm2  the Phase 1 bake-off VoxCPM2 output (plain reference cloning, no engine at all)
The chunk check (per-chunk duration check and retries) is not ablated: it never triggered in engine/full on these
sets (retries counted below), so removing it would give the same outputs.

For each variant and condition: two-way bootstrap CIs (paper_common.py) of CER (Egyptian text; digits: best of
spelled/digit reference) or WER (English), canonical-script CER for code-switch, ECAPA/WavLM similarity, Egyptian
share under both classifiers, DNSMOS and failure share; paired differences variant - full with CIs and p-values.
"Share of the BA gain" = (full - variant) / (full - raw_voxcpm2) on the same utterances: descriptive only.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from paper_common import (ENGINE_ROOT, RESULTS, jl, bakeoff_rows, engine_rows, held_out, attach_dialect, add_canonical_cer,
                          two_way_bootstrap, paired, fmt, ci, dump)

VARIANTS = {"full": "full", "no_normaliser": "p_abl_nonorm", "no_loanwords": "p_abl_noloan",
            "no_transcript": "p_abl_notrans", "raw_reference": "p_abl_rawref"}
CONDS = ["egy_main", "egy_digits", "cs", "en"]
BINARY = {"c1", "c2", "failed"}


def keys_for(cond):
    k = ["err", "sim_ecapa", "sim_wavlm", "mos_p808", "failed"]
    if cond != "en":
        k += ["c1", "c2"]
    if cond == "cs":
        k += ["cer_canonical"]
    return k


def load_variant(name):
    if name == "raw_voxcpm2":
        rows = bakeoff_rows("voxcpm2")
    else:
        exp = VARIANTS[name]
        if not os.path.exists(f"{ENGINE_ROOT}/{exp}/scores.jsonl"):
            return []
        rows = engine_rows(exp, name)
    rows = [r for r in held_out(rows) if r["cond"] in CONDS]
    return add_canonical_cer(attach_dialect(rows))


def boot(rows, key):
    return two_way_bootstrap([r.get(key) for r in rows], [r["text_id"] for r in rows], [r["voice_id"] for r in rows])


def main():
    names = list(VARIANTS) + ["raw_voxcpm2"]
    R = {n: load_variant(n) for n in names}
    R = {n: r for n, r in R.items() if r}
    out = dict(design=__doc__.strip(), variants={}, paired_vs_full={}, share_of_gain={})
    meta_full = {m["utt_id"]: m for m in jl(f"{ENGINE_ROOT}/full/meta.jsonl")}
    for n, rows in R.items():
        v = {}
        for c in CONDS:
            rr = [r for r in rows if r["cond"] == c]
            if not rr:
                continue
            v[c] = dict(n=len(rr), metrics={k: boot(rr, k) for k in keys_for(c)},
                        retries=sum(r.get("n_retries") or 0 for r in rr) if n != "raw_voxcpm2" else None,
                        reused_from_full=sum(1 for r in rr if r.get("reused_from_full")) if n not in ("full", "raw_voxcpm2") else None)
        out["variants"][n] = v
    full = R["full"]
    for n, rows in R.items():
        if n == "full":
            continue
        out["paired_vs_full"][n] = {}
        for c in CONDS:
            a = [r for r in rows if r["cond"] == c]
            b = [r for r in full if r["cond"] == c]
            if a:
                out["paired_vs_full"][n][c] = {k: paired(a, b, k, binary=k in BINARY) for k in keys_for(c)}
    if "raw_voxcpm2" in R:
        for n in R:
            if n in ("full", "raw_voxcpm2"):
                continue
            out["share_of_gain"][n] = {}
            for c in CONDS:
                pv, pr = out["paired_vs_full"][n].get(c), out["paired_vs_full"]["raw_voxcpm2"].get(c)
                if not pv or not pr:
                    continue
                sh = {}
                for k in keys_for(c):
                    gain = -pr[k]["diff"] if pr.get(k) else None      # full - raw
                    loss = -pv[k]["diff"] if pv.get(k) else None      # full - variant
                    sh[k] = round(loss / gain, 3) if gain not in (None, 0) and loss is not None and abs(gain) > 1e-9 else None
                out["share_of_gain"][n][c] = sh
    held = [m for m in meta_full.values() if m["utt_id"] in {r["utt_id"] for r in full}]
    out["chunk_check_full"] = dict(utterances=len(held), with_retry=sum(1 for m in held if (m.get("n_retries") or 0) > 0),
                                   retries=sum(m.get("n_retries") or 0 for m in held),
                                   chunks_failed=sum(m.get("chunks_failed") or 0 for m in held))
    vinfo = {}
    for n, exp in VARIANTS.items():
        vv = jl(f"{ENGINE_ROOT}/{exp}/voices.jsonl") if exp != "full" else []
        if vv:
            vinfo[n] = [dict(voice_id=x["voice_id"], mode=x.get("mode"), seconds=x.get("seconds"), transcript=x.get("transcript")) for x in vv]
    out["voices"] = vinfo
    out["command"] = "/tmp/ba/venvs/eval/bin/python BA/eval/scripts/ablations.py"
    dump(out, f"{RESULTS}/ablations.json")
    write_md(out)


LABEL = {"err": "CER/WER", "cer_canonical": "CER canonical", "sim_ecapa": "ECAPA", "sim_wavlm": "WavLM", "c1": "Egy clf 1",
         "c2": "Egy clf 2", "mos_p808": "DNSMOS", "failed": "fail"}


def write_md(o):
    L = ["# Ablations of the BA engine\n",
         "Generated by `BA/eval/scripts/ablations.py` from `ablations.json`. Held-out texts, seed index 0, same utterances in every "
         "variant. Intervals: 95% two-way bootstrap (texts and voices). Error = CER for Egyptian text (digits: best of spelled/digit "
         "reference), WER for English.\n"]
    order = [n for n in ["full", "no_normaliser", "no_loanwords", "no_transcript", "raw_reference", "raw_voxcpm2"] if n in o["variants"]]
    for c in CONDS:
        ks = keys_for(c)
        L.append(f"## {c}\n")
        L.append("| variant | n | " + " | ".join(LABEL[k] for k in ks) + " | reused from full |")
        L.append("|---|---|" + "---|" * len(ks) + "---|")
        for n in order:
            v = o["variants"][n].get(c)
            if v:
                L.append(f"| {n} | {v['n']} | " + " | ".join(ci(v["metrics"][k]) for k in ks) + f" | {v['reused_from_full'] if v['reused_from_full'] is not None else '-'} |")
        L.append("\nPaired difference variant - full [95% CI] (bootstrap p):\n")
        L.append("| variant | " + " | ".join(LABEL[k] for k in ks) + " |")
        L.append("|---|" + "---|" * len(ks))
        for n in order[1:]:
            p = o["paired_vs_full"].get(n, {}).get(c)
            if p:
                L.append(f"| {n} | " + " | ".join(f"{fmt(p[k]['diff'])} [{fmt(p[k]['ci_lo'])}, {fmt(p[k]['ci_hi'])}] (p {p[k]['p_bootstrap']})" if p.get(k) else "-" for k in ks) + " |")
        L.append("")
    if o["share_of_gain"]:
        L.append("## Share of the BA - raw gain lost when a component is removed (descriptive)\n")
        L.append("(full - variant) / (full - raw VoxCPM2) on the same utterances; 1 = the whole gain, 0 = none, negative = removing it helps.\n")
        L.append("| variant | condition | CER/WER | ECAPA | Egy clf 1 | Egy clf 2 |\n|---|---|---|---|---|---|")
        for n, d in o["share_of_gain"].items():
            for c, s in d.items():
                L.append(f"| {n} | {c} | {fmt(s.get('err'))} | {fmt(s.get('sim_ecapa'))} | {fmt(s.get('c1'))} | {fmt(s.get('c2'))} |")
    cc = o["chunk_check_full"]
    L.append(f"\nChunk check in full BA on these utterances: {cc['with_retry']} of {cc['utterances']} needed a retry ({cc['retries']} retries); "
             f"{cc['chunks_failed']} chunks failed after retries, so an ablation of the check would reproduce full BA here.\n")
    obs = f"{RESULTS}/ablations_observations.md"
    if os.path.exists(obs):
        L.append("## Interpretation\n")
        L.append(open(obs).read())
    open(f"{RESULTS}/ablations.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
