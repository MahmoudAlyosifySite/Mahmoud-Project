"""Aggregate per-utterance scores into BA/eval/results/bakeoff.json, bakeoff_utterances.csv and bakeoff.md,
and apply gate G1 mechanically.

Failure rule (fixed before any output was inspected):
  F1 missing   : synthesis raised an error or wrote no file
  F2 empty     : speech-active duration < 0.3 s or peak amplitude < 0.01
  F3 cut off   : Chatterbox-family output >= 39.9 s (the 1000-speech-token cap = 40 s)
  F4 duration  : characters per second of speech (normalised reference text without spaces /
                 speech-active duration) outside [0.5 x P5, 1.5 x P95] of real clips of that language
                 (arz: lahgtna_eg + niletts calibration clips; en: FLEURS en_us; fr: FLEURS fr_fr)
Headline Egyptian CER (used by G1) = mean CER over condition "egy" excluding the 8 digit texts
(32 texts x 3 voices = 96 utterances); failed utterances are included with whatever the ASR heard
(missing/empty ones count as CER 1.0).
G1 failure rate = failures / all utterances of the system's full matrix.
G1 speaker similarity = mean ECAPA cosine on the headline Egyptian set.
"""
import csv, json, os, sys, statistics as st
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matrix import build, OUT_ROOT, EVAL
from score import PROFILE, norm_ar, norm_latin, SCORES

RES = f"{EVAL}/results"
ELIGIBLE = ["cb-v3", "cb-v2", "voxcpm2"]
PARAMS_B = {"cb-v3": 0.5, "cb-v3-pypi": 0.5, "cb-v2": 0.5, "voxcpm2": 2.0, "voxcpm2-prompt": 2.0, "namaa": 0.5, "lahgtna": 0.5,
            "omnivoice": None, "voicetut": None}
LICENCE = {"cb-v3": "MIT", "cb-v3-pypi": "MIT (diagnostic variant)", "cb-v2": "MIT", "voxcpm2": "Apache-2.0", "voxcpm2-prompt": "Apache-2.0",
           "namaa": "MIT (baseline only)", "lahgtna": "MIT (baseline only)",
           "omnivoice": "CC-BY-NC (reference only)", "voicetut": "derived from CC-BY-NC (reference only)"}
CB_FAMILY = {"cb-v3", "cb-v3-pypi", "cb-v2", "namaa", "lahgtna"}


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def nchars(text, lang):
    t = norm_ar(text) if lang == "arz" else norm_latin(text, lang)
    return len(t.replace(" ", ""))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def median(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.median(xs)), 4) if xs else None


def main():
    calib = {r["calib_id"]: r for r in jl(f"{EVAL}/calib.jsonl")}
    cs = jl(f"{SCORES}/_calib.jsonl")
    # ---- calibration: char-rate bands
    cps = {"arz": [], "en": [], "fr": []}
    for r in cs:
        c = calib[r["key"]]
        if r["active_dur"] > 0:
            setlang = {"lahgtna_eg": "arz", "niletts": "arz", "fleurs_en_us": "en", "fleurs_fr_fr": "fr"}.get(c["set"])
            if setlang:
                cps[setlang].append(nchars(c["text"], c["lang"]) / r["active_dur"])
    band = {l: (round(0.5 * float(np.percentile(v, 5)), 3), round(1.5 * float(np.percentile(v, 95)), 3)) for l, v in cps.items()}
    calib_rows = []
    for s in ["lahgtna_eg", "niletts", "fleurs_ar_eg", "fleurs_en_us", "fleurs_fr_fr"]:
        rr = [r for r in cs if r["set"] == s]
        if not rr: continue
        d = [r.get("dialect") for r in rr if r.get("dialect")]
        calib_rows.append(dict(set=s, n=len(rr), cer_mean=mean([r.get("cer") for r in rr]), cer_median=median([r.get("cer") for r in rr]),
                               wer_mean=mean([r.get("wer") for r in rr]),
                               dialect_egyptian_share=round(d.count("Egyptian") / len(d), 4) if d else None,
                               dialect_msa_share=round(d.count("MSA") / len(d), 4) if d else None,
                               mos_p808=mean([r.get("mos_p808") for r in rr]), mos_ovrl=mean([r.get("mos_ovrl") for r in rr])))
    rp = json.load(open(f"{SCORES}/_refpairs.json"))
    spk_cal = {}
    for lang in ["ar_eg", "en_us", "fr_fr"]:
        pr = [p for p in rp["pairs"] if p["lang"] == lang]
        df = [d for d in rp["diff"] if d["same_lang"] and d["a"].startswith(lang)]
        spk_cal[lang] = dict(ceiling_ecapa=mean([p["ecapa"] for p in pr]), ceiling_wavlm=mean([p["wavlm"] for p in pr]),
                             floor_ecapa=mean([d["ecapa"] for d in df]), floor_wavlm=mean([d["wavlm"] for d in df]),
                             n_pairs=len(pr), n_diff=len(df))
    alld = rp["diff"]
    spk_cal["all_different_pairs"] = dict(floor_ecapa=mean([d["ecapa"] for d in alld]), floor_wavlm=mean([d["wavlm"] for d in alld]), n=len(alld))
    ref_rows = rp["refs"]

    # ---- per utterance
    systems = [s for s in PROFILE if os.path.exists(f"{OUT_ROOT}/{s}/meta.jsonl")]
    utt_rows, agg = [], {}
    for s in systems:
        meta = {r["utt_id"]: r for r in jl(f"{OUT_ROOT}/{s}/meta.jsonl")}
        sc = {r["key"]: r for r in jl(f"{SCORES}/{s}.jsonl")}
        rows = []
        for j in build(PROFILE[s]):
            m = meta.get(j["utt_id"]); r = sc.get(j["utt_id"])
            if m is None:
                continue  # not synthesised (run incomplete)
            row = dict(system=s, utt_id=j["utt_id"], condition=j["condition"], lang=j["lang"], text_id=j["text_id"],
                       category=j["category"], voice_id=j["voice_id"], seed=j["seed"], gen_sec=m["gen_sec"],
                       audio_sec=m["audio_sec"], rtf=m["rtf"], peak_vram_gb=m["peak_vram_gb"], error=m["error"])
            fail = None
            if r is None:
                row["scored"] = False
            else:
                row["scored"] = True
                for k in ["active_dur", "peak", "asr", "cer", "wer", "cer_vs_digit_text", "wer_vs_digit_text",
                          "sim_ecapa", "sim_wavlm", "dialect", "mos_p808", "mos_ovrl"]:
                    row[k] = r.get(k)
                if r.get("missing"):
                    fail = "missing"; row["cer"] = 1.0; row["wer"] = 1.0
                elif r["active_dur"] < 0.3 or r["peak"] < 0.01:
                    fail = "empty"
                elif s in CB_FAMILY and (m["audio_sec"] or 0) >= 39.9:
                    fail = "cutoff_cap"
                else:
                    c = nchars(j["text_spoken"] or j["text"], j["lang"]) / r["active_dur"]
                    row["chars_per_sec"] = round(c, 3)
                    lo, hi = band[j["lang"]]
                    if c < lo: fail = "too_slow"
                    elif c > hi: fail = "too_fast"
                if row.get("cer_vs_digit_text") is not None:
                    row["cer_digits_best"] = min(row["cer"], row["cer_vs_digit_text"])
                    row["wer_digits_best"] = min(row["wer"], row["wer_vs_digit_text"])
            row["fail"] = fail
            rows.append(row)
        utt_rows += rows
        # ---- aggregates
        def summ(rr):
            sc_ = [x for x in rr if x["scored"]]
            d = [x["dialect"] for x in sc_ if x.get("dialect")]
            return dict(n=len(rr), n_scored=len(sc_), n_fail=sum(1 for x in sc_ if x["fail"]),
                        fail_rate=round(sum(1 for x in sc_ if x["fail"]) / len(sc_), 4) if sc_ else None,
                        fail_kinds={k: sum(1 for x in sc_ if x["fail"] == k) for k in sorted({x["fail"] for x in sc_ if x["fail"]})},
                        cer_mean=mean([x.get("cer") for x in sc_]), cer_median=median([x.get("cer") for x in sc_]),
                        wer_mean=mean([x.get("wer") for x in sc_]), wer_median=median([x.get("wer") for x in sc_]),
                        sim_ecapa=mean([x.get("sim_ecapa") for x in sc_]), sim_wavlm=mean([x.get("sim_wavlm") for x in sc_]),
                        dialect_egyptian_share=round(d.count("Egyptian") / len(d), 4) if d else None,
                        dialect_msa_share=round(d.count("MSA") / len(d), 4) if d else None,
                        mos_p808=mean([x.get("mos_p808") for x in sc_]), mos_ovrl=mean([x.get("mos_ovrl") for x in sc_]))
        conds = {}
        egy = [x for x in rows if x["condition"] == "egy"]
        conds["egy_main"] = summ([x for x in egy if x["category"] != "digits"])
        dg = [x for x in egy if x["category"] == "digits"]
        conds["egy_digits"] = summ(dg)
        conds["egy_digits"]["cer_digits_best_mean"] = mean([x.get("cer_digits_best") for x in dg if x["scored"]])
        conds["egy_all40"] = summ(egy)
        for c in ["diac", "cs", "en", "fr", "cross_en", "cross_fr"]:
            rr = [x for x in rows if x["condition"] == c]
            if rr: conds[c] = summ(rr)
        for l in ["arz", "en", "fr"]:
            rr = [x for x in rows if x["condition"] == "long" and x["lang"] == l]
            if rr: conds[f"long_{l}"] = summ(rr)
        ok = [x for x in rows if not x["error"] and x["audio_sec"]]
        speed = dict(rtf_gpu_aggregate=round(sum(x["gen_sec"] for x in ok) / sum(x["audio_sec"] for x in ok), 4) if ok else None,
                     rtf_gpu_median=median([x["rtf"] for x in ok]),
                     peak_vram_gb=max([x["peak_vram_gb"] or 0 for x in rows]) if rows else None)
        iso = [x for x in jl(f"{OUT_ROOT}/{s}-speed/meta.jsonl") if not x["error"] and x["audio_sec"]]
        if iso:
            speed["rtf_gpu_isolated_24egy"] = round(sum(x["gen_sec"] for x in iso) / sum(x["audio_sec"] for x in iso), 4)
            speed["peak_vram_gb_isolated"] = max(x["peak_vram_gb"] or 0 for x in iso)
        cpu = jl(f"{OUT_ROOT}/cpu/{s}.json") if False else (json.load(open(f"{OUT_ROOT}/cpu/{s}.json")) if os.path.exists(f"{OUT_ROOT}/cpu/{s}.json") else None)
        if cpu:
            speed["rtf_cpu2_aggregate"] = round(sum(x["gen_sec"] for x in cpu) / sum(x["audio_sec"] for x in cpu), 3)
            speed["rtf_cpu2_items"] = cpu
        agg[s] = dict(licence=LICENCE[s], params_b=PARAMS_B[s], profile=PROFILE[s], overall=summ(rows),
                      egyptian_all=summ([x for x in rows if x["lang"] == "arz"]), conditions=conds, speed=speed)

    # ---- G1
    g1 = gate_g1(agg)
    out = dict(failure_band_chars_per_sec=band, calibration=calib_rows, speaker_calibration=spk_cal, reference_clips=ref_rows,
               systems=agg, gate_g1=g1,
               commands=dict(synth="BA/eval/README.md", score="score.py", aggregate="aggregate.py"),
               seeds="seed per utterance = crc32(utt_id) mod 2^31 (matrix.py)")
    json.dump(out, open(f"{RES}/bakeoff.json", "w"), indent=1, ensure_ascii=False)
    keys = []
    for r in utt_rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(f"{RES}/bakeoff_utterances.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(utt_rows)
    write_md(out)


def gate_g1(agg):
    lines, cand = [], []
    for s in ELIGIBLE:
        if s not in agg:
            lines.append(f"{s}: not run -> not a candidate"); continue
        fr = agg[s]["overall"]["fail_rate"]; c = agg[s]["conditions"]["egy_main"]
        ok = fr is not None and fr <= 0.05
        lines.append(f"{s}: failure rate {fr} ({agg[s]['overall']['n_fail']}/{agg[s]['overall']['n_scored']}) -> {'passes' if ok else 'FAILS'} the <= 0.05 filter")
        if ok: cand.append(s)
    if not cand:
        lines.append("No eligible system passes the failure-rate filter: G1 selects nothing.")
        return dict(selected=None, steps=lines)
    cand.sort(key=lambda s: agg[s]["conditions"]["egy_main"]["cer_mean"])
    best = cand[0]; bc = agg[best]["conditions"]["egy_main"]["cer_mean"]
    for s in cand:
        lines.append(f"{s}: Egyptian CER (mean, 96 utt) = {agg[s]['conditions']['egy_main']['cer_mean']}")
    lines.append(f"Lowest Egyptian CER: {best} ({bc})")
    tied = [best]
    for s in cand[1:]:
        c = agg[s]["conditions"]["egy_main"]["cer_mean"]; rel = (c - bc) / bc if bc else float("inf")
        lines.append(f"{s}: relative gap ({c} - {bc}) / {bc} = {rel:.4f} -> {'within' if rel <= 0.10 else 'outside'} 10%")
        if rel <= 0.10: tied.append(s)
    if len(tied) == 1:
        lines.append(f"G1 selects {best}.")
        return dict(selected=best, steps=lines)
    tied.sort(key=lambda s: -agg[s]["conditions"]["egy_main"]["sim_ecapa"])
    top = tied[0]; ts = agg[top]["conditions"]["egy_main"]["sim_ecapa"]
    for s in tied:
        lines.append(f"{s}: speaker similarity (ECAPA, Egyptian main set) = {agg[s]['conditions']['egy_main']['sim_ecapa']}")
    tied2 = [top] + [s for s in tied[1:] if ts - agg[s]["conditions"]["egy_main"]["sim_ecapa"] <= 0.02]
    for s in tied[1:]:
        d = ts - agg[s]["conditions"]["egy_main"]["sim_ecapa"]
        lines.append(f"{s}: similarity gap {ts} - {agg[s]['conditions']['egy_main']['sim_ecapa']} = {d:.4f} -> {'within' if d <= 0.02 else 'outside'} 0.02")
    if len(tied2) == 1:
        lines.append(f"G1 selects {top}.")
        return dict(selected=top, steps=lines)
    tied2.sort(key=lambda s: (agg[s]["params_b"], agg[s]["speed"].get("rtf_gpu_isolated_24egy", agg[s]["speed"]["rtf_gpu_aggregate"])))
    for s in tied2:
        lines.append(f"{s}: {agg[s]['params_b']}B params, GPU RTF {agg[s]['speed'].get('rtf_gpu_isolated_24egy', agg[s]['speed']['rtf_gpu_aggregate'])}")
    lines.append(f"Smaller/faster: G1 selects {tied2[0]}.")
    return dict(selected=tied2[0], steps=lines)


def f(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def write_md(o):
    A = o["systems"]; L = []
    L.append("# BA Phase 1 bake-off: results\n")
    L.append("Generated by `BA/eval/scripts/aggregate.py` from `bakeoff.json` / `bakeoff_utterances.csv`. "
             "Numbers only in sections 1-4; interpretation is in section 5.\n")
    L.append("## 1. Headline (Egyptian main set: 32 non-digit texts x 3 voices)\n")
    L.append("| system | licence | n | fail (all conds) | Egy CER mean | Egy CER median | sim ECAPA | sim WavLM | dialect=Egyptian | dialect=MSA | DNSMOS P.808 | GPU RTF (full run) | GPU RTF (isolated, 24 egy) | peak VRAM GB | CPU-2thr RTF |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s, a in A.items():
        c = a["conditions"]["egy_main"]; sp = a["speed"]
        L.append(f"| {s} | {a['licence']} | {c['n_scored']} | {f(a['overall']['fail_rate'])} ({a['overall']['n_fail']}/{a['overall']['n_scored']}) | {f(c['cer_mean'])} | {f(c['cer_median'])} | {f(c['sim_ecapa'])} | {f(c['sim_wavlm'])} | {f(c['dialect_egyptian_share'])} | {f(c['dialect_msa_share'])} | {f(c['mos_p808'])} | {f(sp['rtf_gpu_aggregate'])} | {f(sp.get('rtf_gpu_isolated_24egy'))} | {f(sp['peak_vram_gb'],2)} | {f(sp.get('rtf_cpu2_aggregate'))} |")
    L.append("\nRTF = generation seconds / audio seconds (sum over utterances, reference encoding included, warm-up excluded). The full runs of cb-v2 and cb-v3-pypi overlapped with CPU benchmarks and downloads, so the isolated re-run of the first 24 Egyptian jobs is the comparable speed number. CPU: `taskset -c 0,1`, `torch.set_num_threads(2)`, 3 short sentences.\n")
    L.append("## 2. All conditions\n")
    L.append("CER for Egyptian-text conditions (egy_*, diac, cs, long_arz); WER for en/fr/cross_*/long_en/long_fr.\n")
    L.append("| system | condition | n | fail | CER mean | WER mean | WER median | sim ECAPA | sim WavLM | dialect=Egy | DNSMOS P.808 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for s, a in A.items():
        for cn, c in a["conditions"].items():
            extra = f" (best-of spelled/digit ref CER {f(c.get('cer_digits_best_mean'))})" if cn == "egy_digits" else ""
            L.append(f"| {s} | {cn}{extra} | {c['n_scored']} | {c['n_fail']} {c['fail_kinds'] or ''} | {f(c['cer_mean'])} | {f(c['wer_mean'])} | {f(c['wer_median'])} | {f(c['sim_ecapa'])} | {f(c['sim_wavlm'])} | {f(c['dialect_egyptian_share'])} | {f(c['mos_p808'])} |")
    L.append("\n## 3. Calibration (real speech)\n")
    L.append("| set | n | ASR CER mean | CER median | WER mean | dialect=Egyptian | dialect=MSA | DNSMOS P.808 | DNSMOS OVRL |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in o["calibration"]:
        L.append(f"| {r['set']} | {r['n']} | {f(r['cer_mean'])} | {f(r['cer_median'])} | {f(r['wer_mean'])} | {f(r['dialect_egyptian_share'])} | {f(r['dialect_msa_share'])} | {f(r['mos_p808'])} | {f(r['mos_ovrl'])} |")
    L.append("\nSpeaker similarity calibration (FLEURS reference vs a second clip of the same speaker = ceiling; reference vs other chosen speakers of the same language = floor):\n")
    L.append("| language | ceiling ECAPA | ceiling WavLM | floor ECAPA | floor WavLM |")
    L.append("|---|---|---|---|---|")
    for l, r in o["speaker_calibration"].items():
        L.append(f"| {l} | {f(r.get('ceiling_ecapa'))} | {f(r.get('ceiling_wavlm'))} | {f(r.get('floor_ecapa'))} | {f(r.get('floor_wavlm'))} |")
    L.append(f"\nFailure band (chars/s of speech, from real clips): {o['failure_band_chars_per_sec']}\n")
    L.append("Reference clips (DNSMOS, dialect label of the six ar_eg FLEURS clips):\n")
    L.append("| voice | DNSMOS P.808 | dialect |")
    L.append("|---|---|---|")
    for r in o["reference_clips"]:
        L.append(f"| {r['voice_id']} | {f(r['mos_p808'])} | {r.get('dialect','-')} |")
    L.append("\n## 4. Gate G1 (applied mechanically)\n")
    L.append("Rule (execution-plan.md): among licence-eligible systems with failure rate <= 5%: lowest Egyptian CER wins; "
             "within 10% relative -> higher speaker similarity; within 0.02 -> smaller and faster.\n")
    for l in o["gate_g1"]["steps"]:
        L.append(f"- {l}")
    L.append(f"\n**Selected: {o['gate_g1']['selected']}**\n")
    obs = f"{RES}/observations.md"
    if os.path.exists(obs):
        L.append("## 5. Observations and caveats (interpretation)\n")
        L.append(open(obs).read())
    open(f"{RES}/bakeoff.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
