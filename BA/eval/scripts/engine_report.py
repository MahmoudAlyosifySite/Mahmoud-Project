"""BA engine vs raw VoxCPM2, and gate G3 (venv /tmp/ba/venvs/eval).

Usage: python engine_report.py   -> BA/eval/results/engine.json, engine.md

Inputs: the BA engine run (synth_ba.py full + score_manifest.py full), the Phase 1 bake-off
utterances (results/bakeoff_utterances.csv: systems voxcpm2 and voxcpm2-prompt), the tuning
decisions (sweep.json, chunk_length.json, codeswitch_policy.json, reference_thresholds.json),
asr_check.json, watermark.json and bench.json.

Held-out rule: tuning texts (engine_common.TUNE_*) are excluded from every number in the
comparison and from G3. Both systems are summarised on exactly the same utterances
(same texts, voices and seeds), with the Phase 1 failure rule and metrics.

G3 as written in execution-plan.md, applied per language:
  failure rate <= 5%      (all held-out utterances whose output language is that language:
                           same-language, cross-language and passages)
  intelligibility         Egyptian: CER on the Egyptian main set <= ASR CER on real colloquial
                           Egyptian (lahgtna_eg calibration) + 0.10; English WER <= 0.08; French WER <= 0.10
                           (same-language sentence sets)
  similarity              mean ECAPA similarity on the same-language set >= 0.8 x the real
                           same-speaker ceiling of that language (bake-off speaker calibration)
  Egyptian dialect        dialect-ID share labelled Egyptian on the Egyptian main set >= 0.6 x the
                           share on real colloquial Egyptian speech (lahgtna_eg)
Cross-language similarity is reported against the Egyptian ceiling as supplementary information;
the plan's G3 text does not say which conditions enter the similarity bound.
"""
import csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from engine_common import RESULTS, TUNE_IDS, jl  # noqa: E402
from analyze_engine import load, mean, canonical  # noqa: E402

COND_ORDER = ["egy_main", "egy_digits", "diac", "cs", "en", "fr", "cross_en", "cross_fr", "long_arz", "long_en", "long_fr"]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def bakeoff_rows(system):
    rows = []
    for r in csv.DictReader(open(f"{RESULTS}/bakeoff_utterances.csv", encoding="utf-8")):
        if r["system"] != system or r["scored"] != "True":
            continue
        d = dict(r)
        for k in ("cer", "wer", "cer_vs_digit_text", "wer_vs_digit_text", "sim_ecapa", "sim_wavlm", "mos_p808",
                  "gen_sec", "audio_sec", "rtf", "cer_digits_best", "wer_digits_best"):
            d[k] = fnum(r.get(k))
        d["fail"] = r["fail"] or None
        d["dialect"] = r["dialect"] or None
        d["cer_best"] = d["cer_digits_best"] if d["cer_digits_best"] is not None else d["cer"]
        d["wer_best"] = d["wer_digits_best"] if d["wer_digits_best"] is not None else d["wer"]
        d["err"] = d["cer_best"] if d["lang"] == "arz" else d["wer_best"]
        rows.append(d)
    return rows


def cond_of(r):
    c = r["condition"]
    if c == "egy":
        return "egy_digits" if r["category"] == "digits" else "egy_main"
    if c == "long":
        return f"long_{r['lang']}"
    return c


def summarise(rows):
    d = [r["dialect"] for r in rows if r.get("dialect")]
    fk = {}
    for r in rows:
        if r["fail"]:
            fk[r["fail"]] = fk.get(r["fail"], 0) + 1
    return dict(n=len(rows), n_fail=sum(1 for r in rows if r["fail"]), fail_kinds=fk,
                cer=mean([r["cer_best"] for r in rows]), wer=mean([r["wer_best"] for r in rows]),
                cer_spelled_ref=mean([r["cer"] for r in rows]),
                sim_ecapa=mean([r["sim_ecapa"] for r in rows]), sim_wavlm=mean([r["sim_wavlm"] for r in rows]),
                dialect_egyptian=round(d.count("Egyptian") / len(d), 4) if d else None,
                mos_p808=mean([r["mos_p808"] for r in rows]))


def table(rows, held_out_only=True):
    out = {}
    for c in COND_ORDER:
        rr = [r for r in rows if cond_of(r) == c and (not held_out_only or r["text_id"] not in TUNE_IDS)]
        if rr:
            out[c] = summarise(rr)
            if c == "cs":
                import jiwer
                from engine_common import _text_map
                T = _text_map()
                cc = []
                for r in rr:
                    hyp = canonical(r.get("asr") or "")
                    cc.append(float(jiwer.cer(canonical(T[r["text_id"]]["text"]), hyp)) if hyp else 1.0)
                out[c]["cer_canonical"] = mean(cc)
    return out


def g3(ba, cal, spk):
    lahg = next(c for c in cal if c["set"] == "lahgtna_eg")
    res = {}
    spec = {"arz": dict(fail_conds=["egy_main", "egy_digits", "diac", "cs", "long_arz"], err=("egy_main", "cer"),
                        bound=round(lahg["cer_mean"] + 0.10, 4), ceil=spk["ar_eg"]["ceiling_ecapa"], sim_cond="egy_main"),
            "en": dict(fail_conds=["en", "cross_en", "long_en"], err=("en", "wer"), bound=0.08,
                       ceil=spk["en_us"]["ceiling_ecapa"], sim_cond="en"),
            "fr": dict(fail_conds=["fr", "cross_fr", "long_fr"], err=("fr", "wer"), bound=0.10,
                       ceil=spk["fr_fr"]["ceiling_ecapa"], sim_cond="fr")}
    for lang, s in spec.items():
        n = sum(ba[c]["n"] for c in s["fail_conds"] if c in ba)
        nf = sum(ba[c]["n_fail"] for c in s["fail_conds"] if c in ba)
        fr = nf / n if n else None
        err = ba[s["err"][0]][s["err"][1]]
        sim = ba[s["sim_cond"]]["sim_ecapa"]
        sim_bar = round(0.8 * s["ceil"], 4)
        checks = {
            "failure_rate": dict(value=round(fr, 4), arithmetic=f"{nf}/{n} = {fr:.4f} <= 0.05", passed=fr <= 0.05),
            "intelligibility": dict(value=err, arithmetic=f"{s['err'][1].upper()} {err} <= {s['bound']}" +
                                    (f" (= {lahg['cer_mean']} + 0.10)" if lang == "arz" else ""), passed=err <= s["bound"]),
            "similarity": dict(value=sim, arithmetic=f"ECAPA {sim} >= 0.8 x {s['ceil']} = {sim_bar}", passed=sim >= sim_bar),
        }
        if lang == "arz":
            bar = round(0.6 * lahg["dialect_egyptian_share"], 4)
            dv = ba["egy_main"]["dialect_egyptian"]
            checks["dialect"] = dict(value=dv, arithmetic=f"Egyptian share {dv} >= 0.6 x {lahg['dialect_egyptian_share']} = {bar}",
                                     passed=dv >= bar)
        passed = all(c["passed"] for c in checks.values())
        name = {"arz": "Egyptian Arabic" if passed else ("Arabic (dialect bar not met)" if all(
            c["passed"] for k, c in checks.items() if k != "dialect") else "not listed"),
                "en": "English" if passed else "not listed", "fr": "French" if passed else "not listed"}[lang]
        res[lang] = dict(checks=checks, passed=passed, listed_as=name)
    return res


def main():
    bo = json.load(open(f"{RESULTS}/bakeoff.json"))
    cal, spk = bo["calibration"], bo["speaker_calibration"]
    ba_rows = [dict(r, condition=r["condition"]) for r in load("full")]
    raw_rows = bakeoff_rows("voxcpm2")
    prm_rows = bakeoff_rows("voxcpm2-prompt")
    ba_t, raw_t, prm_t = table(ba_rows), table(raw_rows), table(prm_rows)
    ba_all = table(ba_rows, held_out_only=False)
    gate = g3(ba_t, cal, spk)
    gate_raw = g3(raw_t, cal, spk)
    supp = {c: dict(ba=ba_t[c]["sim_ecapa"], raw=raw_t[c]["sim_ecapa"], ar_eg_ceiling=spk["ar_eg"]["ceiling_ecapa"],
                    ratio_ba=round(ba_t[c]["sim_ecapa"] / spk["ar_eg"]["ceiling_ecapa"], 4))
            for c in ("cross_en", "cross_fr") if c in ba_t and c in raw_t}
    meta = jl("/home/jovyan/ba-work/engine/full/meta.jsonl")
    retries = dict(utterances_with_retry=sum(1 for m in meta if (m.get("n_retries") or 0) > 0),
                   total_retries=sum(m.get("n_retries") or 0 for m in meta),
                   chunks_failing_after_retries=sum(m.get("chunks_failed") or 0 for m in meta), n=len(meta))
    extra = {}
    for name in ("sweep", "chunk_length", "codeswitch_policy", "reference_thresholds", "asr_check", "watermark", "bench"):
        p = f"{RESULTS}/{name}.json"
        if os.path.exists(p):
            d = json.load(open(p))
            if name == "reference_thresholds":
                extra[name] = d.get("adopted", d["decision"])
            else:
                extra[name] = d.get("decision", d.get("summary", d)) if name not in ("watermark", "bench") else d
    out = dict(held_out_rule="tuning texts excluded: " + " ".join(sorted(TUNE_IDS)),
               ba=ba_t, raw_voxcpm2=raw_t, voxcpm2_prompt_phase1=prm_t, ba_all_texts=ba_all,
               gate_g3=gate, gate_g3_raw_voxcpm2_for_reference=gate_raw, cross_language_similarity=supp,
               engine_checks=retries, decisions=extra,
               commands=dict(synth="BA/eval/scripts/run_engine.sh full", report="engine_report.py"))
    json.dump(out, open(f"{RESULTS}/engine.json", "w"), indent=1, ensure_ascii=False)
    write_md(out)


def f(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def write_md(o):
    L = ["# BA engine: results\n",
         "Generated by `BA/eval/scripts/engine_report.py` from `engine.json`. Numbers in sections 1-6; interpretation in section 7.\n",
         f"Held out: every number below excludes the tuning texts ({o['held_out_rule'].split(': ')[1]}). "
         "BA and raw VoxCPM2 are compared on the same utterances (texts, voices, seeds). CER for Egyptian-text conditions, "
         "WER for English/French. Egyptian digit sentences and passages use the better of the spelled and the digit reference, as in Phase 1.\n",
         "## 1. BA engine vs raw VoxCPM2 (Phase 1 bake-off)\n",
         "| condition | n | fail BA | fail raw | error BA | error raw | sim ECAPA BA | sim ECAPA raw | dialect=Egy BA | dialect=Egy raw | DNSMOS BA | DNSMOS raw |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in COND_ORDER:
        if c not in o["ba"]:
            continue
        b, r = o["ba"][c], o["raw_voxcpm2"].get(c, {})
        key = "cer" if c in ("egy_main", "egy_digits", "diac", "cs", "long_arz") else "wer"
        L.append(f"| {c} | {b['n']} | {b['n_fail']} | {r.get('n_fail', '-')} | {f(b[key])} | {f(r.get(key))} | {f(b['sim_ecapa'])} | "
                 f"{f(r.get('sim_ecapa'))} | {f(b['dialect_egyptian'])} | {f(r.get('dialect_egyptian'))} | {f(b['mos_p808'])} | {f(r.get('mos_p808'))} |")
    cs_b, cs_r = o["ba"].get("cs", {}), o["raw_voxcpm2"].get("cs", {})
    L.append(f"\nCode-switch CER with Latin words mapped to Arabic script on both sides (canonical): BA {f(cs_b.get('cer_canonical'))}, raw {f(cs_r.get('cer_canonical'))}.")
    p = o["voxcpm2_prompt_phase1"].get("egy_main")
    if p:
        L.append(f"Phase 1 VoxCPM2 with the gold FLEURS transcript (voxcpm2-prompt), same held-out Egyptian main utterances: CER {f(p['cer'])}, "
                 f"ECAPA {f(p['sim_ecapa'])}, dialect=Egyptian {f(p['dialect_egyptian'])}, n={p['n']}.")
    L.append("\n## 2. Gate G3 (applied mechanically)\n")
    for lang, g in o["gate_g3"].items():
        L.append(f"**{lang}: {'PASS' if g['passed'] else 'FAIL'} -> listed as: {g['listed_as']}**\n")
        for k, c in g["checks"].items():
            L.append(f"- {k}: {c['arithmetic']} -> {'pass' if c['passed'] else 'FAIL'}")
        L.append("")
    L.append("Raw VoxCPM2 under the same rule (for reference): " + ", ".join(
        f"{l} {'PASS' if g['passed'] else 'FAIL'} ({', '.join(k for k, c in g['checks'].items() if not c['passed']) or 'all checks pass'})"
        for l, g in o["gate_g3_raw_voxcpm2_for_reference"].items()) + ".\n")
    if o["cross_language_similarity"]:
        L.append("Cross-language similarity (Egyptian speaker speaking English/French) against the Egyptian same-speaker ceiling "
                 "(supplementary; not part of G3 as written): " + "; ".join(
                     f"{c}: BA {f(v['ba'])}, raw {f(v['raw'])}, BA/ceiling {f(v['ratio_ba'])}" for c, v in o["cross_language_similarity"].items()) + ".\n")
    L.append("## 3. Engine checks during the run\n")
    e = o["engine_checks"]
    L.append(f"{e['n']} utterances; {e['utterances_with_retry']} needed at least one retry ({e['total_retries']} retries in total); "
             f"{e['chunks_failing_after_retries']} chunks still failed the duration check after two retries.\n")
    d = o["decisions"]
    L.append("## 4. Tuning decisions (tuning texts only)\n")
    if "sweep" in d:
        L.append(f"- cfg / steps (sweep.json): {json.dumps(d['sweep'])}")
    if "chunk_length" in d:
        L.append(f"- max characters per chunk (chunk_length.json): {json.dumps(d['chunk_length'])}")
    if "codeswitch_policy" in d:
        L.append(f"- Latin words in Egyptian text (codeswitch_policy.json): {d['codeswitch_policy']}")
    if "reference_thresholds" in d:
        rt = d["reference_thresholds"]
        L.append(f"- reference rejection (reference_thresholds.json, \"adopted\"): noise estimate ({rt.get('noise_estimator', 'p95-p10').split(':')[0]}) "
                 f"< {rt['REF_SNR_MIN_DB']} dB rejected, < {rt.get('REF_SNR_WARN_DB', '-')} dB warned; clipped-sample share > {rt['REF_CLIP_MAX_RATIO']} rejected")
    if "asr_check" in d:
        a = d["asr_check"]
        L.append("- reference ASR whisper-large-v3-turbo vs large-v3 CER (asr_check.json): " + "; ".join(
            f"{s} {f(v['turbo_cer'])} vs {f(v['v3_cer'])}" for s, v in a.items()))
    L.append("")
    if "bench" in d:
        b = d["bench"]
        L.append("## 5. Speed and memory (A40)\n")
        L.append("| measure | value |\n|---|---|")
        for k, v in b.get("summary", {}).items():
            L.append(f"| {k} | {v} |")
        L.append("")
    if "watermark" in d:
        w = d["watermark"]
        L.append("## 6. Watermark detection\n")
        L.append("| audio | n | detected | mean score | mean bit match |\n|---|---|---|---|---|")
        for k, v in w.get("summary", {}).items():
            L.append(f"| {k} | {v['n']} | {v['detected']} | {f(v['mean_score'])} | {f(v['mean_bits_match'])} |")
        L.append("")
    sp = f"{RESULTS}/space_e2e.json"
    if os.path.exists(sp):
        e2e = json.load(open(sp))
        L.append("## 6b. Space app, local end-to-end (gradio_client, FLEURS ar_eg_f6 reference, no transcript; space_e2e.json)\n")
        L.append(f"- consent unticked: {e2e['consent_unticked'].split(':')[0]}")
        for k, sc in enumerate(e2e.get("scores", [])):
            L.append(f"- request {k}: {e2e.get(f'request_{k}_seconds')} s wall time; CER {f(sc['cer'])} (vs digit text {f(sc['cer_vs_digit_text'])}); "
                     f"ECAPA {f(sc['sim_ecapa'])}; dialect {sc['dialect']}; DNSMOS {f(sc['mos_p808'])}; watermark detected {e2e['watermark'][k]['watermarked']}")
        L.append(f"- copies of the uploaded reference left in Gradio's cache after the requests: {len(e2e['uploaded_reference_copies_left_in_gradio_cache'])}\n")
    pt = f"{RESULTS}/pytest.txt"
    if os.path.exists(pt):
        L.append("## 6c. Tests\n")
        L.append("```\n" + open(pt).read().strip() + "\n```\n")
    obs = f"{RESULTS}/engine_observations.md"
    if os.path.exists(obs):
        L.append("## 7. Interpretation\n")
        L.append(open(obs).read())
    open(f"{RESULTS}/engine.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
