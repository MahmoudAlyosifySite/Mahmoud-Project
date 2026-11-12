"""Turn the Phase 2 tuning experiments into decisions (venv /tmp/ba/venvs/eval or ba).

Usage: python analyze_engine.py ref|cs|chunk|sweep
Rules (written before the experiments were scored):

ref   -> results/reference_thresholds.json
      Per language and degradation family (pink noise; hard clipping), mean ECAPA similarity to the
      CLEAN original clip and mean CER (arz) / WER (en) over 3 voices x 3 texts per level. A level is
      harmful if similarity falls more than 0.03 below the clean-reference result, or the error rises
      more than 0.05 absolute, or any output fails. Threshold = midpoint, in the engine's own measure
      (snr_proxy dB; clip ratio), between the mildest harmful level and the next milder level, taking
      the stricter of the two languages. If no level is harmful, the most severe tested level is the
      bound. The false-rejection rate of the chosen thresholds on the 230 real calibration clips is reported.
cs    -> results/codeswitch_policy.json
      Canonical-script CER: Latin words in reference and ASR output are mapped to Arabic script with the
      loanword lexicon before Arabic normalisation, so the same word in either script counts as the same.
      Lower mean CER wins; within 10% relative, the higher Egyptian dialect share wins; still tied: "keep".
chunk -> results/chunk_length.json
      Per language: lowest mean error (CER arz, best of spelled/digit reference; WER en), counting
      failures as error 1.0; within 10% relative or 0.005 absolute, the larger chunk wins (fewer joins).
sweep -> results/sweep.json
      Per language on the tuning set: drop settings with any failed utterance; primary metric CER (arz;
      digit texts use the better of spelled/digit reference) or WER (en); candidates within 10% relative
      (+0.005 absolute) of the best; Egyptian candidates must also keep the dialect share within 0.10 of
      the best setting; among candidates the highest mean ECAPA similarity wins; similarity ties (<=0.005)
      go to fewer steps, then cfg closest to 2.0 (the package default).
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from engine_common import ENGINE_ROOT, RESULTS, EVAL, REF_CONDITIONS, CHUNK_SETTINGS, SWEEP, jl  # noqa: E402
from score import norm_ar, err_rates  # noqa: E402

BAND = json.load(open(f"{RESULTS}/bakeoff.json"))["failure_band_chars_per_sec"]
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")


def mean(x):
    x = [v for v in x if v is not None]
    return round(float(np.mean(x)), 4) if x else None


def load(exp):
    meta = {r["utt_id"]: r for r in jl(f"{ENGINE_ROOT}/{exp}/meta.jsonl")}
    man = {r["utt_id"]: r for r in jl(f"{ENGINE_ROOT}/{exp}/manifest.jsonl")}
    sc = {r["key"]: r for r in jl(f"{ENGINE_ROOT}/{exp}/scores.jsonl")}
    rows = []
    for k, m in man.items():
        r = dict(m["job"]); r.update(meta.get(k, {})); s = sc.get(k)
        if s is None:
            continue
        r.update({x: s.get(x) for x in ("asr", "cer", "wer", "cer_vs_digit_text", "wer_vs_digit_text", "sim_ecapa",
                                        "sim_wavlm", "dialect", "mos_p808", "active_dur", "peak")})
        r["fail"] = failure(r, s, m)
        if r["fail"] in ("missing", "empty"):
            r["cer"] = r["wer"] = 1.0
        r["cer_best"] = min(r["cer"], r["cer_vs_digit_text"]) if r.get("cer_vs_digit_text") is not None else r["cer"]
        r["wer_best"] = min(r["wer"], r["wer_vs_digit_text"]) if r.get("wer_vs_digit_text") is not None else r["wer"]
        r["err"] = r["cer_best"] if r["lang"] == "arz" else r["wer_best"]
        rows.append(r)
    return rows


def failure(r, s, m):
    """Phase 1 failure rule (aggregate.py F1, F2, F4; F3 is Chatterbox-only)."""
    from aggregate import nchars
    if s.get("missing") or m.get("missing"):
        return "missing"
    if s["active_dur"] < 0.3 or s["peak"] < 0.01:
        return "empty"
    c = nchars(m["ref_text"], m["lang"]) / s["active_dur"]
    lo, hi = BAND[m["lang"]]
    return "too_slow" if c < lo else ("too_fast" if c > hi else None)


def summ(rows):
    d = [r["dialect"] for r in rows if r.get("dialect")]
    return dict(n=len(rows), n_fail=sum(1 for r in rows if r["fail"]), err=mean([r["err"] for r in rows]),
                cer=mean([r["cer_best"] for r in rows]), wer=mean([r["wer_best"] for r in rows]),
                sim_ecapa=mean([r["sim_ecapa"] for r in rows]), sim_wavlm=mean([r["sim_wavlm"] for r in rows]),
                dialect_egyptian=round(d.count("Egyptian") / len(d), 4) if d else None,
                mos_p808=mean([r["mos_p808"] for r in rows]),
                gen_sec=mean([r.get("gen_sec") for r in rows]), rtf=mean([r.get("rtf") for r in rows]),
                retries=sum(r.get("n_retries") or 0 for r in rows))


# ------------------------------------------------------------------ ref
def analyse_ref():
    rows = load("ref")
    vinfo = {(v["voice_id"], v["ref_cond"]): v for v in jl(f"{ENGINE_ROOT}/ref/voices.jsonl")}
    out = {"levels": {}, "decision": {}}
    fam = {"noise": ["clean", "snr30", "snr20", "snr15", "snr10", "snr5"], "clip": ["clean", "clip6", "clip12", "clip20"]}
    bounds = {}
    for lang in ("arz", "en"):
        L = [r for r in rows if r["lang"] == lang]
        for cond in REF_CONDITIONS:
            rr = [r for r in L if r["ref_cond"] == cond]
            vs = {r["voice_id"] for r in rr}
            out["levels"][f"{lang}/{cond}"] = dict(**summ(rr), snr_proxy_db=mean([vinfo[(v, cond)]["snr_proxy_db"] for v in vs]),
                                                   clip_ratio=mean([vinfo[(v, cond)]["clip_ratio"] for v in vs]))
        clean = out["levels"][f"{lang}/clean"]
        for f, conds in fam.items():
            harm = None
            for i, c in enumerate(conds[1:], 1):
                lv = out["levels"][f"{lang}/{c}"]
                bad = (clean["sim_ecapa"] - lv["sim_ecapa"] > 0.03) or (lv["err"] - clean["err"] > 0.05) or lv["n_fail"] > 0
                lv["harmful"] = bool(bad)
                if bad and harm is None:
                    harm = (c, conds[i - 1])
            key = "snr_proxy_db" if f == "noise" else "clip_ratio"
            if harm:
                a, b = out["levels"][f"{lang}/{harm[0]}"][key], out["levels"][f"{lang}/{harm[1]}"][key]
                bound = (a + b) / 2
            else:
                bound = out["levels"][f"{lang}/{conds[-1]}"][key]
            bounds.setdefault(f, []).append(dict(lang=lang, first_harmful=harm[0] if harm else None, bound=round(bound, 5)))
    snr_min = max(b["bound"] for b in bounds["noise"])
    clip_max = min(b["bound"] for b in bounds["clip"])
    # false rejections on real clean speech
    sys.path.insert(0, SRC)
    from ba_tts.reference import analyse
    calib = jl(f"{EVAL}/calib.jsonl")
    cal = []
    for c in calib:
        rec = analyse(c["path"])
        cal.append(dict(set=c["set"], snr=rec.snr_db, clip=rec.raw_clip_ratio))
    fr = {s: dict(n=sum(1 for c in cal if c["set"] == s),
                  rejected_noisy=sum(1 for c in cal if c["set"] == s and c["snr"] < snr_min),
                  rejected_clipped=sum(1 for c in cal if c["set"] == s and c["clip"] > clip_max),
                  snr_min_observed=round(min(c["snr"] for c in cal if c["set"] == s), 2),
                  snr_p5=round(float(np.percentile([c["snr"] for c in cal if c["set"] == s], 5)), 2),
                  clip_max_observed=max(c["clip"] for c in cal if c["set"] == s)) for s in sorted({c["set"] for c in cal})}
    out["decision"] = dict(REF_SNR_MIN_DB=round(snr_min, 1), REF_CLIP_MAX_RATIO=round(clip_max, 5), bounds=bounds,
                           calibration_false_rejections=fr)
    json.dump(out, open(f"{RESULTS}/reference_thresholds.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out["decision"], indent=1))


# ------------------------------------------------------------------ cs
def canonical(s):
    sys.path.insert(0, SRC)
    from ba_tts.text.loanwords_arz import LOANWORDS
    s = s.replace("الـ", "ال").replace("للـ", "لل")
    s = re.sub(r"[A-Za-z][A-Za-z\-']*", lambda m: LOANWORDS.get(m.group(0).lower(), m.group(0).lower()), s)
    return norm_ar(s)


def analyse_cs():
    import jiwer
    rows = load("cs")
    for r in rows:
        ref, hyp = canonical(r["text_spoken"] or r_text(r)), canonical(r.get("asr") or "")
        r["cer_canonical"] = float(jiwer.cer(ref, hyp)) if hyp else 1.0
    pol = {}
    for p in ("keep", "arabic"):
        rr = [r for r in rows if r["latin"] == p]
        pol[p] = dict(**summ(rr), cer_canonical=mean([r["cer_canonical"] for r in rr]))
    a, b = pol["keep"]["cer_canonical"], pol["arabic"]["cer_canonical"]
    best = "keep" if a <= b else "arabic"
    other = "arabic" if best == "keep" else "keep"
    steps = [f"canonical CER keep={a}, arabic={b}"]
    if abs(a - b) <= 0.10 * min(a, b):
        steps.append("within 10% relative -> dialect share decides")
        da, db = pol["keep"]["dialect_egyptian"], pol["arabic"]["dialect_egyptian"]
        best = "arabic" if db > da else "keep"
        steps.append(f"dialect keep={da}, arabic={db} -> {best}")
    else:
        steps.append(f"{best} is lower by more than 10% relative")
    out = dict(policies=pol, decision=best, steps=steps, examples=[dict(utt_id=r["utt_id"], asr=r.get("asr")) for r in rows[:12]])
    json.dump(out, open(f"{RESULTS}/codeswitch_policy.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=1, ensure_ascii=False))


def r_text(r):
    from engine_common import _text_map
    return _text_map()[r["text_id"]]["text"]


# ------------------------------------------------------------------ chunk
def analyse_chunk():
    rows = load("chunk")
    out = {"settings": {}, "decision": {}}
    for lang in ("arz", "en"):
        res = {}
        for mc in CHUNK_SETTINGS:
            rr = [r for r in rows if r["lang"] == lang and r["max_chunk_chars"] == mc]
            res[mc] = dict(**summ(rr), n_chunks=mean([r.get("n_chunks") for r in rr]))
        out["settings"][lang] = res
        best_err = min(v["err"] for v in res.values())
        cands = [mc for mc, v in res.items() if v["err"] <= best_err * 1.10 + 0.005]
        out["decision"][lang] = max(cands)
    json.dump(out, open(f"{RESULTS}/chunk_length.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------ sweep
def analyse_sweep():
    rows = load("sweep")
    out = {"settings": {}, "decision": {}, "steps": {}}
    for lang in ("arz", "en"):
        res = {}
        for cfg, st in SWEEP:
            tag = f"cfg{cfg}_s{st}"
            rr = [r for r in rows if r["lang"] == lang and r["setting"] == tag]
            res[tag] = dict(cfg_value=cfg, inference_timesteps=st, **summ(rr))
        out["settings"][lang] = res
        lines = []
        ok = {t: v for t, v in res.items() if v["n_fail"] == 0 and v["n"] > 0}
        lines.append(f"settings without failures: {sorted(ok)}")
        best = min(v["err"] for v in ok.values())
        cands = {t: v for t, v in ok.items() if v["err"] <= best * 1.10 + 0.005}
        lines.append(f"best error {best}; within 10%+0.005: {sorted(cands)}")
        if lang == "arz":
            bd = max(v["dialect_egyptian"] for v in ok.values())
            cands = {t: v for t, v in cands.items() if v["dialect_egyptian"] >= bd - 0.10}
            lines.append(f"best dialect share {bd}; keeping within 0.10: {sorted(cands)}")
        top = max(v["sim_ecapa"] for v in cands.values())
        tied = [t for t, v in cands.items() if top - v["sim_ecapa"] <= 0.005]
        lines.append(f"highest similarity {top}; within 0.005: {sorted(tied)}")
        tied.sort(key=lambda t: (res[t]["inference_timesteps"], abs(res[t]["cfg_value"] - 2.0)))
        out["decision"][lang] = dict(cfg_value=res[tied[0]]["cfg_value"], inference_timesteps=res[tied[0]]["inference_timesteps"])
        lines.append(f"chosen: {tied[0]}")
        out["steps"][lang] = lines
    json.dump(out, open(f"{RESULTS}/sweep.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("decision", "steps")}, indent=1))


if __name__ == "__main__":
    {"ref": analyse_ref, "cs": analyse_cs, "chunk": analyse_chunk, "sweep": analyse_sweep}[sys.argv[1]]()
