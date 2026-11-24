"""Second, independent Arabic dialect classifier (venv /tmp/ba/venvs/eval).

Classifier 1 (the protocol's): badrex/mms-300m-arabic-dialect-identifier (MMS-300m, CC-BY-4.0, 5 classes,
  trained on MGB-3 ADI-5 Al Jazeera broadcast speech). "Egyptian" = top label Egyptian.
Classifier 2 (this script): IbrahimAmin/hubert-arabic-spoken-dialect-classifier (HuBERT-base-ls960,
  Apache-2.0, 18 classes = MSA + 17 country dialects, trained on ADI-17 (YouTube) + QASR (MSA)); plain
  transformers HubertForSequenceClassification, safetensors, no remote code; revision pinned below.
  "Egyptian" = top label EGY (strict: SUD is not merged, although classifier 1's Egyptian class
  includes Sudanese). P(EGY) is kept as a continuous score.
Both are run on 16 kHz mono audio, whole clip, top label (as in score.py).

Usage:
  python dialect_classifier2.py score [exp ...]   GPU. Scores, with BOTH classifiers, every Arabic-text
        output of the bake-off (all systems in score.PROFILE, plus the listening-pack clips), the BA
        engine run (engine/full), the real-speech calibration clips (lahgtna_eg, NileTTS, FLEURS ar_eg),
        the six ar_eg reference clips, and any extra engine experiments named. Cache (resumable):
        /home/jovyan/ba-work/paper/dialect2/scores.jsonl, one row per wav path.
  python dialect_classifier2.py report            CPU. -> BA/eval/results/dialect_classifier2.json, .md

The classifier-1 labels used in the analysis are the stored ones from the original scoring
(bakeoff_utterances.csv, engine/full/scores.jsonl, scores/_calib.jsonl); the fresh classifier-1 run here
is used only for clips that were never scored (listening pack, references) and to check that the stored
labels reproduce.
"""
import csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from paper_common import (PAPER_WORK, RESULTS, EVAL, ENGINE_ROOT, BAKEOFF_ROOT, TUNE_IDS, jl, bakeoff_rows, engine_rows,
                          two_way_bootstrap, cohen_kappa, fmt, ci, dump)

C1 = dict(model="badrex/mms-300m-arabic-dialect-identifier", revision="6220d51bce99d1fbca7ed96a5ba7966ffde99d22", egy="Egyptian")
C2 = dict(model="IbrahimAmin/hubert-arabic-spoken-dialect-classifier", revision="428745539b9f99a94310c971b9a87591939c5cad", egy="EGY")
CACHE = f"{PAPER_WORK}/dialect2/scores.jsonl"
SYSTEMS = ["cb-v3", "cb-v3-pypi", "cb-v2", "voxcpm2", "namaa", "lahgtna", "omnivoice", "voicetut", "voxcpm2-prompt"]


class DialectID:
    def __init__(self, spec):
        import torch
        from transformers import pipeline
        self.spec = spec
        self.pipe = pipeline("audio-classification", model=spec["model"], revision=spec["revision"], device=0,
                             torch_dtype=torch.float32, model_kwargs={"use_safetensors": True})
        self.n = len(self.pipe.model.config.id2label)

    def __call__(self, y16):
        out = self.pipe({"raw": y16, "sampling_rate": 16000}, top_k=self.n)
        sc = {o["label"]: float(o["score"]) for o in out}
        return dict(label=out[0]["label"], p_egy=round(sc.get(self.spec["egy"], 0.0), 4),
                    top3={o["label"]: round(float(o["score"]), 4) for o in out[:3]})


# ------------------------------------------------------------------ what to score
def sources(extra_exps=()):
    """[(path, source, meta)] for every Arabic-text clip."""
    from matrix import build
    from score import PROFILE
    out = []
    for s in SYSTEMS:
        for j in build(PROFILE[s]):
            if j["lang"] == "arz":
                p = f"{BAKEOFF_ROOT}/{s}/{j['utt_id']}.wav"
                if os.path.exists(p):
                    out.append((p, f"bakeoff:{s}", dict(utt_id=j["utt_id"])))
    lt = json.load(open(f"{RESULTS}/listening_test.json"))
    for r in lt["ratings"]:
        p = f"{BAKEOFF_ROOT}/{r['system']}-pack/pack__{r['text_id']}__{r['voice']}.wav"
        out.append((p, f"pack:{r['system']}", dict(clip=r["clip"])))
    for exp in ("full",) + tuple(extra_exps):
        for m in jl(f"{ENGINE_ROOT}/{exp}/manifest.jsonl"):
            if m["lang"] == "arz" and not m.get("missing") and os.path.exists(m["wav"]):
                out.append((m["wav"], f"engine:{exp}", dict(utt_id=m["utt_id"])))
    for c in jl(f"{EVAL}/calib.jsonl"):
        if c["lang"] == "arz":
            out.append((c["path"], f"calib:{c['set']}", dict(calib_id=c["calib_id"])))
    for r in jl(f"{EVAL}/refs.jsonl"):
        if r["lang"] == "ar_eg":
            out.append((r["path"], "ref", dict(voice_id=r["voice_id"])))
            out.append((r["pair_path"], "ref_pair", dict(voice_id=r["voice_id"])))
    return out


def score(extra):
    from score import load16
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    done = {r["path"] for r in jl(CACHE)}
    todo = [s for s in sources(extra) if s[0] not in done]
    seen = set(); todo = [s for s in todo if not (s[0] in seen or seen.add(s[0]))]
    print(f"{len(todo)} clips to score", flush=True)
    if not todo:
        return
    c1, c2 = DialectID(C1), DialectID(C2)
    with open(CACHE, "a") as f:
        for i, (p, src, meta) in enumerate(todo):
            y = load16(p)
            rec = dict(path=p, source=src, **meta, seconds=round(len(y) / 16000, 2))
            if len(y) < 1600 or float(np.max(np.abs(y))) < 1e-4:
                rec["empty"] = True
            else:
                rec["c1"] = c1(y)
                rec["c2"] = c2(y)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            if i % 50 == 0:
                print(i, src, rec.get("c1", {}).get("label"), rec.get("c2", {}).get("label"), flush=True)


# ------------------------------------------------------------------ analysis
def load_cache():
    return {r["path"]: r for r in jl(CACHE)}


def c2_egy(rec):
    return None if rec is None or "c2" not in rec else float(rec["c2"]["label"] == C2["egy"])


def calibrated_thresholds(cache=None):
    """Per classifier: the P(Egyptian) threshold with the largest Youden J (TPR - FPR) separating real
    colloquial Egyptian speech (lahgtna_eg + NileTTS calibration clips, positives) from read MSA by Egyptian
    speakers (FLEURS ar_eg, negatives). Ties go to the lower threshold."""
    cache = cache or load_cache()
    cal = [r for r in cache.values() if r["source"].startswith("calib:") and "c2" in r]
    out = {}
    for k in ("c1", "c2"):
        pos = np.array([r[k]["p_egy"] for r in cal if r["source"] in ("calib:lahgtna_eg", "calib:niletts")])
        neg = np.array([r[k]["p_egy"] for r in cal if r["source"] == "calib:fleurs_ar_eg"])
        best = None
        for t in np.unique(np.concatenate([pos, neg])):
            j = float((pos >= t).mean() - (neg >= t).mean())
            if best is None or j > best[0] + 1e-12:
                best = (j, float(t), float((pos >= t).mean()), float((neg >= t).mean()))
        egy = (C1 if k == "c1" else C2)["egy"]
        top = dict(tpr=float(np.mean([r[k]["label"] == egy for r in cal if r["source"] in ("calib:lahgtna_eg", "calib:niletts")])),
                   fpr=float(np.mean([r[k]["label"] == egy for r in cal if r["source"] == "calib:fleurs_ar_eg"])))
        out[k] = dict(threshold=round(best[1], 4), youden_j=round(best[0], 4), tpr_colloquial=round(best[2], 4),
                      fpr_read_msa=round(best[3], 4), n_pos=len(pos), n_neg=len(neg),
                      top_label_rule=dict(tpr_colloquial=round(top["tpr"], 4), fpr_read_msa=round(top["fpr"], 4)))
    return out


def label_rows(rr, cache, thr):
    for r in rr:
        rec = cache.get(r["wav"])
        r["c1"] = None if r.get("dialect") is None else float(r["dialect"] == "Egyptian")
        r["c2"] = c2_egy(rec)
        r["c2_p"] = None if rec is None or "c2" not in rec else rec["c2"]["p_egy"]
        r["c1_p"] = None if rec is None or "c1" not in rec else rec["c1"]["p_egy"]
        r["c1_fresh"] = None if rec is None or "c1" not in rec else rec["c1"]["label"]
        r["c1_cal"] = None if r["c1_p"] is None else float(r["c1_p"] >= thr["c1"]["threshold"])
        r["c2_cal"] = None if r["c2_p"] is None else float(r["c2_p"] >= thr["c2"]["threshold"])
    return rr


def system_rows():
    """Bake-off systems (seed 0) plus BA engine (held-out and all texts), with both classifier labels."""
    cache = load_cache()
    thr = calibrated_thresholds(cache)
    rows = {}
    for s in SYSTEMS:
        rows[s] = label_rows([r for r in bakeoff_rows(s) if r["lang"] == "arz"], cache, thr)
    rows["ba"] = label_rows([r for r in engine_rows("full", "ba") if r["lang"] == "arz"], cache, thr)
    return rows


def summarise(rr, cer_key):
    t, v = [r["text_id"] for r in rr], [r["voice_id"] for r in rr]
    return dict(n=len(rr),
                cer=two_way_bootstrap([r.get(cer_key) for r in rr], t, v),
                share_c1=two_way_bootstrap([r["c1"] for r in rr], t, v),
                share_c2=two_way_bootstrap([r["c2"] for r in rr], t, v),
                share_c1_calibrated=two_way_bootstrap([r.get("c1_cal") for r in rr], t, v),
                share_c2_calibrated=two_way_bootstrap([r.get("c2_cal") for r in rr], t, v),
                mean_p_egy_c2=round(float(np.mean([r["c2_p"] for r in rr if r["c2_p"] is not None])), 4) if rr else None,
                kappa=cohen_kappa(*zip(*[(r["c1"], r["c2"]) for r in rr if r["c1"] is not None and r["c2"] is not None]))
                if any(r["c1"] is not None and r["c2"] is not None for r in rr) else None)


def ranks(vals):
    from scipy import stats
    return stats.rankdata(vals)


def report():
    from scipy import stats
    cache = load_cache()
    rows = system_rows()
    out = dict(classifier_1=C1, classifier_2=dict(C2, licence="Apache-2.0", base="facebook/hubert-base-ls960",
                                                  training_data="ADI-17 (YouTube, 17 country dialects) + QASR (Al Jazeera MSA)",
                                                  classes=18, rule="top label == EGY (SUD not merged)"),
               classifier_1_info=dict(licence="CC-BY-4.0", base="facebook/mms-300m", training_data="MGB-3 ADI-5 (Al Jazeera broadcast)",
                                      classes=5, rule="top label == Egyptian (class covers Egyptian and Sudanese)"),
               bootstrap="two-way (texts and voices), 10,000 resamples, 95% percentile intervals (paper_common.py)")
    # ---- calibration on real speech
    cal = {}
    for s in ("lahgtna_eg", "niletts", "fleurs_ar_eg"):
        rr = [r for r in cache.values() if r["source"] == f"calib:{s}" and "c2" in r]
        stored = {r["key"]: r for r in jl(f"{BAKEOFF_ROOT}/scores/_calib.jsonl")}
        c1 = [float(stored[r["calib_id"]].get("dialect") == "Egyptian") for r in rr if stored.get(r["calib_id"], {}).get("dialect")]
        c2 = [c2_egy(r) for r in rr]
        labs = {}
        for r in rr:
            labs[r["c2"]["label"]] = labs.get(r["c2"]["label"], 0) + 1
        cal[s] = dict(n=len(rr), share_c1=round(float(np.mean(c1)), 4), share_c2=round(float(np.mean(c2)), 4),
                      mean_p_egy_c2=round(float(np.mean([r["c2"]["p_egy"] for r in rr])), 4),
                      c2_label_counts=dict(sorted(labs.items(), key=lambda kv: -kv[1])),
                      kappa=cohen_kappa([float(stored[r["calib_id"]].get("dialect") == "Egyptian") for r in rr],
                                        [c2_egy(r) for r in rr]))
    out["calibration_real_speech"] = cal
    refs = [r for r in cache.values() if r["source"] in ("ref", "ref_pair") and "c2" in r]
    out["reference_clips"] = [dict(voice_id=r["voice_id"], clip=r["source"], c1=r["c1"]["label"], c2=r["c2"]["label"],
                                   c2_p_egy=r["c2"]["p_egy"]) for r in sorted(refs, key=lambda r: (r["voice_id"], r["source"]))]
    # ---- reproduction of stored classifier-1 labels
    rep = [(r["dialect"], r["c1_fresh"]) for s in rows for r in rows[s] if r.get("dialect") and r.get("c1_fresh")]
    out["classifier_1_reproduction"] = dict(n=len(rep), same_label=sum(1 for a, b in rep if a == b))
    # ---- per system
    bo = json.load(open(f"{RESULTS}/bakeoff.json"))["systems"]
    per = {}
    for s, rr in rows.items():
        if s == "ba":
            ho = [r for r in rr if r["text_id"] not in TUNE_IDS]
            per[s] = dict(egy_main=summarise([r for r in ho if r["cond"] == "egy_main"], "cer_best"),
                          egy_digits=summarise([r for r in ho if r["cond"] == "egy_digits"], "cer_best"),
                          all_arabic_text=summarise(ho, "cer_best"), subset="held-out texts only (engine.md)")
        else:
            per[s] = dict(egy_main=summarise([r for r in rr if r["cond"] == "egy_main"], "cer"),
                          egy_digits=summarise([r for r in rr if r["cond"] == "egy_digits"], "cer_best"),
                          all_arabic_text=summarise(rr, "cer_best"), licence=bo[s]["licence"],
                          failure_rate_all_conditions=bo[s]["overall"]["fail_rate"])
    raw_ho = [r for r in rows["voxcpm2"] if r["text_id"] not in TUNE_IDS]
    per["voxcpm2_heldout"] = dict(egy_main=summarise([r for r in raw_ho if r["cond"] == "egy_main"], "cer"),
                                  egy_digits=summarise([r for r in raw_ho if r["cond"] == "egy_digits"], "cer_best"),
                                  subset="raw VoxCPM2 on the same held-out utterances as ba")
    out["systems"] = per
    # ---- agreement pooled over every generated Egyptian-text clip
    pool = [(r["c1"], r["c2"]) for s in rows for r in rows[s] if r["c1"] is not None and r["c2"] is not None]
    out["agreement_all_generated"] = cohen_kappa(*zip(*pool))
    pool_m = [(r["c1"], r["c2"]) for s in SYSTEMS for r in rows[s] if r["cond"] == "egy_main" and r["c1"] is not None and r["c2"] is not None]
    out["agreement_egy_main_bakeoff"] = cohen_kappa(*zip(*pool_m))
    calpool = []
    stored = {r["key"]: r for r in jl(f"{BAKEOFF_ROOT}/scores/_calib.jsonl")}
    for r in cache.values():
        if r["source"].startswith("calib:") and "c2" in r and stored.get(r["calib_id"], {}).get("dialect"):
            calpool.append((float(stored[r["calib_id"]]["dialect"] == "Egyptian"), c2_egy(r)))
    out["agreement_real_speech"] = cohen_kappa(*zip(*calpool))
    # ---- ranking (9 bake-off systems, Egyptian main set)
    names = SYSTEMS
    s1 = [per[s]["egy_main"]["share_c1"]["mean"] for s in names]
    s2 = [per[s]["egy_main"]["share_c2"]["mean"] for s in names]
    cer = [per[s]["egy_main"]["cer"]["mean"] for s in names]
    eligible = ["cb-v3", "cb-v2", "voxcpm2"]
    rk = dict(order_by_c1=[n for _, n in sorted(zip(s1, names), reverse=True)],
              order_by_c2=[n for _, n in sorted(zip(s2, names), reverse=True)],
              order_by_cer=[n for _, n in sorted(zip(cer, names))],
              spearman_c1_c2=dict(zip(("rho", "p"), map(lambda x: round(float(x), 4), stats.spearmanr(s1, s2)))),
              kendall_c1_c2=dict(zip(("tau", "p"), map(lambda x: round(float(x), 4), stats.kendalltau(s1, s2)))),
              spearman_cer_vs_share_c1=dict(zip(("rho", "p"), map(lambda x: round(float(x), 4), stats.spearmanr(cer, s1)))),
              spearman_cer_vs_share_c2=dict(zip(("rho", "p"), map(lambda x: round(float(x), 4), stats.spearmanr(cer, s2)))),
              eligible_systems={s: dict(cer=per[s]["egy_main"]["cer"]["mean"], share_c1=per[s]["egy_main"]["share_c1"]["mean"],
                                        share_c2=per[s]["egy_main"]["share_c2"]["mean"]) for s in eligible})
    e = rk["eligible_systems"]
    rk["dissociation_check"] = dict(
        lowest_cer_eligible=min(eligible, key=lambda s: e[s]["cer"]),
        highest_share_c1_eligible=max(eligible, key=lambda s: e[s]["share_c1"]),
        highest_share_c2_eligible=max(eligible, key=lambda s: e[s]["share_c2"]),
        voxcpm2_above_cbv3_c1=e["voxcpm2"]["share_c1"] > e["cb-v3"]["share_c1"],
        voxcpm2_above_cbv3_c2=e["voxcpm2"]["share_c2"] > e["cb-v3"]["share_c2"],
        cbv3_c2_ci=per["cb-v3"]["egy_main"]["share_c2"], voxcpm2_c2_ci=per["voxcpm2"]["egy_main"]["share_c2"],
        cis_overlap_c2=not (per["cb-v3"]["egy_main"]["share_c2"]["hi"] < per["voxcpm2"]["egy_main"]["share_c2"]["lo"]))
    # ---- robustness: P(Egyptian) thresholds calibrated on real speech (Youden J)
    thr = calibrated_thresholds(cache)
    out["calibrated_thresholds"] = thr
    sc1 = [per[s]["egy_main"]["share_c1_calibrated"]["mean"] for s in names]
    sc2 = [per[s]["egy_main"]["share_c2_calibrated"]["mean"] for s in names]
    rk["calibrated"] = dict(
        order_by_c1=[n for _, n in sorted(zip(sc1, names), reverse=True)],
        order_by_c2=[n for _, n in sorted(zip(sc2, names), reverse=True)],
        spearman_c1_c2=dict(zip(("rho", "p"), map(lambda x: round(float(x), 4), stats.spearmanr(sc1, sc2)))),
        spearman_cer_vs_share_c1=dict(zip(("rho", "p"), map(lambda x: round(float(x), 4), stats.spearmanr(cer, sc1)))),
        spearman_cer_vs_share_c2=dict(zip(("rho", "p"), map(lambda x: round(float(x), 4), stats.spearmanr(cer, sc2)))),
        eligible_systems={s: dict(share_c1=per[s]["egy_main"]["share_c1_calibrated"]["mean"],
                                  share_c2=per[s]["egy_main"]["share_c2_calibrated"]["mean"]) for s in eligible})
    out["ranking"] = rk
    # ---- paired: voxcpm2 vs cb-v3 (same utterances) under classifier 2, and BA vs raw (held-out)
    from paper_common import paired
    vx = [r for r in rows["voxcpm2"] if r["cond"] == "egy_main"]
    cb = [r for r in rows["cb-v3"] if r["cond"] == "egy_main"]
    bam = [r for r in rows["ba"] if r["cond"] == "egy_main" and r["text_id"] not in TUNE_IDS]
    rawm = [r for r in raw_ho if r["cond"] == "egy_main"]
    out["paired_c2"] = dict(
        voxcpm2_minus_cbv3_egy_main=paired(vx, cb, "c2", binary=True),
        ba_minus_voxcpm2_heldout_egy_main=paired(bam, rawm, "c2", binary=True),
        voxcpm2_minus_cbv3_egy_main_p_egy=paired(vx, cb, "c2_p"),
        voxcpm2_minus_cbv3_egy_main_c2_calibrated=paired(vx, cb, "c2_cal", binary=True),
        voxcpm2_minus_cbv3_egy_main_c1_calibrated=paired(vx, cb, "c1_cal", binary=True),
        ba_minus_voxcpm2_heldout_egy_main_c2_calibrated=paired(bam, rawm, "c2_cal", binary=True))
    # ---- listening test (owner, 1 rater, 48 clips)
    lt = json.load(open(f"{RESULTS}/listening_test.json"))
    val = {"egy": 1.0, "mid": 0.5, "no": 0.0}
    clips = []
    for r in lt["ratings"]:
        p = f"{BAKEOFF_ROOT}/{r['system']}-pack/pack__{r['text_id']}__{r['voice']}.wav"
        c = cache.get(p)
        if c and "c2" in c:
            clips.append(dict(clip=r["clip"], system=r["system"], text_id=r["text_id"], rating=val[r["dialect"]],
                              words_ok=r["words"] == "ok", c1=c["c1"]["label"], c1_egy=float(c["c1"]["label"] == "Egyptian"),
                              c1_p_egy=c["c1"]["p_egy"], c2=c["c2"]["label"], c2_egy=float(c["c2"]["label"] == "EGY"), c2_p_egy=c["c2"]["p_egy"]))
    lt_sys = {}
    for s in sorted({c["system"] for c in clips}):
        cc = [c for c in clips if c["system"] == s]
        lt_sys[s] = dict(n=len(cc), owner_mean=round(float(np.mean([c["rating"] for c in cc])), 4),
                         owner_counts={k: sum(1 for c in cc if c["rating"] == v) for k, v in val.items()},
                         c1_share_rated_clips=round(float(np.mean([c["c1_egy"] for c in cc])), 4),
                         c2_share_rated_clips=round(float(np.mean([c["c2_egy"] for c in cc])), 4),
                         c1_share_egy_main=per[s]["egy_main"]["share_c1"]["mean"], c2_share_egy_main=per[s]["egy_main"]["share_c2"]["mean"])
    ss = sorted(lt_sys)
    om = [lt_sys[s]["owner_mean"] for s in ss]

    def sp(a, b):
        r = stats.spearmanr(a, b)
        return dict(rho=round(float(r[0]), 4), p=round(float(r[1]), 4))
    out["listening_test"] = dict(
        note="one rater (the owner), 6 sentences x 8 raw systems = 48 clips; rating egy=1, partly=0.5, not=0",
        per_system=lt_sys,
        system_level=dict(n_systems=len(ss), owner_vs_c1_egy_main=sp(om, [lt_sys[s]["c1_share_egy_main"] for s in ss]),
                          owner_vs_c2_egy_main=sp(om, [lt_sys[s]["c2_share_egy_main"] for s in ss]),
                          owner_vs_c1_rated_clips=sp(om, [lt_sys[s]["c1_share_rated_clips"] for s in ss]),
                          owner_vs_c2_rated_clips=sp(om, [lt_sys[s]["c2_share_rated_clips"] for s in ss])),
        clip_level=dict(n=len(clips), owner_vs_c1_p_egy=sp([c["rating"] for c in clips], [c["c1_p_egy"] for c in clips]),
                        owner_vs_c2_p_egy=sp([c["rating"] for c in clips], [c["c2_p_egy"] for c in clips]),
                        kappa_owner_not_vs_c1=cohen_kappa([float(c["rating"] > 0) for c in clips], [c["c1_egy"] for c in clips]),
                        kappa_owner_not_vs_c2=cohen_kappa([float(c["rating"] > 0) for c in clips], [c["c2_egy"] for c in clips])),
        clips=clips)
    out["command"] = "/tmp/ba/venvs/eval/bin/python BA/eval/scripts/dialect_classifier2.py score && ... report"
    dump(out, f"{RESULTS}/dialect_classifier2.json")
    write_md(out)


def write_md(o):
    L = ["# Second dialect classifier\n",
         "Generated by `BA/eval/scripts/dialect_classifier2.py report` from `dialect_classifier2.json`. Classifier 1 = "
         f"`{o['classifier_1']['model']}` ({o['classifier_1_info']['training_data']}); classifier 2 = `{o['classifier_2']['model']}` "
         f"({o['classifier_2']['training_data']}; {o['classifier_2']['licence']}). Egyptian = top label (classifier 2: EGY only). "
         f"CIs: {o['bootstrap']}.\n",
         "## 1. Calibration on real speech\n",
         "| set | n | Egyptian share, classifier 1 | Egyptian share, classifier 2 | mean P(EGY), clf 2 | kappa | classifier 2 labels |",
         "|---|---|---|---|---|---|---|"]
    for s, c in o["calibration_real_speech"].items():
        L.append(f"| {s} | {c['n']} | {fmt(c['share_c1'])} | {fmt(c['share_c2'])} | {fmt(c['mean_p_egy_c2'])} | {fmt(c['kappa']['kappa'])} | "
                 + ", ".join(f"{k} {v}" for k, v in list(c["c2_label_counts"].items())[:5]) + " |")
    L.append("\nReference clips (FLEURS ar_eg, read MSA): " + "; ".join(f"{r['voice_id']}/{r['clip']}: {r['c1']} / {r['c2']}" for r in o["reference_clips"]))
    rp = o["classifier_1_reproduction"]
    L.append(f"\nStored classifier-1 labels reproduced by the fresh run: {rp['same_label']}/{rp['n']}.\n")
    L.append("## 2. Egyptian main set per system (bake-off: 32 texts x 3 voices; BA: held-out 24 x 3)\n")
    L.append("| system | n | CER [95% CI] | Egyptian share clf 1 [CI] | Egyptian share clf 2 [CI] | mean P(EGY) clf 2 | kappa |")
    L.append("|---|---|---|---|---|---|---|")
    for s, p in o["systems"].items():
        m = p["egy_main"]
        L.append(f"| {s} | {m['n']} | {ci(m['cer'])} | {ci(m['share_c1'])} | {ci(m['share_c2'])} | {fmt(m['mean_p_egy_c2'])} | {fmt((m['kappa'] or {}).get('kappa'))} |")
    L.append("\nDigit sentences (CER = best of spelled/digit reference):\n")
    L.append("| system | n | CER | share clf 1 | share clf 2 |\n|---|---|---|---|---|")
    for s, p in o["systems"].items():
        m = p.get("egy_digits")
        if m and m["n"]:
            L.append(f"| {s} | {m['n']} | {ci(m['cer'])} | {ci(m['share_c1'])} | {ci(m['share_c2'])} |")
    a, b, c = o["agreement_all_generated"], o["agreement_egy_main_bakeoff"], o["agreement_real_speech"]
    L.append("\n## 3. Agreement (Egyptian vs not)\n")
    L.append(f"- all generated Egyptian-text clips (bake-off + BA): kappa {fmt(a['kappa'])}, observed agreement {fmt(a['observed_agreement'])}, n={a['n']}, table {a['table']}")
    L.append(f"- bake-off Egyptian main set, 9 systems: kappa {fmt(b['kappa'])}, observed {fmt(b['observed_agreement'])}, n={b['n']}")
    L.append(f"- real speech (150 calibration clips): kappa {fmt(c['kappa'])}, observed {fmt(c['observed_agreement'])}, n={c['n']}\n")
    r = o["ranking"]
    L.append("## 4. Ranking of the 9 bake-off systems (Egyptian main set)\n")
    L.append(f"- by classifier 1 share: {', '.join(r['order_by_c1'])}")
    L.append(f"- by classifier 2 share: {', '.join(r['order_by_c2'])}")
    L.append(f"- by CER (low to high): {', '.join(r['order_by_cer'])}")
    L.append(f"- Spearman share clf 1 vs clf 2: rho {r['spearman_c1_c2']['rho']} (p {r['spearman_c1_c2']['p']}); Kendall tau {r['kendall_c1_c2']['tau']} (p {r['kendall_c1_c2']['p']})")
    L.append(f"- Spearman CER vs share: clf 1 rho {r['spearman_cer_vs_share_c1']['rho']} (p {r['spearman_cer_vs_share_c1']['p']}); clf 2 rho {r['spearman_cer_vs_share_c2']['rho']} (p {r['spearman_cer_vs_share_c2']['p']})")
    d = r["dissociation_check"]
    L.append(f"- licence-eligible systems: lowest CER {d['lowest_cer_eligible']}; highest Egyptian share clf 1 {d['highest_share_c1_eligible']}, clf 2 {d['highest_share_c2_eligible']}; "
             f"VoxCPM2 above Chatterbox V3: clf 1 {d['voxcpm2_above_cbv3_c1']}, clf 2 {d['voxcpm2_above_cbv3_c2']} (clf 2 CIs overlap: {d['cis_overlap_c2']})\n")
    th = o["calibrated_thresholds"]
    L.append("### 4b. Robustness: decision thresholds calibrated on real speech\n")
    L.append("Threshold on P(Egyptian) with the largest Youden J separating real colloquial Egyptian (lahgtna_eg + NileTTS, n=100) "
             "from read MSA by Egyptian speakers (FLEURS ar_eg, n=50):\n")
    L.append("| classifier | threshold | TPR colloquial | FPR read MSA | top-label rule TPR / FPR |\n|---|---|---|---|---|")
    for k, t in th.items():
        L.append(f"| {k} | {t['threshold']} | {fmt(t['tpr_colloquial'])} | {fmt(t['fpr_read_msa'])} | {fmt(t['top_label_rule']['tpr_colloquial'])} / {fmt(t['top_label_rule']['fpr_read_msa'])} |")
    L.append("\n| system | share clf 1, calibrated [CI] | share clf 2, calibrated [CI] |\n|---|---|---|")
    for s, p in o["systems"].items():
        m = p["egy_main"]
        L.append(f"| {s} | {ci(m['share_c1_calibrated'])} | {ci(m['share_c2_calibrated'])} |")
    rc = r["calibrated"]
    L.append(f"\n- by calibrated clf 1: {', '.join(rc['order_by_c1'])}\n- by calibrated clf 2: {', '.join(rc['order_by_c2'])}")
    L.append(f"- Spearman calibrated clf 1 vs clf 2: rho {rc['spearman_c1_c2']['rho']} (p {rc['spearman_c1_c2']['p']}); CER vs share: clf 1 rho "
             f"{rc['spearman_cer_vs_share_c1']['rho']} (p {rc['spearman_cer_vs_share_c1']['p']}), clf 2 rho {rc['spearman_cer_vs_share_c2']['rho']} (p {rc['spearman_cer_vs_share_c2']['p']})\n")
    L.append("Paired (two-way bootstrap over the shared text-voice cells; c2 = classifier 2 top label, c2_p = P(EGY), *_cal = calibrated threshold):\n")
    for k, p in o["paired_c2"].items():
        if p:
            L.append(f"- {k}: {fmt(p['mean_a'])} vs {fmt(p['mean_b'])}, difference {fmt(p['diff'])} [{fmt(p['ci_lo'])}, {fmt(p['ci_hi'])}], "
                     f"bootstrap p {p['p_bootstrap']}, Wilcoxon p {p['p_wilcoxon']}, Cohen's h {p.get('cohens_h', '-')}, d_z {p['d_z']}, n cells {p['n_cells']}")
    lt = o["listening_test"]
    L.append("\n## 5. Owner's listening test (1 rater, 48 clips) vs the classifiers\n")
    L.append("| system | owner mean (egy=1, partly=.5, not=0) | clf 1 share, rated clips | clf 2 share, rated clips | clf 1 share, main set | clf 2 share, main set |")
    L.append("|---|---|---|---|---|---|")
    for s, v in sorted(lt["per_system"].items(), key=lambda kv: -kv[1]["owner_mean"]):
        L.append(f"| {s} | {fmt(v['owner_mean'])} | {fmt(v['c1_share_rated_clips'])} | {fmt(v['c2_share_rated_clips'])} | {fmt(v['c1_share_egy_main'])} | {fmt(v['c2_share_egy_main'])} |")
    sl, cl = lt["system_level"], lt["clip_level"]
    L.append(f"\nSystem level (n={sl['n_systems']}), Spearman owner mean vs share on the main set: clf 1 rho {sl['owner_vs_c1_egy_main']['rho']} (p {sl['owner_vs_c1_egy_main']['p']}), "
             f"clf 2 rho {sl['owner_vs_c2_egy_main']['rho']} (p {sl['owner_vs_c2_egy_main']['p']}). Clip level (n={cl['n']}), Spearman rating vs P(Egyptian): "
             f"clf 1 {cl['owner_vs_c1_p_egy']['rho']} (p {cl['owner_vs_c1_p_egy']['p']}), clf 2 {cl['owner_vs_c2_p_egy']['rho']} (p {cl['owner_vs_c2_p_egy']['p']}); "
             f"kappa (owner Egyptian-or-partly vs classifier Egyptian): clf 1 {cl['kappa_owner_not_vs_c1']['kappa']}, clf 2 {cl['kappa_owner_not_vs_c2']['kappa']}.\n")
    obs = f"{RESULTS}/dialect_classifier2_observations.md"
    if os.path.exists(obs):
        L.append("## 6. Interpretation\n")
        L.append(open(obs).read())
    open(f"{RESULTS}/dialect_classifier2.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "score":
        score(sys.argv[2:])
    else:
        report()
