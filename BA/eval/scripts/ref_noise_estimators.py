"""Compare noise estimators for the reference gate (CPU; venv eval or ba). -> results/reference_noise_estimators.json

Why: analyze_engine.py ref showed that the Phase 1 SNR proxy (95th - 10th percentile of frame RMS) cannot
separate clean recordings with few pauses from noisy ones (the mechanical threshold rejects about half of the
real clean calibration clips). Same audio as the ref experiment: the 230 real calibration clips plus the 12 FLEURS
references (clean), and the 6 experiment voices degraded with the same seeded pink noise (synth_ba.degrade).
Estimators (dB): p95 minus the 10th/5th/2nd/1st percentile of 25 ms frame RMS, and p95 minus the median RMS of
frames inside pauses >= 150 ms found by the engine's VAD (p2 when there is no pause).
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
sys.path.insert(0, SRC)
import numpy as np
from engine_common import EVAL, RESULTS, REF_VOICES, jl, load_refs, seed_of
from ba_tts.audio import load_audio, resample, frame_rms_db
from ba_tts.reference import speech_segments


def estimators(y16):
    r = frame_rms_db(y16)
    p95 = np.percentile(r, 95)
    out = {f"p95-p{q}": float(p95 - np.percentile(r, q)) for q in (10, 5, 2, 1)}
    segs = speech_segments(y16)
    gaps = [(segs[i][1], segs[i + 1][0]) for i in range(len(segs) - 1) if segs[i + 1][0] - segs[i][1] >= 0.15]
    idx = [k for a, b in gaps for k in range(int(a / 0.01) + 2, int(b / 0.01) - 2) if 0 <= k < len(r)]
    out["p95-pause_median"] = float(p95 - (np.median(r[idx]) if len(idx) >= 10 else np.percentile(r, 2)))
    return out


def main():
    from synth_ba import degrade
    real = []
    for c in jl(f"{EVAL}/calib.jsonl"):
        y, sr = load_audio(c["path"]); real.append(dict(set=c["set"], **estimators(resample(y, sr, 16000))))
    for v, r in load_refs().items():
        y, sr = load_audio(r["path"]); real.append(dict(set="fleurs_ref", **estimators(resample(y, sr, 16000))))
    deg = []
    for v in REF_VOICES:
        y, sr = load_audio(load_refs()[v]["path"])
        for cond in ("clean", "snr30", "snr20", "snr15", "snr10", "snr5"):
            yd = degrade(y, sr, cond, seed_of(f"{v}{cond}"))
            deg.append(dict(voice=v, cond=cond, **estimators(resample(yd, sr, 16000))))
    keys = ["p95-p10", "p95-p5", "p95-p2", "p95-p1", "p95-pause_median"]
    summ = {}
    for k in keys:
        rv = np.array([x[k] for x in real])
        lv = {c: round(float(np.mean([d[k] for d in deg if d["cond"] == c])), 2) for c in ("clean", "snr30", "snr20", "snr15", "snr10", "snr5")}
        mx = {c: round(float(np.max([d[k] for d in deg if d["cond"] == c])), 2) for c in ("snr20", "snr15", "snr10")}
        summ[k] = dict(real_clean_min=round(float(rv.min()), 2), real_clean_p1=round(float(np.percentile(rv, 1)), 2),
                       real_clean_p5=round(float(np.percentile(rv, 5)), 2), degraded_mean=lv, degraded_max=mx,
                       margin_p5_minus_snr20max=round(float(np.percentile(rv, 5)) - mx["snr20"], 2),
                       margin_p5_minus_snr15max=round(float(np.percentile(rv, 5)) - mx["snr15"], 2))
    json.dump(dict(summary=summ, real=real, degraded=deg), open(f"{RESULTS}/reference_noise_estimators.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


def adopt():
    """Thresholds adopted for the engine, written into reference_thresholds.json under "adopted".
    Noise estimator p95-p1 (best separation above). Reject below the midpoint between the noisiest-looking
    real clean clip (min) and the cleanest-looking reference degraded to SNR 15 dB (max): no real clean clip is
    rejected, every SNR <= 15 dB reference is. Warn (voice still made, user told) below the midpoint between the
    mean of the SNR 20 dB and SNR 30 dB levels, where the pre-written rule already saw harm. Clipping: the
    pre-written rule's bound."""
    est = json.load(open(f"{RESULTS}/reference_noise_estimators.json"))
    th = json.load(open(f"{RESULTS}/reference_thresholds.json"))
    s = est["summary"]["p95-p1"]
    reject = round((s["degraded_max"]["snr15"] + s["real_clean_min"]) / 2, 1)
    warn = round((s["degraded_mean"]["snr20"] + s["degraded_mean"]["snr30"]) / 2, 1)
    real = est["real"]
    by = {}
    for x in real:
        b = by.setdefault(x["set"], dict(n=0, rejected=0, warned=0))
        b["n"] += 1; b["rejected"] += x["p95-p1"] < reject; b["warned"] += reject <= x["p95-p1"] < warn
    deg = {c: dict(n=sum(1 for d in est["degraded"] if d["cond"] == c),
                   rejected=sum(1 for d in est["degraded"] if d["cond"] == c and d["p95-p1"] < reject),
                   warned=sum(1 for d in est["degraded"] if d["cond"] == c and reject <= d["p95-p1"] < warn))
           for c in ("clean", "snr30", "snr20", "snr15", "snr10", "snr5")}
    th["adopted"] = dict(
        noise_estimator="p95-p1: 95th minus 1st percentile of 25 ms frame RMS (dB) at 16 kHz",
        REF_SNR_MIN_DB=reject, REF_SNR_WARN_DB=warn, REF_CLIP_MAX_RATIO=th["decision"]["REF_CLIP_MAX_RATIO"],
        arithmetic=dict(reject=f"({s['degraded_max']['snr15']} + {s['real_clean_min']}) / 2 = {reject}",
                        warn=f"({s['degraded_mean']['snr20']} + {s['degraded_mean']['snr30']}) / 2 = {warn}"),
        real_clean_clips=by, degraded_references=deg,
        deviation=("The pre-written rule with the Phase 1 proxy (p95-p10) gives "
                   f"{th['decision']['REF_SNR_MIN_DB']} dB, which rejects about half of the real clean calibration clips "
                   "(see decision.calibration_false_rejections); that proxy confounds few pauses with noise. The rule's "
                   "finding that SNR 20-30 dB references already lower similarity is kept as a warning, not a rejection."))
    json.dump(th, open(f"{RESULTS}/reference_thresholds.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(th["adopted"], indent=1))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "adopt":
    adopt()
