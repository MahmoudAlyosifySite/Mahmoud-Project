"""Check the reference-transcription ASR (venv /tmp/ba/venvs/ba).

Runs ba_tts.asr.Transcriber (openai/whisper-large-v3-turbo, MIT) on the Phase 1 real-speech
calibration clips (calib.jsonl: 50 colloquial Egyptian lahgtna_eg, 50 NileTTS, 50 FLEURS ar_eg,
30 en_us, 30 fr_fr), with the language given and with language detection, and compares CER/WER
with openai/whisper-large-v3 on the same clips (Phase 1 scores, _calib.jsonl). Same text
normalisation as score.py. Writes BA/eval/results/asr_check.json.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from score import err_rates, load16  # noqa: E402
from engine_common import EVAL, RESULTS, jl  # noqa: E402

CALIB_SCORES = "/home/jovyan/ba-work/bakeoff/scores/_calib.jsonl"


def main():
    from ba_tts.asr import Transcriber
    from ba_tts.defaults import ASR_MODEL
    t0 = time.time(); tr = Transcriber(ASR_MODEL, device="cuda"); load_s = time.time() - t0
    v3 = {r["key"]: r for r in jl(CALIB_SCORES)}
    rows = jl(f"{EVAL}/calib.jsonl")
    out = []
    for r in rows:
        y = load16(r["path"])
        t1 = time.time(); txt, _ = tr(y, r["lang"]); el = time.time() - t1
        txt_auto, det = tr(y, None)
        c, w = err_rates(r["text"], txt, r["lang"])
        out.append(dict(calib_id=r["calib_id"], set=r["set"], lang=r["lang"], turbo=txt, turbo_cer=c, turbo_wer=w,
                        detected_lang=det, sec=round(el, 3), audio_s=round(len(y) / 16000, 2),
                        v3=v3.get(r["calib_id"], {}).get("asr"), v3_cer=v3.get(r["calib_id"], {}).get("cer"),
                        v3_wer=v3.get(r["calib_id"], {}).get("wer")))
        print(r["calib_id"], round(c, 3), out[-1]["v3_cer"], det, flush=True)
    summ = {}
    for s in sorted({o["set"] for o in out}):
        oo = [o for o in out if o["set"] == s]
        exp_lang = oo[0]["lang"]
        summ[s] = dict(n=len(oo), turbo_cer=round(float(np.mean([o["turbo_cer"] for o in oo])), 4),
                       turbo_wer=round(float(np.mean([o["turbo_wer"] for o in oo])), 4),
                       v3_cer=round(float(np.mean([o["v3_cer"] for o in oo if o["v3_cer"] is not None])), 4),
                       v3_wer=round(float(np.mean([o["v3_wer"] for o in oo if o["v3_wer"] is not None])), 4),
                       language_detected_correctly=round(float(np.mean([o["detected_lang"] == exp_lang for o in oo])), 4),
                       turbo_sec_per_clip=round(float(np.mean([o["sec"] for o in oo])), 3),
                       mean_clip_s=round(float(np.mean([o["audio_s"] for o in oo])), 2))
    res = dict(model=ASR_MODEL, licence="MIT (model card of openai/whisper-large-v3-turbo)", compared_with="openai/whisper-large-v3 (Phase 1 scorer)",
               device="NVIDIA A40, fp16", load_s=round(load_s, 1), summary=summ, utterances=out,
               command="/tmp/ba/venvs/ba/bin/python BA/eval/scripts/asr_check.py")
    json.dump(res, open(f"{RESULTS}/asr_check.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
