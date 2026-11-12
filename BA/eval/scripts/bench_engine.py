"""Speed and memory of the BA engine on the A40 (venv /tmp/ba/venvs/ba). Nothing else on the GPU.

Usage: python bench_engine.py [--no-optimize]   -> BA/eval/results/bench.json (one section per mode)
  optimize    torch.compile on (the default on a plain GPU)
  no-optimize torch.compile off (as on ZeroGPU, which does not support torch.compile)
Measures, in a fresh process: model load (VoxCPM2 from the local Hub cache, incl. compile warm-up),
Whisper and watermark load, create_voice on a ~10 s reference, generation of a ~10 s Egyptian
sentence and a ~10 s English sentence (3 timed runs after one warm-up, median), the ~120-word
Egyptian and English passages (2 timed runs, median), and peak GPU memory (allocated and reserved).
Times include text normalisation, chunk checks, loudness normalisation and the watermark.
"""
import json, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from engine_common import RESULTS, _text_map  # noqa: E402

REF = "/home/jovyan/ba-work/refs/ar_eg_m1.wav"
SENT = {"arz": "لو عايز نصيحتي، متستعجلش في القرار ده خالص، اقعد مع نفسك كام يوم وفكر كويس، واسأل أبوك وأمك، وبعدين اختار اللي يريحك إنت",
        "en": "My grandmother still writes letters by hand and sends them to everyone in the family on their birthday, even though most of us reply with a text."}


def main():
    mode = "no_optimize" if "--no-optimize" in sys.argv else "optimize"
    T = _text_map()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    from ba_tts import BA
    ba = BA.from_pretrained("openbmb/VoxCPM2", device="cuda", optimize=(mode == "optimize"))
    torch.cuda.synchronize(); load_tts = time.time() - t0
    t1 = time.time(); _ = ba.asr; torch.cuda.synchronize(); load_asr = time.time() - t1
    from ba_tts import watermark as wm
    t2 = time.time(); wm._get("generator", "cuda"); load_wm = time.time() - t2
    t3 = time.time(); v = ba.create_voice(REF, language="arz"); torch.cuda.synchronize(); cv = time.time() - t3
    t3b = time.time(); v2 = ba.create_voice(REF, language="arz"); torch.cuda.synchronize(); cv2 = time.time() - t3b
    ba.speak("تجربة قصيرة.", v, "arz"); ba.speak("A short warm-up.", v, "en")
    res = dict(mode=mode, model_load_s=round(load_tts, 1), asr_load_s=round(load_asr, 1), watermark_load_s=round(load_wm, 1),
               create_voice_s_first=round(cv, 2), create_voice_s=round(cv2, 2), runs={})

    def timed(text, lang, n):
        out = []
        for _ in range(n):
            torch.cuda.synchronize(); s = time.time()
            sr, wav, info = ba.speak(text, v, lang, return_info=True)
            torch.cuda.synchronize(); el = time.time() - s
            out.append(dict(gen_s=round(el, 2), audio_s=round(len(wav) / sr, 2), rtf=round(el / (len(wav) / sr), 3),
                            chunks=len(info["chunks"]), retries=sum(len(c["attempts"]) - 1 for c in info["chunks"])))
        return dict(runs=out, median_gen_s=round(statistics.median(o["gen_s"] for o in out), 2),
                    median_audio_s=round(statistics.median(o["audio_s"] for o in out), 2),
                    median_rtf=round(statistics.median(o["rtf"] for o in out), 3), chars=len(text), words=len(text.split()))
    for lang in ("arz", "en"):
        res["runs"][f"sentence_{lang}"] = timed(SENT[lang], lang, 3)
        res["runs"][f"passage_{lang}"] = timed(T[f"long_{lang}"]["text"], lang, 2)
        print(json.dumps(res["runs"][f"passage_{lang}"]), flush=True)
    res["peak_vram_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    res["peak_vram_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
    res["gpu"] = torch.cuda.get_device_name(0); res["torch"] = torch.__version__
    p = f"{RESULTS}/bench.json"
    allr = json.load(open(p)) if os.path.exists(p) else {}
    allr[mode] = res
    s = {}
    for m, r in allr.items():
        if m == "summary" or not isinstance(r, dict) or "runs" not in r:
            continue
        s[f"{m}: model load (s)"] = r["model_load_s"]
        s[f"{m}: Whisper + watermark load (s)"] = round(r["asr_load_s"] + r["watermark_load_s"], 1)
        s[f"{m}: create_voice on a 10 s clip (s)"] = r["create_voice_s"]
        for k, v in r["runs"].items():
            s[f"{m}: {k} ({v['words']} words -> {v['median_audio_s']} s audio): time (s)"] = v["median_gen_s"]
        s[f"{m}: peak VRAM allocated / reserved (GB)"] = f"{r['peak_vram_allocated_gb']} / {r['peak_vram_reserved_gb']}"
    allr["summary"] = s
    allr["command"] = "/tmp/ba/venvs/ba/bin/python BA/eval/scripts/bench_engine.py [--no-optimize]"
    json.dump(allr, open(p, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
