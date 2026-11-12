"""Watermark detection on real BA outputs (venv /tmp/ba/venvs/ba). -> BA/eval/results/watermark.json

Watermarked: the first 60 held-out outputs of the BA engine run (full/), as written (48 kHz WAV),
after MP3 at 128 kbit/s (ffmpeg, libmp3lame), after MP3 at 64 kbit/s, and resampled to 16 kHz.
Not watermarked (false positives): 60 raw VoxCPM2 outputs from the Phase 1 bake-off (same
utterance ids) as WAV and as MP3 128k, and 60 real speech clips (calib.jsonl).
Decision threshold: AudioSeal detection probability >= 0.5 (ba_tts.watermark.DETECT_THRESHOLD).
"""
import json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, soundfile as sf
from engine_common import ENGINE_ROOT, RESULTS, EVAL, TUNE_IDS, jl  # noqa: E402

N = 60


def mp3(y, sr, kbps):
    with tempfile.TemporaryDirectory() as d:
        a, b, c = f"{d}/a.wav", f"{d}/a.mp3", f"{d}/b.wav"
        sf.write(a, y, sr)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", a, "-c:a", "libmp3lame", "-b:a", f"{kbps}k", b], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", b, c], check=True)
        z, zsr = sf.read(c, dtype="float32")
    return (z.mean(1) if z.ndim > 1 else z), zsr


def main():
    from ba_tts.watermark import detect
    from ba_tts.audio import resample
    man = [m for m in jl(f"{ENGINE_ROOT}/full/manifest.jsonl") if not m.get("missing") and m["job"]["text_id"] not in TUNE_IDS][:N]
    res = {k: [] for k in ("ba_wav48k", "ba_mp3_128k", "ba_mp3_64k", "ba_resampled_16k",
                           "raw_voxcpm2_wav", "raw_voxcpm2_mp3_128k", "real_speech_wav")}
    for m in man:
        y, sr = sf.read(m["wav"], dtype="float32")
        res["ba_wav48k"].append(detect(y, sr))
        res["ba_mp3_128k"].append(detect(*mp3(y, sr, 128)))
        res["ba_mp3_64k"].append(detect(*mp3(y, sr, 64)))
        res["ba_resampled_16k"].append(detect(resample(y, sr, 16000), 16000))
        raw = f"/home/jovyan/ba-work/bakeoff/voxcpm2/{m['utt_id']}.wav"
        if os.path.exists(raw):
            yr, srr = sf.read(raw, dtype="float32")
            res["raw_voxcpm2_wav"].append(detect(yr, srr))
            res["raw_voxcpm2_mp3_128k"].append(detect(*mp3(yr, srr, 128)))
        print(m["utt_id"], res["ba_wav48k"][-1], res["ba_mp3_128k"][-1], flush=True)
    for c in jl(f"{EVAL}/calib.jsonl")[::4][:N]:
        y, sr = sf.read(c["path"], dtype="float32")
        res["real_speech_wav"].append(detect(y, sr))
    summ = {k: dict(n=len(v), detected=sum(1 for x in v if x["watermarked"]),
                    mean_score=round(float(np.mean([x["score"] for x in v])), 4) if v else None,
                    min_score=round(float(np.min([x["score"] for x in v])), 4) if v else None,
                    max_score=round(float(np.max([x["score"] for x in v])), 4) if v else None,
                    mean_bits_match=round(float(np.mean([x["bits_match"] for x in v])), 4) if v else None)
            for k, v in res.items()}
    out = dict(method="AudioSeal (facebook/audioseal generator_base/detector_base, 16-bit message 'BA'), threshold 0.5",
               licence="audioseal 0.2.1 on PyPI: MIT; weights facebook/audioseal: MIT", ffmpeg=subprocess.run(
                   ["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0],
               summary=summ, utterances=[m["utt_id"] for m in man], command="/tmp/ba/venvs/ba/bin/python BA/eval/scripts/watermark_check.py")
    json.dump(out, open(f"{RESULTS}/watermark.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
