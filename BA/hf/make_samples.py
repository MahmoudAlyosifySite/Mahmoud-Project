"""Audio samples for the model card, from synthetic voices only (decision D7). Venv /tmp/ba/venvs/ba, GPU.

1. VoxCPM2 voice design makes two synthetic Egyptian voices (a man and a woman) from a short text
   description. Selection rule for the design seed, fixed before the samples were made: the first seed
   in DESIGN_SEEDS whose clip is 8-12 s long and whose clip the Phase 1 dialect classifier labels
   Egyptian (``--candidates`` writes the candidate clips, BA/hf/score_samples.py labels them).
2. The reference is loudness-normalised and watermarked like any BA output.
3. BA clones each synthetic voice (no transcript given: the reference is transcribed by Whisper, the
   product's default path) and speaks 2 Egyptian, 1 English and 1 French sentence.
4. Everything is written as WAV (for scoring) and MP3 128 kbit/s (for the card) to OUT, with
   manifest.jsonl (for BA/hf/score_samples.py) and samples.json (generation details).

No FLEURS or other third-party voice is used anywhere in this script.

    python BA/hf/make_samples.py [--model openbmb/VoxCPM2]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

OUT = "/home/jovyan/ba-work/publish/samples"
VOICES = {
    "synthetic_man": dict(
        description="An Egyptian man in his thirties, warm, calm voice, speaking Egyptian Arabic",
        text="صباح الخير يا جماعة. أنا بحب أبدأ يومي بفنجان قهوة على البلكونة، وبعدين أنزل أتمشى شوية "
             "على الكورنيش قبل ما أروح الشغل."),
    "synthetic_woman": dict(
        description="An Egyptian woman in her thirties, clear, friendly voice, speaking Egyptian Arabic",
        text="أهلاً بيكم. النهارده الجو حلو أوي، فقررت أعمل أكلة بسيطة في البيت وأعزم صحابي على الغدا، "
             "وبعدين نقعد نتكلم ونضحك."),
}
SENTENCES = [
    ("arz1", "arz", "أنا رايح الشغل دلوقتي وهرجع على المغرب إن شاء الله."),
    ("arz2", "arz", "الميعاد اتغير للساعة 7:30، والتذكرة بقت ب 150 جنيه."),
    ("en", "en", "The meeting has been moved to Thursday afternoon, so please update your calendar."),
    # (first French choice, "... ferme plus tôt les jours fériés", replaced: Whisper writes the homophone
    # "plutôt" for "plus tôt", so its WER says nothing about the audio)
    ("fr", "fr", "Le concert commence à 21h, mais il vaut mieux arriver un peu en avance."),
]
REF_MIN_S, REF_MAX_S = 8.0, 12.0
DESIGN_SEEDS = [11, 23, 37, 41, 53, 67, 79, 97, 101, 113]


def mp3(wav_path, mp3_path):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path, "-codec:a", "libmp3lame", "-b:a", "128k",
                    mp3_path], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openbmb/VoxCPM2")
    ap.add_argument("--candidates", action="store_true", help="only write the design candidates (8-12 s) for scoring")
    ap.add_argument("--seeds", default="", help="chosen design seeds, e.g. synthetic_man=41,synthetic_woman=11")
    a = ap.parse_args()
    chosen = dict(kv.split("=") for kv in a.seeds.split(",") if kv)
    from ba_tts import BA, detect_watermark
    from ba_tts import defaults as D
    from ba_tts import watermark as wm
    from ba_tts.audio import normalize_loudness, trim_edges
    from ba_tts.engine import _seed_all
    os.makedirs(OUT, exist_ok=True)
    ba = BA.from_pretrained(a.model, device="cuda", optimize=False)
    sr = ba.sample_rate

    def design(v, seed):
        _seed_all(seed)
        y = ba.tts.generate(text=f"({v['description']}){v['text']}", cfg_value=2.0, inference_timesteps=10)
        return trim_edges(np.asarray(y, dtype=np.float32), sr)

    if a.candidates:
        cdir = f"{OUT}/candidates"
        os.makedirs(cdir, exist_ok=True)
        rows = []
        for vid, v in VOICES.items():
            for seed in DESIGN_SEEDS:
                y = design(v, seed)
                dur = len(y) / sr
                print(vid, seed, round(dur, 2), flush=True)
                if REF_MIN_S <= dur <= REF_MAX_S:
                    path = f"{cdir}/{vid}_s{seed}.wav"
                    sf.write(path, y, sr)
                    rows.append(dict(utt_id=f"{vid}_s{seed}", wav=path, lang="arz", ref_text=v["text"], ref_text_alt=None,
                                     ref_voice_path=None, dialect=True, kind="candidate", voice=vid))
                else:
                    rows.append(dict(utt_id=f"{vid}_s{seed}", skipped=f"{dur:.2f} s outside {REF_MIN_S}-{REF_MAX_S} s"))
        with open(f"{cdir}/manifest_all.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(f"{cdir}/manifest.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                if "wav" in r:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return

    summary, manifest = dict(model=a.model, voices={}, design_seed_rule=__doc__.split("1. ")[1].split("2. ")[0].strip()), []
    for vid, v in VOICES.items():
        seed = int(chosen[vid])
        y, _ = sf.read(f"{OUT}/candidates/{vid}_s{seed}.wav", dtype="float32")   # the clip that was scored
        tries = [dict(seed=seed, seconds=round(len(y) / sr, 2))]
        y = normalize_loudness(y, sr, D.OUTPUT_LUFS)
        y = wm.embed(y, sr, device="cuda")
        peak = float(np.max(np.abs(y)))
        if peak > 0.999:
            y = y * (0.999 / peak)
        ref_wav = f"{OUT}/{vid}_reference.wav"
        sf.write(ref_wav, y, sr)
        mp3(ref_wav, ref_wav[:-4] + ".mp3")
        voice = ba.create_voice(ref_wav, language="arz")
        info_v = dict(description=v["description"], design_text=v["text"], design_seed=seed, design_tries=tries,
                      reference_seconds=round(len(y) / sr, 2), transcript=voice.transcript, mode=voice.mode,
                      reference_report=voice.report, samples={})
        manifest.append(dict(utt_id=f"{vid}_reference", wav=ref_wav, lang="arz", ref_text=v["text"], ref_text_alt=None,
                             ref_voice_path=None, dialect=True, kind="reference", voice=vid))
        for sid, lang, text in SENTENCES:
            sr2, wav, info = ba.speak(text, voice, lang, return_info=True)
            path = f"{OUT}/{vid}_{sid}.wav"
            sf.write(path, wav, sr2)
            mp3(path, path[:-4] + ".mp3")
            ok = all(c["ok"] for c in info["chunks"])
            info_v["samples"][sid] = dict(language=lang, text=text, normalized_text=info["normalized_text"],
                                          engine_checks_ok=ok, chunks=info["chunks"], seconds=info["seconds"],
                                          gen_seconds=info["gen_seconds"], seed=info["seed"])
            spoken = info["normalized_text"] if lang == "arz" else text
            manifest.append(dict(utt_id=f"{vid}_{sid}", wav=path, lang=lang, ref_text=spoken,
                                 ref_text_alt=text if spoken != text else None, ref_voice_path=ref_wav,
                                 dialect=(lang == "arz"), kind="sample", voice=vid, engine_checks_ok=ok))
            print(vid, sid, "ok" if ok else "FAILED engine checks", info["seconds"], "s", flush=True)
        summary["voices"][vid] = info_v
    # watermark on the MP3s actually published
    import librosa
    det = {}
    for m in manifest:
        p = m["wav"][:-4] + ".mp3"
        yy, ss = librosa.load(p, sr=None, mono=True)
        det[os.path.basename(p)] = detect_watermark(yy, ss, device="cuda")
    summary["watermark_mp3"] = det
    with open(f"{OUT}/manifest.jsonl", "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    json.dump(summary, open(f"{OUT}/samples.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(det, indent=1))


if __name__ == "__main__":
    main()
