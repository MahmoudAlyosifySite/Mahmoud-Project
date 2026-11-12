"""Synthesis with the BA engine (venv /tmp/ba/venvs/ba, package ba-tts) for every Phase 2 experiment.

Usage: python synth_ba.py <exp> [--limit N]
  ref    reference-quality thresholds: FLEURS references degraded with pink noise (SNR 30..5 dB)
         or hard clipping (+6/+12/+20 dB gain); cloning quality measured against the clean original
  cs     code-switch policy: texts/codeswitch_tune.jsonl x 3 Egyptian voices x {keep, arabic}
  chunk  long-text chunk length: tuning passages x 2 voices x max_chars {150, 300, 1000}
  sweep  cfg/steps: tuning texts (engine_common.TUNE_*) x their Phase 1 voices x 6 settings
  full   final evaluation: the full Phase 1 matrix (324 utterances) with the engine defaults
Each job writes <ENGINE_ROOT>/<exp>/<utt_id>.wav and appends to meta.jsonl (resumable) and
manifest.jsonl (input for score_manifest.py). Voices are built once per (voice, condition) with
create_voice (automatic Whisper transcript: the product path) and kept in memory only.
"""
import json, os, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, soundfile as sf, torch
from engine_common import (ENGINE_ROOT, CHUNK_SETTINGS, REF_CONDITIONS, REF_TEXTS, REF_VOICES, SWEEP, TUNE_EGY,
                           TUNE_EN, _text_map, build, jl, job, load_refs, seed_of, tuning_jobs, tuning_passage)


def pink_noise(n, rng):
    """1/f noise by spectral shaping of white noise."""
    w = rng.standard_normal(n)
    f = np.fft.rfft(w)
    k = np.arange(len(f)); k[0] = 1
    p = np.fft.irfft(f / np.sqrt(k), n)
    return (p / np.std(p)).astype(np.float32)


def degrade(y, sr, cond, seed):
    from ba_tts.audio import frame_rms_db, resample
    if cond == "clean":
        return y
    if cond.startswith("snr"):
        snr = float(cond[3:])
        y16 = resample(y, sr, 16000)
        r = frame_rms_db(y16)
        act = r > np.percentile(r, 95) - 30          # speech frames
        sp_pow = float(np.mean((10 ** (r[act] / 20)) ** 2))
        n = pink_noise(len(y), np.random.default_rng(seed))
        n *= np.sqrt(sp_pow / 10 ** (snr / 10))
        return (y + n).astype(np.float32)
    if cond.startswith("clip"):
        g = 10 ** (float(cond[4:]) / 20)
        peak = float(np.max(np.abs(y)))
        return np.clip(y / peak * 0.9 * g, -1.0, 1.0).astype(np.float32)
    raise ValueError(cond)


def build_jobs(exp):
    T = _text_map()
    R = load_refs()
    jobs = []
    if exp == "ref":
        for v in REF_VOICES:
            lang = "arz" if v.startswith("ar_eg") else "en"
            for cond in REF_CONDITIONS:
                for tid in REF_TEXTS[lang]:
                    uid = f"ref__{cond}__{tid}__{v}"
                    jobs.append(job("ref", uid, T[tid], lang, v, ref_cond=cond))
    elif exp == "cs":
        eg = ["ar_eg_m1", "ar_eg_f4", "ar_eg_m2", "ar_eg_f5", "ar_eg_m3", "ar_eg_f6"]
        rows = jl(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "texts/codeswitch_tune.jsonl"))
        for i, t in enumerate(rows):
            for k in range(3):
                v = eg[(3 * i + k) % 6]
                for pol in ("keep", "arabic"):
                    jobs.append(job("cs", f"cs__{pol}__{t['id']}__{v}", t, "arz", v, latin=pol))
    elif exp == "chunk":
        for lang, voices in (("arz", ["ar_eg_m1", "ar_eg_f4"]), ("en", ["en_us_m1", "en_us_f3"])):
            row = tuning_passage(lang)
            for mc in CHUNK_SETTINGS:
                for v in voices:
                    jobs.append(job("chunk", f"chunk__{mc}__{row['id']}__{v}", row, lang, v, max_chunk_chars=mc))
    elif exp == "sweep":
        for cfg, steps in SWEEP:
            tag = f"cfg{cfg}_s{steps}"
            for j in tuning_jobs(tag, cfg_value=cfg, inference_timesteps=steps):
                j["utt_id"] = f"sweep__{tag}__{j['utt_id']}"
                jobs.append(j)
    elif exp == "full":
        for j in build("full"):
            jj = dict(j); jj["exp"] = "full"
            jobs.append(jj)
    else:
        raise SystemExit(f"unknown experiment {exp}")
    return jobs


def main():
    exp = sys.argv[1]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    jobs = build_jobs(exp)[:limit] if limit else build_jobs(exp)
    out = f"{ENGINE_ROOT}/{exp}"
    os.makedirs(out, exist_ok=True)
    done = {r["utt_id"] for r in jl(f"{out}/meta.jsonl")}
    todo = [j for j in jobs if j["utt_id"] not in done]
    print(f"[{exp}] {len(jobs)} jobs, {len(todo)} to do", flush=True)
    if not todo:
        return
    from ba_tts import BA, ReferenceAudioError
    import ba_tts.defaults as D
    if exp == "ref":   # thresholds are what this experiment measures: accept every reference
        D.REF_SNR_MIN_DB = -1e9; D.REF_CLIP_MAX_RATIO = 1.0
    t0 = time.time()
    ba = BA.from_pretrained("openbmb/VoxCPM2", device="cuda")
    load_s = time.time() - t0
    voices = {}
    ref_info_p = f"{out}/voices.jsonl"
    ref_logged = {(r["voice_id"], r.get("ref_cond", "clean")) for r in jl(ref_info_p)}
    from ba_tts.audio import load_audio
    from ba_tts.reference import analyse
    warm = True
    for j in todo:
        key = (j["voice_id"], j.get("ref_cond", "clean"))
        if key not in voices:
            y, sr = load_audio(j["ref_path"])
            yd = degrade(y, sr, key[1], seed_of(f"{key[0]}{key[1]}"))
            tv = time.time()
            try:
                v = ba.create_voice((sr, yd), language="arz" if j["voice_id"].startswith("ar_eg") else j["voice_id"][:2])
                err = None
            except ReferenceAudioError as e:
                v, err = None, f"ReferenceAudioError[{e.code}]"
            rec = analyse((sr, yd))
            voices[key] = v
            if key not in ref_logged:
                with open(ref_info_p, "a") as f:
                    f.write(json.dumps(dict(voice_id=key[0], ref_cond=key[1], snr_proxy_db=round(rec.snr_db, 2),
                                            clip_ratio=rec.raw_clip_ratio, speech_s=round(rec.speech_s, 2),
                                            transcript=None if v is None else v.transcript, mode=None if v is None else v.mode,
                                            report=None if v is None else v.report, error=err,
                                            create_voice_s=round(time.time() - tv, 3)), ensure_ascii=False) + "\n")
        v = voices[key]
        if warm and v is not None:   # one unrecorded warm-up call (as in Phase 1)
            ba.speak("warm up", v, "en", watermark=True); warm = False
        kw = {k: j[k] for k in ("cfg_value", "inference_timesteps", "max_chunk_chars") if k in j}
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t1 = time.time(); err = None; info = None; dur = None
        try:
            if v is None:
                raise RuntimeError("no voice (reference rejected)")
            if "latin" in j:
                from ba_tts.text import arz
                D.CODESWITCH_POLICY = j["latin"]
            sr_out, wav, info = ba.speak(j["text"], v, j["lang"], seed=j["seed"], return_info=True, **kw)
            torch.cuda.synchronize()
            dur = len(wav) / sr_out
            sf.write(f"{out}/{j['utt_id']}.wav", wav, sr_out)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:300]
            traceback.print_exc()
        el = time.time() - t1
        rec = dict(system="ba", exp=exp, utt_id=j["utt_id"], seed=j["seed"], gen_sec=round(el, 3),
                   audio_sec=None if dur is None else round(dur, 3), rtf=None if not dur else round(el / dur, 4),
                   peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3), error=err,
                   n_chunks=None if info is None else len(info["chunks"]),
                   n_retries=None if info is None else sum(len(c["attempts"]) - 1 for c in info["chunks"]),
                   chunks_failed=None if info is None else sum(1 for c in info["chunks"] if not c["ok"]),
                   normalized_text=None if info is None else info["normalized_text"],
                   voice_mode=None if v is None else v.mode, model_load_s=round(load_s, 1))
        with open(f"{out}/meta.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        man = dict(utt_id=j["utt_id"], wav=f"{out}/{j['utt_id']}.wav", lang=j["lang"],
                   ref_text=j.get("text_spoken") or j["text"], ref_text_alt=j["text"] if j.get("text_spoken") else None,
                   ref_voice_path=j["ref_path"], dialect=j["lang"] == "arz", missing=err is not None,
                   job={k: j[k] for k in j if k not in ("text",)})
        with open(f"{out}/manifest.jsonl", "a") as f:
            f.write(json.dumps(man, ensure_ascii=False) + "\n")
        print(f"[{exp}] {j['utt_id']} audio={rec['audio_sec']} rtf={rec['rtf']} retries={rec['n_retries']} err={err}", flush=True)


if __name__ == "__main__":
    main()
