"""Synthesis for the paper experiments (step A of BA/paper/PLAN.md). One experiment per process; the
venv depends on the system (ba: BA engine; vox: raw VoxCPM2; cb: Chatterbox family).

Usage: python synth_paper.py <exp> [--limit N]

  seeds_ba / seeds_voxcpm2 / seeds_cbv3 / seeds_namaa
      Egyptian held-out main + digit sentences (matrix.build("full"), condition egy, tuning texts excluded:
      24 + 6 texts x their 3 Phase 1 voices = 90 jobs) with two NEW seeds each: utt_id "<phase-1 utt_id>#s1"
      and "#s2", seed = crc32(utt_id) as everywhere else. Seed index 0 is the existing run (engine/full for BA,
      the bake-off for the others), so every system has 3 seeds per utterance.
  abl_nonorm   BA without the number/symbol normaliser. Egyptian text: ba_tts.text.arz.normalize with
               normalize_numbers, ABBREV and SYMBOLS switched off (the loanword policy stays on);
               English text: no normalisation (only ba_tts.text.clean).
  abl_noloan   BA with Latin words in Egyptian text kept in Latin script (latin="keep"); everything else default.
  abl_notrans  BA in reference-only mode: the voice is prepared exactly as in BA, but its transcript is not used
               (speak(mode="reference")).
  abl_rawref   BA without reference preparation: the voice is the raw FLEURS clip resampled to 16 kHz (no speech
               trimming, window choice, pause shortening, loudness normalisation or quality checks); its
               transcript is the automatic Whisper transcript of that raw clip, normalised as create_voice does.
      Ablation jobs: held-out Egyptian main, digits, code-switch and English (condition egy, cs, en of the
      Phase 1 matrix; 72 + 18 + 18 + 42 = 150), same utt_ids, voices and seeds as the BA run (engine/full).
      When an ablation leaves the text the model receives unchanged (identical to engine/full's
      normalized_text) and the voice is unchanged (abl_nonorm, abl_noloan), the generation is the same
      computation as in engine/full, so that output is reused (hard link) and flagged reused_from_full.
  long_ba / long_voxcpm2
      texts/long_v2.jsonl (new held-out passages) x 2 voices per language (arz: ar_eg_m1, ar_eg_f4; en:
      en_us_m1, en_us_f3; fr: fr_fr_m1, fr_fr_f3, the Phase 1 passage voices), seed = crc32(utt_id).

Output: /home/jovyan/ba-work/engine/p_<exp>/<utt_id>.wav, meta.jsonl and manifest.jsonl in the format of
synth_ba.py, so score_manifest.py p_<exp> and analyze_engine.load("p_<exp>") work unchanged. Resumable.
Raw VoxCPM2 and Chatterbox use synth_voxcpm / synth_chatterbox (load, make_fn) and synth_common.run_jobs.
"""
import json, os, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_common import ENGINE_ROOT, EVAL, TUNE_IDS, jl, load_refs, seed_of  # noqa: E402
from matrix import build  # noqa: E402

LONG_VOICES = {"arz": ["ar_eg_m1", "ar_eg_f4"], "en": ["en_us_m1", "en_us_f3"], "fr": ["fr_fr_m1", "fr_fr_f3"]}
BA_SYSTEM = {"seeds_ba", "abl_nonorm", "abl_noloan", "abl_notrans", "abl_rawref", "long_ba"}


def seed_jobs():
    base = [j for j in build("full") if j["condition"] == "egy" and j["text_id"] not in TUNE_IDS]
    out = []
    for k in (1, 2):
        for j in base:
            uid = f"{j['utt_id']}#s{k}"
            out.append(dict(j, utt_id=uid, seed=seed_of(uid), seed_index=k, base_utt_id=j["utt_id"]))
    return out


def ablation_jobs():
    return [dict(j, seed_index=0) for j in build("full") if j["condition"] in ("egy", "cs", "en") and j["text_id"] not in TUNE_IDS]


def long_jobs():
    R = load_refs()
    out = []
    for t in jl(f"{EVAL}/texts/long_v2.jsonl"):
        for v in LONG_VOICES[t["lang"]]:
            uid = f"longv2__{t['id']}__{v}"
            out.append(dict(utt_id=uid, condition="long_v2", text_id=t["id"], lang=t["lang"], category="passage",
                            text=t["text"], text_spoken=t.get("text_spoken"), voice_id=v, ref_path=R[v]["path"],
                            ref_transcript=R[v]["transcript"], seed=seed_of(uid), seed_index=0, words=t.get("words")))
    return out


def jobs_for(exp):
    if exp.startswith("seeds_"):
        return seed_jobs()
    if exp.startswith("abl_"):
        return ablation_jobs()
    if exp.startswith("long_"):
        return long_jobs()
    raise SystemExit(f"unknown experiment {exp}")


def write_manifest(out, jobs, system):
    meta = {r["utt_id"]: r for r in jl(f"{out}/meta.jsonl")}
    with open(f"{out}/manifest.jsonl.tmp", "w", encoding="utf-8") as f:
        for j in jobs:
            m = meta.get(j["utt_id"])
            if m is None:
                continue
            f.write(json.dumps(dict(utt_id=j["utt_id"], wav=f"{out}/{j['utt_id']}.wav", lang=j["lang"],
                                    ref_text=j.get("text_spoken") or j["text"], ref_text_alt=j["text"] if j.get("text_spoken") else None,
                                    ref_voice_path=j["ref_path"], dialect=j["lang"] == "arz", missing=m.get("error") is not None,
                                    job=dict({k: j[k] for k in j if k != "text"}, system=system)), ensure_ascii=False) + "\n")
    os.replace(f"{out}/manifest.jsonl.tmp", f"{out}/manifest.jsonl")


# ------------------------------------------------------------------ raw VoxCPM2 / Chatterbox
def run_baseline(exp, jobs, limit):
    from synth_common import run_jobs
    system = {"seeds_voxcpm2": "voxcpm2", "long_voxcpm2": "voxcpm2", "seeds_cbv3": "cb-v3", "seeds_namaa": "namaa"}[exp]
    out = f"{ENGINE_ROOT}/p_{exp}"
    os.makedirs(out, exist_ok=True)
    if system == "voxcpm2":
        import synth_voxcpm as S
        model = S.load("cuda", optimize=True)
        fn = S.make_fn(system, model)
    else:
        import synth_chatterbox as S
        model = S.load(system, "cuda")
        fn = S.make_fn(system, model)
    # run_jobs writes under matrix.OUT_ROOT/<out_name>; a relative name puts the files in ENGINE_ROOT/p_<exp>
    run_jobs(system, jobs, fn, limit=limit, out_name=os.path.relpath(out, "/home/jovyan/ba-work/bakeoff"))
    write_manifest(out, jobs, system)


# ------------------------------------------------------------------ BA engine variants
def ablated_text(exp, text, lang):
    """The text the model receives under an ablation (None = the engine's own normalisation)."""
    from ba_tts.text import clean, normalize
    import ba_tts.text.arz as A
    if exp == "abl_nonorm":
        if lang != "arz":
            return clean(text)
        saved = (A.normalize_numbers, A.ABBREV, A.SYMBOLS)
        A.normalize_numbers, A.ABBREV, A.SYMBOLS = (lambda t: t), [], []
        try:
            return normalize(text, "arz")
        finally:
            A.normalize_numbers, A.ABBREV, A.SYMBOLS = saved
    if exp == "abl_noloan":
        return normalize(text, lang, latin="keep") if lang == "arz" else normalize(text, lang)
    return None


def run_ba(exp, jobs, limit):
    import numpy as np, soundfile as sf, torch
    out = f"{ENGINE_ROOT}/p_{exp}"
    os.makedirs(out, exist_ok=True)
    done = {r["utt_id"] for r in jl(f"{out}/meta.jsonl")}
    todo = [j for j in jobs if j["utt_id"] not in done][:limit] if limit else [j for j in jobs if j["utt_id"] not in done]
    print(f"[{exp}] {len(jobs)} jobs, {len(todo)} to do", flush=True)
    full_meta = {r["utt_id"]: r for r in jl(f"{ENGINE_ROOT}/full/meta.jsonl")}
    if not todo:
        write_manifest(out, jobs, "ba"); return
    # reuse pass first (no model needed)
    from ba_tts.text import clean
    if exp in ("abl_nonorm", "abl_noloan"):
        rest = []
        for j in todo:
            t = ablated_text(exp, j["text"], j["lang"])
            fm = full_meta.get(j["utt_id"])
            if fm and not fm.get("error") and clean(t) == fm["normalized_text"]:
                dst = f"{out}/{j['utt_id']}.wav"
                if not os.path.exists(dst):
                    os.link(f"{ENGINE_ROOT}/full/{j['utt_id']}.wav", dst)
                rec = dict(fm, exp=f"p_{exp}", reused_from_full=True, input_text=clean(t))
                with open(f"{out}/meta.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                rest.append(j)
        print(f"[{exp}] reused {len(todo) - len(rest)} unchanged inputs from engine/full; {len(rest)} to generate", flush=True)
        todo = rest
    if not todo:
        write_manifest(out, jobs, "ba"); return
    from ba_tts import BA, ReferenceAudioError
    from ba_tts.audio import load_audio, resample
    from ba_tts.engine import _end_punct
    from ba_tts.text import normalize
    from ba_tts.voice import Voice
    t0 = time.time()
    ba = BA.from_pretrained("openbmb/VoxCPM2", device="cuda")
    load_s = time.time() - t0
    voices, vlog = {}, f"{out}/voices.jsonl"
    logged = {r["voice_id"] for r in jl(vlog)}
    warm = True
    for j in todo:
        vid = j["voice_id"]
        vlang = "arz" if vid.startswith("ar_eg") else vid[:2]
        if vid not in voices:
            y, sr = load_audio(j["ref_path"])
            tv = time.time()
            if exp == "abl_rawref":
                y16 = resample(y, sr, 16000).astype(np.float32)
                tr, _ = ba.asr(y16, vlang)
                tr = _end_punct(normalize(tr, vlang)) if tr and len(tr.strip()) >= 2 else None
                v = Voice(audio=y16, sample_rate=16000, transcript=tr, language=vlang,
                          report=dict(mode="raw_clip", seconds=round(len(y16) / 16000, 2), transcript_source="asr (raw clip)"))
            else:
                v = ba.create_voice((sr, y), language=vlang)
            voices[vid] = v
            if vid not in logged:
                with open(vlog, "a", encoding="utf-8") as f:
                    f.write(json.dumps(dict(voice_id=vid, transcript=v.transcript, mode=v.mode, seconds=round(v.seconds, 2),
                                            report=v.report, create_voice_s=round(time.time() - tv, 3)), ensure_ascii=False) + "\n")
        v = voices[vid]
        if warm:
            ba.speak("warm up", v, "en", watermark=True); warm = False
        kw = dict(seed=j["seed"], return_info=True)
        text = j["text"]
        t = ablated_text(exp, j["text"], j["lang"])
        if t is not None:
            text, kw["normalize_text"] = t, False
        if exp == "abl_notrans":
            kw["mode"] = "reference"
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t1 = time.time(); err = None; info = None; dur = None
        try:
            sr_out, wav, info = ba.speak(text, v, j["lang"], **kw)
            torch.cuda.synchronize()
            dur = len(wav) / sr_out
            sf.write(f"{out}/{j['utt_id']}.wav", wav, sr_out)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:300]
            traceback.print_exc()
        el = time.time() - t1
        rec = dict(system="ba", exp=f"p_{exp}", utt_id=j["utt_id"], seed=j["seed"], gen_sec=round(el, 3),
                   audio_sec=None if dur is None else round(dur, 3), rtf=None if not dur else round(el / dur, 4),
                   peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3), error=err,
                   n_chunks=None if info is None else len(info["chunks"]),
                   n_retries=None if info is None else sum(len(c["attempts"]) - 1 for c in info["chunks"]),
                   chunks_failed=None if info is None else sum(1 for c in info["chunks"] if not c["ok"]),
                   normalized_text=None if info is None else info["normalized_text"],
                   voice_mode=None if info is None else info["mode"], model_load_s=round(load_s, 1), reused_from_full=False)
        if info is not None and exp.startswith("long_"):
            rec["chunks"] = [dict(chars=len(c["text"]), ok=c["ok"], attempts=len(c["attempts"])) for c in info["chunks"]]
        with open(f"{out}/meta.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{exp}] {j['utt_id']} audio={rec['audio_sec']} rtf={rec['rtf']} retries={rec['n_retries']} err={err}", flush=True)
    write_manifest(out, jobs, "ba")


def main():
    exp = sys.argv[1]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    jobs = jobs_for(exp)
    if exp in BA_SYSTEM:
        run_ba(exp, jobs, limit)
    else:
        run_baseline(exp, jobs, limit)


if __name__ == "__main__":
    main()
