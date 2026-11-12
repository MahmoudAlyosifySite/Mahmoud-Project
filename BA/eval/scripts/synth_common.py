"""Helpers shared by the synthesis scripts: resumable loop, timing, VRAM, per-utterance meta log."""
import json, os, time, traceback
import numpy as np
import soundfile as sf


def run_jobs(system, jobs, synth_fn, device="cuda", limit=None, out_name=None):
    """synth_fn(job) -> (np.float32 mono wav, sample_rate). Writes
    /home/jovyan/ba-work/bakeoff/<system>/<utt_id>.wav and appends one JSON line per
    utterance to .../<system>/meta.jsonl (resumable: existing wavs are skipped)."""
    import torch
    from matrix import OUT_ROOT
    out = f"{OUT_ROOT}/{out_name or system}"
    os.makedirs(out, exist_ok=True)
    meta_p = f"{out}/meta.jsonl"
    done = set()
    if os.path.exists(meta_p):
        for l in open(meta_p):
            done.add(json.loads(l)["utt_id"])
    if limit:
        jobs = jobs[:limit]
    n = 0
    todo = [j for j in jobs if j["utt_id"] not in done]
    if todo:  # one unrecorded warm-up call so the first timed utterance is not inflated
        try:
            synth_fn(todo[0])
        except Exception:
            traceback.print_exc()
    for job in jobs:
        if job["utt_id"] in done:
            continue
        torch.manual_seed(job["seed"]); np.random.seed(job["seed"] % (2**32))
        if device == "cuda":
            torch.cuda.manual_seed_all(job["seed"]); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time(); err = None; dur = None
        try:
            wav, sr = synth_fn(job)
            if device == "cuda":
                torch.cuda.synchronize()
            el = time.time() - t0
            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            dur = len(wav) / sr
            sf.write(f"{out}/{job['utt_id']}.wav", wav, sr)
        except Exception as e:
            el = time.time() - t0
            err = f"{type(e).__name__}: {e}"[:500]
            traceback.print_exc()
        rec = dict(system=system, utt_id=job["utt_id"], seed=job["seed"], gen_sec=round(el, 3),
                   audio_sec=None if dur is None else round(dur, 3),
                   rtf=None if not dur else round(el / dur, 4), sr=None if err else sr,
                   peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3) if device == "cuda" else None,
                   error=err)
        with open(meta_p, "a") as f:
            f.write(json.dumps(rec) + "\n")
        n += 1
        print(f"[{system}] {n} {job['utt_id']} audio={rec['audio_sec']} rtf={rec['rtf']} err={err}", flush=True)
