"""VoxCPM2 synthesis (PyPI voxcpm, venv /tmp/ba/venvs/vox). openbmb/VoxCPM2, Apache-2.0.

Systems:
  voxcpm2         plain reference cloning: generate(text, reference_wav_path=ref)
  voxcpm2-prompt  "ultimate cloning": reference_wav_path=ref + prompt_wav_path=ref + prompt_text=<FLEURS transcript>
                  (Egyptian main set only, profile egy_main)
Package defaults otherwise (cfg_value=2.0, inference_timesteps=10, retry_badcase=True,
normalize=False, denoise=False). The ModelScope ZipEnhancer denoiser is not loaded
(load_denoiser=False; it is only used when denoise=True, which is not the default).
No language tag exists for this model.

Usage: python synth_voxcpm.py <system> <profile> [limit] [--cpu-bench] [--no-optimize]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from matrix import build
from synth_common import run_jobs


def load(device, optimize=True):
    from voxcpm import VoxCPM
    return VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=optimize, device=device)


def make_fn(system, model):
    sr = model.tts_model.sample_rate

    def fn(job):
        if system == "voxcpm2-prompt":
            wav = model.generate(text=job["text"], reference_wav_path=job["ref_path"],
                                 prompt_wav_path=job["ref_path"], prompt_text=job["ref_transcript"])
        else:
            wav = model.generate(text=job["text"], reference_wav_path=job["ref_path"])
        return wav, sr
    return fn


def cpu_bench(system):
    torch.set_num_threads(2)
    model = load("cpu", optimize=False)
    fn = make_fn(system, model)
    jobs = [j for j in build("full") if j["text_id"] in ("egy16", "en03", "fr06") and j["condition"] in ("egy", "en", "fr")]
    seen, recs = set(), []
    for j in jobs:
        if j["text_id"] in seen: continue
        seen.add(j["text_id"])
        torch.manual_seed(j["seed"])
        t0 = time.time(); wav, sr = fn(j); el = time.time() - t0
        recs.append(dict(system=system, utt_id=j["utt_id"], gen_sec=round(el, 2), audio_sec=round(len(wav) / sr, 2),
                         rtf=round(el / (len(wav) / sr), 3), threads=torch.get_num_threads()))
        print(recs[-1], flush=True)
    os.makedirs("/home/jovyan/ba-work/bakeoff/cpu", exist_ok=True)
    with open(f"/home/jovyan/ba-work/bakeoff/cpu/{system}.json", "w") as f:
        json.dump(recs, f, indent=1)


if __name__ == "__main__":
    system, profile = sys.argv[1], sys.argv[2]
    if "--cpu-bench" in sys.argv:
        cpu_bench(system); sys.exit()
    args = [a for a in sys.argv[3:] if not a.startswith("--")]
    limit = int(args[0]) if args else None
    model = load("cuda", optimize="--no-optimize" not in sys.argv)
    run_jobs(system, build(profile), make_fn(system, model), limit=limit, out_name=system + (f"-{profile}" if profile in ("pack", "speed") else ""))
