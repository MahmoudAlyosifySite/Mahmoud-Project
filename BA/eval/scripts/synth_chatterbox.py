"""Chatterbox family synthesis (PyPI chatterbox-tts 0.1.7, venv /tmp/ba/venvs/cb).

Systems:
  cb-v2    ResembleAI/chatterbox multilingual V2, PyPI defaults (repetition_penalty=2.0)
  cb-v3    same model with T3 replaced by t3_mtl23ls_v3.safetensors; upstream V3 inference
           details reproduced: repetition_penalty=1.2, the final speech token's audio
           (960 samples = 40 ms at 24 kHz) is dropped (here: after the watermark is applied), and
           NO AlignmentStreamAnalyzer: upstream master's t3.py (read, not run) no longer creates it,
           while PyPI 0.1.7 does and forces EOS whenever the last two speech tokens are equal.
           Disabled here by swapping in a pass-through analyzer.
  cb-v3-pypi  V3 weights + repetition_penalty=1.2 + last-token trim, but PyPI 0.1.7's analyzer left
           on (the first run; kept as a diagnostic variant)
  namaa    T3 from NAMAA-Space/NAMAA-Egyptian-TTS t3_mtl23ls_v2.safetensors (as on its card), PyPI defaults
  lahgtna  T3 from oddadmix/lahgtna-chatterbox-v1 t3_mtl23ls_v2.safetensors, PyPI defaults
All other sampling settings are the package defaults. language_id: ar / en / fr.

Usage: python synth_chatterbox.py <system> <profile> [limit] [--cpu-bench]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from matrix import build
from synth_common import run_jobs

LANG = {"arz": "ar", "en": "en", "fr": "fr"}
T3 = {"cb-v3": ("ResembleAI/chatterbox", "t3_mtl23ls_v3.safetensors"),
      "cb-v3-pypi": ("ResembleAI/chatterbox", "t3_mtl23ls_v3.safetensors"),
      "namaa": ("NAMAA-Space/NAMAA-Egyptian-TTS", "t3_mtl23ls_v2.safetensors"),
      "lahgtna": ("oddadmix/lahgtna-chatterbox-v1", "t3_mtl23ls_v2.safetensors")}


class _NoAnalyzer:
    """Pass-through stand-in for PyPI's AlignmentStreamAnalyzer (upstream V3 code has none)."""
    def __init__(self, tfmr, queue, text_tokens_slice, alignment_layer_idx=9, eos_idx=0):
        self.eos_idx = eos_idx

    def step(self, logits, next_token=None):
        return logits


def load(system, device):
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    if system == "cb-v3":
        import chatterbox.models.t3.t3 as t3mod
        t3mod.AlignmentStreamAnalyzer = _NoAnalyzer
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    if system in T3:
        repo, fn = T3[system]
        sd = load_file(hf_hub_download(repo, fn))
        if "model" in sd.keys():
            sd = sd["model"][0]
        missing, unexpected = model.t3.load_state_dict(sd, strict=True), None
        model.t3.to(device).eval()
    return model


def make_fn(system, model):
    v3 = system in ("cb-v3", "cb-v3-pypi")
    kw = {"repetition_penalty": 1.2} if v3 else {}

    def fn(job):
        wav = model.generate(job["text"], language_id=LANG[job["lang"]], audio_prompt_path=job["ref_path"], **kw)
        wav = wav.squeeze(0).cpu().numpy()
        if v3:
            wav = wav[:-960] if len(wav) > 960 else wav
        return wav, model.sr
    return fn


def cpu_bench(system):
    """RTF on CPU with 2 threads (run under `taskset -c 0,1`) on 3 short sentences."""
    torch.set_num_threads(2)
    model = load(system, "cpu")
    fn = make_fn(system, model)
    jobs = [j for j in build("full") if j["text_id"] in ("egy16", "en03", "fr06")
            and j["condition"] in ("egy", "en", "fr")]
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
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    model = load(system, "cuda")
    run_jobs(system, build(profile), make_fn(system, model), limit=limit, out_name=system + (f"-{profile}" if profile in ("pack", "speed") else ""))
