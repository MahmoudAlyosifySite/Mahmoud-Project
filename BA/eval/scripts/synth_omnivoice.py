"""Reference-only systems (non-commercial weights; never a BA base). venv /tmp/ba/venvs/omni,
PyPI omnivoice 0.2.1 and voicetut-tts 0.1.1.

Systems:
  omnivoice  k2-fsa/OmniVoice (CC-BY-NC weights). OmniVoice.from_pretrained(fp16, as the package CLI),
             model.generate(text, language=<arz|en>, ref_audio=<ref>, ref_text=<FLEURS transcript>),
             package default generation config, normalize_text=False (default).
  voicetut   mohammedaly22/VoiceTut-TTS (fine-tune of OmniVoice). VoiceTutTTS.from_pretrained (fp16 default),
             synthesize(text, ref_audio, ref_text, language) with its defaults, including its
             Egyptian text normaliser (normalize=True is its default).
Language ids: Egyptian texts -> "arz", English -> "en".

Usage: python synth_omnivoice.py <system> <profile> [limit]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from matrix import build
from synth_common import run_jobs

LANG = {"arz": "arz", "en": "en"}


def load(system):
    if system == "omnivoice":
        from omnivoice.models.omnivoice import OmniVoice
        m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda", dtype=torch.float16)
        def fn(job):
            out = m.generate(text=job["text"], language=LANG[job["lang"]], ref_audio=job["ref_path"],
                             ref_text=job["ref_transcript"])
            return out[0], m.sampling_rate
        return fn
    if system == "voicetut":
        from voicetut_tts import VoiceTutTTS
        tts = VoiceTutTTS.from_pretrained("mohammedaly22/VoiceTut-TTS", device="cuda")
        def fn(job):
            w = tts.synthesize(job["text"], ref_audio=job["ref_path"], ref_text=job["ref_transcript"],
                               language=LANG[job["lang"]])
            return w, tts.model.sampling_rate
        return fn
    raise ValueError(system)


if __name__ == "__main__":
    system, profile = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    run_jobs(system, build(profile), load(system), limit=limit, out_name=system + (f"-{profile}" if profile in ("pack", "speed") else ""))
