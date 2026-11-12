"""ba-tts command line.

    ba-tts --ref a.wav [--ref b.wav] [--ref-text "..."] --lang arz --text "..." --out out.wav
    ba-tts --detect out.wav        # check for the BA watermark
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ba-tts", description="BA voice cloning (Egyptian Arabic, English, French).")
    p.add_argument("--ref", action="append", help="reference recording of the voice (repeatable)")
    p.add_argument("--ref-text", help="what is said in the reference recording(s); transcribed automatically if omitted")
    p.add_argument("--ref-lang", help="language of the reference recording (arz/en/fr); detected if omitted")
    p.add_argument("--lang", help="language of the text: arz (Egyptian Arabic), en, fr")
    p.add_argument("--text", help="text to speak")
    p.add_argument("--text-file", help="read the text from a UTF-8 file")
    p.add_argument("--out", default="out.wav", help="output WAV path")
    p.add_argument("--model", default="openbmb/VoxCPM2", help="Hub repo id or local directory")
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cfg", type=float, default=None, help="guidance scale (default: per-language)")
    p.add_argument("--steps", type=int, default=None, help="diffusion steps (default: per-language)")
    p.add_argument("--no-watermark", action="store_true", help="do not watermark the output")
    p.add_argument("--save-voice", help="also save the voice profile to this .npz file")
    p.add_argument("--voice", help="load a saved voice profile (.npz) instead of --ref")
    p.add_argument("--detect", metavar="AUDIO", help="only check AUDIO for the BA watermark")
    p.add_argument("--info", action="store_true", help="print generation details as JSON")
    a = p.parse_args(argv)

    import soundfile as sf
    if a.detect:
        from .audio import load_audio
        from .watermark import detect
        y, sr = load_audio(a.detect)
        print(json.dumps(detect(y, sr, device=a.device)))
        return 0
    if not a.lang or not (a.text or a.text_file) or not (a.ref or a.voice):
        p.error("--lang, --text (or --text-file) and --ref (or --voice) are required")
    text = a.text if a.text else open(a.text_file, encoding="utf-8").read()
    import torch
    from . import BA, ReferenceAudioError, Voice
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ba = BA.from_pretrained(a.model, device=device)
    try:
        voice = Voice.load(a.voice) if a.voice else ba.create_voice(a.ref, transcript=a.ref_text, language=a.ref_lang)
    except ReferenceAudioError as e:
        print(str(e), file=sys.stderr)
        return 2
    if a.save_voice:
        print("voice saved to", voice.save(a.save_voice), file=sys.stderr)
    sr, wav, info = ba.speak(text, voice, a.lang, seed=a.seed, cfg_value=a.cfg, inference_timesteps=a.steps,
                             watermark=not a.no_watermark, return_info=True)
    sf.write(a.out, wav, sr)
    if a.info:
        info["voice"] = dict(mode=voice.mode, transcript=voice.transcript, language=voice.language, report=voice.report)
        print(json.dumps(info, ensure_ascii=False, indent=1))
    print(a.out, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
