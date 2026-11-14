"""Run the model card's quick-start snippet in a clean environment and keep what is needed to score it.

Run with the clean venv's interpreter, isolated (-I), so ba_tts comes from the installed wheel:

    <venv>/bin/python -I BA/hf/validate_snippet.py <out_dir> [--private]

--private reads the owner's token from ~/.config/ba/hf_token inside this process and passes it as
``token=`` (the repository is private at that point); without it nothing is authenticated.
Reference voices are FLEURS clips (measurement only, never published). Writes <out_dir>/*.wav,
manifest.jsonl (for BA/hf/score_samples.py) and run.json.
"""
import json
import os
import sys
import time

REFS = "/home/jovyan/Mahmoud-Project/BA/eval/refs.jsonl"
# held-out texts (not in the tuning split of BA/eval/scripts/engine_common.py)
JOBS = [
    ("arz_m1_egy02", "ar_eg_m1", "arz", "ماما عاملة ملوخية النهارده وعايزاك تيجي تتغدى معانا", None),
    ("arz_m1_egy03", "ar_eg_m1", "arz", "الجو حر أوي النهارده، مش قادر أنزل من البيت", None),
    ("arz_f6_egy27", "ar_eg_f6", "arz", "الحساب جه 350 جنيه، وده غير البقشيش", "الحساب جه تلتمية وخمسين جنيه، وده غير البقشيش"),
    ("arz_f6_egy02", "ar_eg_f6", "arz", "ماما عاملة ملوخية النهارده وعايزاك تيجي تتغدى معانا", None),
    ("en_m1_en02", "en_us_m1", "en", "I left my umbrella on the train this morning and got completely soaked on the way to work.", None),
    ("fr_m1_fr02", "fr_fr_m1", "fr", "Est-ce que tu sais à quelle heure ferme la pharmacie ce soir ?", None),
]


def main():
    out = sys.argv[1]
    private = "--private" in sys.argv
    os.makedirs(out, exist_ok=True)
    refs = {json.loads(l)["voice_id"]: json.loads(l) for l in open(REFS, encoding="utf-8")}
    token = None
    if private:
        with open(os.path.expanduser("~/.config/ba/hf_token"), encoding="utf-8") as f:
            token = f.read().strip()

    # ---- the quick-start snippet (plus token= while the repository is private) ----
    import soundfile as sf
    import ba_tts
    from ba_tts import BA, detect_watermark
    t0 = time.time()
    ba = BA.from_pretrained("mahmoudalyosify/BA", token=token) if private else BA.from_pretrained("mahmoudalyosify/BA")
    load_s = time.time() - t0
    run = dict(ba_tts_file=ba_tts.__file__, ba_tts_version=ba_tts.__version__, private=private,
               hf_home=os.environ.get("HF_HOME"), hf_token_env=bool(os.environ.get("HF_TOKEN")),
               device=ba.device, model_dir=ba.tts.tts_model.__class__.__name__,
               ba_config_loaded=ba.config is not None,
               ba_config_version=(ba.config or {}).get("ba_version"),
               base_revision=((ba.config or {}).get("base_model") or {}).get("revision"),
               asr=(ba.asr_model, ba.asr_revision), load_seconds=round(load_s, 1), jobs=[])
    import importlib.metadata as md
    run["packages"] = {p: md.version(p) for p in ("torch", "voxcpm", "transformers", "audioseal", "huggingface_hub")}
    from huggingface_hub import scan_cache_dir
    run["hub_cache"] = sorted(f"{r.repo_id}@{','.join(sorted(rv.commit_hash[:12] for rv in r.revisions))}"
                              for r in scan_cache_dir().repos)
    manifest = []
    voices = {}
    for utt, vid, lang, text, spoken in JOBS:
        r = refs[vid]
        t1 = time.time()
        if vid not in voices:
            voices[vid] = ba.create_voice(r["path"], language=lang)
        voice = voices[vid]
        sr, wav, info = ba.speak(text, voice, language=lang, return_info=True)
        el = time.time() - t1
        path = f"{out}/{utt}.wav"
        sf.write(path, wav, sr)
        det = detect_watermark(wav, sr)
        run["jobs"].append(dict(utt_id=utt, voice=vid, lang=lang, sr=sr, seconds=info["seconds"], wall_seconds=round(el, 2),
                                engine_checks_ok=all(c["ok"] for c in info["chunks"]), voice_mode=voice.mode,
                                voice_transcript=voice.transcript, normalized_text=info["normalized_text"], watermark=det))
        manifest.append(dict(utt_id=utt, wav=path, lang=lang, ref_text=spoken or text,
                             ref_text_alt=text if spoken else None, ref_voice_path=r["path"], dialect=(lang == "arz"),
                             engine_checks_ok=all(c["ok"] for c in info["chunks"])))
        print(utt, sr, info["seconds"], "s", det, flush=True)
    with open(f"{out}/manifest.jsonl", "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    json.dump(run, open(f"{out}/run.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in run.items() if k != "jobs"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
