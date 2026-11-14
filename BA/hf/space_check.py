"""Call a running BA Space app through gradio_client and keep what is needed to score it.

    python BA/hf/space_check.py <url or Space id> <out_dir>

1. consent unticked -> must be refused (an error, no audio);
2. consent ticked, FLEURS Egyptian reference (measurement only, never published), no transcript,
   Egyptian text with digits -> audio; 3. a second request in the same session (voice reused).
Writes <out_dir>/*.wav, manifest.jsonl (for BA/hf/score_samples.py) and space_check.json
(timings, refusal message, watermark detection).
"""
import json
import os
import shutil
import sys
import time

REFS = "/home/jovyan/Mahmoud-Project/BA/eval/refs.jsonl"
REF = "ar_eg_f6"
TEXTS = ["القطر هيقوم الساعة 7:30 الصبح من محطة مصر، والتذكرة ب 150 جنيه.",
         "أنا مبسوط أوي إنك جيت، وحشتني والله."]
SPOKEN = ["القطر هيقوم الساعة سبعة ونص الصبح من محطة مصر، والتذكرة بمية وخمسين جنيه.", TEXTS[1]]


def main():
    from gradio_client import Client, handle_file
    target, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    ref = {json.loads(l)["voice_id"]: json.loads(l) for l in open(REFS, encoding="utf-8")}[REF]["path"]
    c = Client(target, verbose=False)
    api = c.view_api(return_format="dict", print_info=False)
    res = dict(target=target, api_endpoints=list(api.get("named_endpoints", {}).keys()))
    t0 = time.time()
    try:
        r = c.predict(handle_file(ref), [], "", False, TEXTS[0], "arz", 1234, 0, api_name="/generate")
        res["consent_unticked"] = f"GENERATED (WRONG): {r}"
    except Exception as e:
        res["consent_unticked"] = f"refused: {str(e)[:200]}"
    res["consent_unticked_seconds"] = round(time.time() - t0, 2)
    man = []
    for i, text in enumerate(TEXTS):
        t0 = time.time()
        r = c.predict(handle_file(ref), [], "", True, text, "arz", 1234, 0, api_name="/generate")
        el = time.time() - t0
        path = r if isinstance(r, str) else (r[0] if isinstance(r, (list, tuple)) else r.get("path"))
        if isinstance(path, dict):
            path = path.get("path") or path.get("name")
        dst = f"{out}/space_{i}.wav"
        shutil.copyfile(path, dst)
        res[f"request_{i}_seconds"] = round(el, 2)
        man.append(dict(utt_id=f"space_{i}", wav=dst, lang="arz", ref_text=SPOKEN[i],
                        ref_text_alt=TEXTS[i] if SPOKEN[i] != TEXTS[i] else None, ref_voice_path=ref, dialect=True))
    with open(f"{out}/manifest.jsonl", "w", encoding="utf-8") as f:
        for m in man:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    import soundfile as sf
    from ba_tts import detect_watermark
    res["watermark"] = [detect_watermark(*sf.read(m["wav"], dtype="float32")) for m in man]
    res["sample_rates"] = [sf.info(m["wav"]).samplerate for m in man]
    json.dump(res, open(f"{out}/space_check.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
