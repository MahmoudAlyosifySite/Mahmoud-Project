"""End-to-end test of the Space app through gradio_client (venv /tmp/ba/venvs/ba).

Start the app first (bound to localhost), then: python space_e2e.py <url>
1. consent unticked -> the app must refuse (error) and generate nothing;
2. a FLEURS Egyptian reference clip, no transcript, Egyptian text with digits -> audio;
3. the same session again with new text (voice reused from session memory);
4. checks that Gradio's copy of the uploaded clip was deleted and lists files left in GRADIO_TEMP_DIR.
Writes /home/jovyan/ba-work/engine/space/ (wav + manifest.jsonl for score_manifest.py space) and
BA/eval/results/space_e2e.json.
"""
import glob, json, os, shutil, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_common import RESULTS, load_refs, ENGINE_ROOT  # noqa: E402

REF = "ar_eg_f6"
TEXTS = ["القطر هيقوم الساعة 7:30 الصبح من محطة مصر، والتذكرة ب 150 جنيه.",
         "أنا مبسوط أوي إنك جيت، وحشتني والله."]
SPOKEN = ["القطر هيقوم الساعة سبعة ونص الصبح من محطة مصر، والتذكرة بمية وخمسين جنيه.", TEXTS[1]]


def main():
    from gradio_client import Client, handle_file
    url = sys.argv[1]
    tmp_dir = os.environ.get("GRADIO_TEMP_DIR", "/tmp/ba/gradio_tmp")
    out_dir = f"{ENGINE_ROOT}/space"; os.makedirs(out_dir, exist_ok=True)
    ref = load_refs()[REF]["path"]
    c = Client(url, verbose=False)
    api = c.view_api(return_format="dict", print_info=False)
    res = dict(api_endpoints=list(api.get("named_endpoints", {}).keys()))
    # 1. consent refused
    try:
        c.predict(handle_file(ref), [], "", False, TEXTS[0], "arz", 1234, 0, api_name="/generate")
        res["consent_unticked"] = "generated (WRONG)"
    except Exception as e:
        res["consent_unticked"] = f"refused: {str(e)[:160]}"
    # 2-3. generation, then a second request in the same session (voice reused)
    man = []
    for i, text in enumerate(TEXTS):
        t0 = time.time()
        r = c.predict(handle_file(ref), [], "", True, text, "arz", 1234, 0, api_name="/generate")
        el = time.time() - t0
        path = r if isinstance(r, str) else (r[0] if isinstance(r, (list, tuple)) else r.get("path"))
        dst = f"{out_dir}/space_e2e_{i}.wav"; shutil.copyfile(path, dst)
        res[f"request_{i}_seconds"] = round(el, 2)
        man.append(dict(utt_id=f"space_e2e_{i}", wav=dst, lang="arz", ref_text=SPOKEN[i],
                        ref_text_alt=TEXTS[i] if SPOKEN[i] != TEXTS[i] else None, ref_voice_path=ref, dialect=True,
                        missing=False, job=dict(text_id=f"space{i}", voice_id=REF, lang="arz")))
    with open(f"{out_dir}/manifest.jsonl", "w") as f:
        for m in man:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    left = [p for p in glob.glob(f"{tmp_dir}/**/*", recursive=True) if os.path.isfile(p)]
    res["uploaded_reference_copies_left_in_gradio_cache"] = [p for p in left if p.endswith(os.path.basename(ref))]
    res["files_left_in_gradio_cache"] = len(left)
    res["file_types_left"] = sorted({os.path.splitext(p)[1] for p in left})
    from ba_tts.watermark import detect
    import soundfile as sf
    res["watermark"] = [detect(*sf.read(m["wav"], dtype="float32")) for m in man]
    json.dump(res, open(f"{RESULTS}/space_e2e.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
