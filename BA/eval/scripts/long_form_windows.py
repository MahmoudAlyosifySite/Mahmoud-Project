"""Experiment 4b: long-form error rates with windowed ASR (venv /tmp/ba/venvs/eval, GPU for ASR, then CPU).
-> BA/eval/results/long_form_windows.json, .md

Why: Whisper large-v3 transcribes audio longer than 30 s sequentially, and on one BA passage
(longv2__lv2_arz02__ar_eg_m1) it fell into a repetition loop ("ايوه" ~100 times) over the first 25 s, giving
CER 0.49; transcribing the same audio in shorter pieces returns the passage text, so the audio is correct. This
secondary analysis re-scores EVERY long-form clip of BOTH systems the same way, so the comparison stays fair.
The primary metric (long_form.py, sequential long-form ASR as fixed in the protocol) is reported unchanged.

Windowing: cut points about every 20 s, each at the quietest 50 ms frame between 16 and 24 s after the previous
cut (so cuts fall in pauses); every window (< 30 s) is transcribed with the language forced and no timestamps,
the window texts are joined with spaces, and CER (Egyptian) or WER (English, French) is computed after the
protocol's normalisation (score.py err_rates) against the passage text. Paired BA - VoxCPM2 differences on the
same passage-voice cells, two-way bootstrap (paper_common.py).

Run: /tmp/ba/venvs/eval/bin/python BA/eval/scripts/long_form_windows.py       (ASR cache: ba-work/paper/long_windows.jsonl)
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from paper_common import ENGINE_ROOT, PAPER_WORK, RESULTS, jl, engine_rows, two_way_bootstrap, paired, ci, dump

SYSTEMS = {"ba": "p_long_ba", "voxcpm2": "p_long_voxcpm2"}
CACHE = f"{PAPER_WORK}/long_windows.jsonl"
WHISPER_LANG = {"arz": "arabic", "en": "english", "fr": "french"}


def cut_points(y, sr=16000, lo=16.0, hi=24.0):
    cuts, start, n = [0], 0, len(y)
    frame = int(0.05 * sr)
    while n - start > hi * sr:
        a, b = start + int(lo * sr), start + int(hi * sr)
        seg = y[a:b][: (b - a) // frame * frame].reshape(-1, frame)
        k = int(np.argmin((seg ** 2).mean(1)))
        start = a + k * frame + frame // 2
        cuts.append(start)
    cuts.append(n)
    return cuts


def transcribe_all():
    import torch
    from transformers import pipeline
    from score import load16
    done = {r["wav"] for r in jl(CACHE)} if os.path.exists(CACHE) else set()
    todo = []
    for s, exp in SYSTEMS.items():
        for m in jl(f"{ENGINE_ROOT}/{exp}/manifest.jsonl"):
            if m["wav"] not in done:
                todo.append((s, m))
    print(f"{len(todo)} clips to transcribe", flush=True)
    if not todo:
        return
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", torch_dtype=torch.float16, device=0)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "a", encoding="utf-8") as f:
        for i, (s, m) in enumerate(todo):
            y = load16(m["wav"])
            cuts = cut_points(y)
            texts = []
            for a, b in zip(cuts[:-1], cuts[1:]):
                gk = {"language": WHISPER_LANG[m["lang"]], "task": "transcribe"}
                texts.append(asr({"raw": y[a:b], "sampling_rate": 16000}, generate_kwargs=gk)["text"].strip())
            rec = dict(system=s, utt_id=m["utt_id"], wav=m["wav"], lang=m["lang"], ref_text=m["ref_text"],
                       windows=[round((b - a) / 16000, 2) for a, b in zip(cuts[:-1], cuts[1:])], asr=" ".join(texts))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            if i % 10 == 0:
                print(i, s, m["utt_id"], rec["windows"], flush=True)


def main():
    if "--no-asr" not in sys.argv:
        transcribe_all()
    from score import err_rates
    cache = {r["wav"]: r for r in jl(CACHE)}
    data = {}
    for s, exp in SYSTEMS.items():
        rows = engine_rows(exp, s)
        for r in rows:
            c = cache.get(r["wav"])
            cer, wer = err_rates(c["ref_text"], c["asr"], c["lang"]) if c else (None, None)
            r["err_win"] = cer if r["lang"] == "arz" else wer
            r["err_seq"] = r.get("err")
        data[s] = rows
    out = {"design": __doc__, "systems": {}, "paired_ba_minus_voxcpm2": {}, "largest_changes": []}
    for s, rows in data.items():
        out["systems"][s] = {}
        for lang in ["arz", "en", "fr"]:
            rl = [r for r in rows if r["lang"] == lang]
            out["systems"][s][lang] = {k: two_way_bootstrap([r[k] for r in rl], [r["text_id"] for r in rl], [r["voice_id"] for r in rl])
                                       for k in ("err_seq", "err_win")}
    for lang in ["arz", "en", "fr"]:
        a = [r for r in data["ba"] if r["lang"] == lang]
        b = [r for r in data["voxcpm2"] if r["lang"] == lang]
        out["paired_ba_minus_voxcpm2"][lang] = {k: paired(a, b, k) for k in ("err_seq", "err_win")}
    allr = [dict(system=s, utt_id=r["utt_id"], err_seq=r["err_seq"], err_win=r["err_win"]) for s, rows in data.items() for r in rows]
    out["largest_changes"] = sorted(allr, key=lambda d: -abs((d["err_seq"] or 0) - (d["err_win"] or 0)))[:6]
    out["command"] = "/tmp/ba/venvs/eval/bin/python BA/eval/scripts/long_form_windows.py"
    dump(out, f"{RESULTS}/long_form_windows.json")
    L = ["# Long-form error rates: sequential vs windowed ASR", "",
         "Generated by `BA/eval/scripts/long_form_windows.py` from `long_form_windows.json`. CER for Egyptian, WER for "
         "English and French; 95% two-way bootstrap CIs. Sequential = the protocol's metric (long_form.md); windowed = "
         "~20 s windows cut at pauses, same normalisation.", "",
         "| system | lang | sequential ASR | windowed ASR |", "|---|---|---|---|"]
    for s, d in out["systems"].items():
        for lang, r in d.items():
            L.append(f"| {s} | {lang} | {ci(r['err_seq'])} | {ci(r['err_win'])} |")
    L += ["", "Paired BA - VoxCPM2 [95% CI] (bootstrap p):", ""]
    for lang, d in out["paired_ba_minus_voxcpm2"].items():
        for k, t in d.items():
            if t:
                L.append(f"- {lang} {k}: {t['diff']:+.4f} [{t['ci_lo']:+.4f}, {t['ci_hi']:+.4f}] (p {t['p_bootstrap']})")
    L += ["", "Largest sequential - windowed changes:", ""]
    for d in out["largest_changes"]:
        L.append(f"- {d['system']} {d['utt_id']}: {d['err_seq']:.3f} -> {d['err_win']:.3f}")
    with open(f"{RESULTS}/long_form_windows.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
