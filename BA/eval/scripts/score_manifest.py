"""Score a BA-engine experiment with the Phase 1 scorer (venv /tmp/ba/venvs/eval).

Usage: python score_manifest.py <exp> [<exp> ...]
Reads /home/jovyan/ba-work/engine/<exp>/manifest.jsonl (written by synth_ba.py) and appends one
row per utterance to /home/jovyan/ba-work/engine/<exp>/scores.jsonl (resumable). Metrics are
exactly those of score.py (Whisper large-v3 CER/WER, ECAPA and WavLM similarity against the
clean FLEURS reference clip, dialect ID for Arabic text, DNSMOS, durations).
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import Scorer, score_one, load_done  # noqa: E402

ROOT = "/home/jovyan/ba-work/engine"


def main():
    S = None
    for exp in sys.argv[1:]:
        man = [json.loads(l) for l in open(f"{ROOT}/{exp}/manifest.jsonl", encoding="utf-8")]
        outp = f"{ROOT}/{exp}/scores.jsonl"
        done = load_done(outp)
        todo = [m for m in man if m["utt_id"] not in done]
        print(f"[{exp}] {len(man)} rows, {len(todo)} to score", flush=True)
        if not todo:
            continue
        if S is None:
            S = Scorer()
        with open(outp, "a") as f:
            for m in todo:
                rec = dict(key=m["utt_id"], exp=exp)
                if m.get("missing") or not os.path.exists(m["wav"]):
                    rec["missing"] = True
                else:
                    try:
                        rec.update(score_one(S, m["wav"], m["lang"], m["ref_text"], m.get("ref_text_alt"),
                                             m["ref_voice_path"], do_dialect=m.get("dialect", False)))
                    except Exception as e:
                        rec["score_error"] = repr(e)[:300]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                print(exp, m["utt_id"], rec.get("cer"), rec.get("wer"), rec.get("sim_ecapa"), rec.get("dialect"), flush=True)


if __name__ == "__main__":
    main()
