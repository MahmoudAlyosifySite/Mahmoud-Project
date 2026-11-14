"""Score audio with the Phase 1 scorer (BA/eval/scripts/score.py; venv /tmp/ba/venvs/eval, GPU).

    python BA/hf/score_samples.py <manifest.jsonl> <out.json>

Manifest rows: utt_id, wav, lang, ref_text, ref_text_alt (or null), ref_voice_path (or null), dialect (bool).
Writes one record per row: CER/WER (Whisper large-v3), ECAPA/WavLM similarity to ref_voice_path,
dialect label (Arabic rows), DNSMOS, durations; plus the G3 intelligibility bound for that language.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval", "scripts"))
from score import Scorer, score_one  # noqa: E402

# G3 intelligibility bounds (execution-plan.md section 4; Egyptian = real colloquial ASR CER 0.0712 + 0.10, engine.md)
BOUND = {"arz": ("cer", 0.1712), "en": ("wer", 0.08), "fr": ("wer", 0.10)}


def main():
    man = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
    S = Scorer()
    out = []
    for m in man:
        rec = dict(utt_id=m["utt_id"], lang=m["lang"])
        rec.update(score_one(S, m["wav"], m["lang"], m["ref_text"], m.get("ref_text_alt"), m.get("ref_voice_path"),
                             do_dialect=m.get("dialect", False)))
        metric, bound = BOUND[m["lang"]]
        vals = [rec.get(metric)] + ([rec.get(f"{metric}_vs_digit_text")] if rec.get(f"{metric}_vs_digit_text") is not None else [])
        rec["error_best"] = min(v for v in vals if v is not None)
        rec["error_metric"], rec["g3_bound"] = metric, bound
        rec["within_g3_bound"] = rec["error_best"] <= bound
        for k in ("engine_checks_ok", "kind", "voice"):
            if k in m:
                rec[k] = m[k]
        out.append(rec)
        print(json.dumps({k: rec.get(k) for k in ("utt_id", "asr", "error_best", "sim_ecapa", "dialect", "mos_p808",
                                                  "within_g3_bound")}, ensure_ascii=False), flush=True)
    json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
