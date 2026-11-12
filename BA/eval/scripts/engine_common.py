"""Phase 2 (BA engine) evaluation: tuning/held-out split and experiment job lists.

The split is fixed here before any tuning run, so defaults are chosen on TUNE texts and final
numbers are reported on the rest only.

  Egyptian tuning texts : egy11 egy14 egy17 egy20 egy23 egy26 egy29 egy32 egy35 egy38
                          (every third text from egy11: 2 everyday, 3 questions, 2 emotion, 2 digits, 2 long;
                           none of egy01-egy10, so the diacritised set stays held out)
  English tuning texts  : en11 en12 en14 en16 en18 en20 (none of en01-en10, so cross_en stays held out)
  French                : never tuned (takes the English defaults)
  Code-switch           : policy chosen on texts/codeswitch_tune.jsonl (new sentences); the 6 evaluation
                          code-switch sentences stay held out
  Long passages         : chunk length chosen on passages built from the tuning sentences;
                          texts/long.jsonl stays held out
Voices: as in the Phase 1 matrix (matrix.py). Seeds: crc32(utt_id) as in Phase 1.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matrix import build, texts as load_texts, refs as load_refs, seed_of, EVAL  # noqa: E402

ENGINE_ROOT = "/home/jovyan/ba-work/engine"
RESULTS = f"{EVAL}/results"
TUNE_EGY = ["egy11", "egy14", "egy17", "egy20", "egy23", "egy26", "egy29", "egy32", "egy35", "egy38"]
TUNE_EN = ["en11", "en12", "en14", "en16", "en18", "en20"]
TUNE_IDS = set(TUNE_EGY) | set(TUNE_EN)

SWEEP = [(1.5, 10), (2.0, 10), (2.5, 10), (3.0, 10), (2.0, 20), (2.5, 20)]
CHUNK_SETTINGS = [150, 300, 1000]
REF_CONDITIONS = ["clean", "snr30", "snr20", "snr15", "snr10", "snr5", "clip6", "clip12", "clip20"]
REF_VOICES = ["ar_eg_m1", "ar_eg_f4", "ar_eg_m2", "en_us_m1", "en_us_f3", "en_us_m2"]
REF_TEXTS = {"arz": ["egy11", "egy17", "egy23"], "en": ["en12", "en16", "en18"]}


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def _text_map():
    T = load_texts()
    m = {}
    for k in ("egy", "egy_diacritised", "codeswitch", "en", "fr", "long"):
        for t in T[k]:
            m[t["id"]] = t
    for t in jl(f"{EVAL}/texts/codeswitch_tune.jsonl"):
        m[t["id"]] = t
    return m


def job(exp, utt_id, text_row, lang, voice_id, **settings):
    R = load_refs()
    r = R[voice_id]
    return dict(exp=exp, utt_id=utt_id, text_id=text_row["id"], lang=lang, category=text_row.get("category"),
                text=text_row["text"], text_spoken=text_row.get("text_spoken"), voice_id=voice_id,
                ref_path=r["path"], seed=seed_of(utt_id), **settings)


def tuning_jobs(settings_tag, **settings):
    """The sweep's tuning set: matrix jobs (same voices and seeds as Phase 1) for the tuning texts."""
    out = []
    for j in build("full"):
        if j["condition"] in ("egy", "en") and j["text_id"] in TUNE_IDS:
            jj = dict(j); jj.update(exp="sweep", setting=settings_tag, **settings)
            out.append(jj)
    return out


def tuning_passage(lang):
    """A long passage built from the tuning sentences (digits kept; spoken form for scoring)."""
    T = _text_map()
    ids = TUNE_EGY if lang == "arz" else TUNE_EN
    sep = " "
    text = sep.join(T[i]["text"].rstrip(".،!؟?") + ("." if lang != "arz" else ".") for i in ids)
    spoken = sep.join((T[i].get("text_spoken") or T[i]["text"]).rstrip(".،!؟?") + "." for i in ids)
    return dict(id=f"tune_passage_{lang}", text=text, text_spoken=spoken if spoken != text else None, category="passage")


def held_out(job_or_row):
    return job_or_row["text_id"] not in TUNE_IDS
