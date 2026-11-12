"""Shared test matrix for the Phase 1 bake-off (stdlib only, imported by every
synthesis script and by the scorer).

profile "full"     : eligible systems (cb-v3, cb-v2, voxcpm2)
profile "egy"      : Egyptian Chatterbox fine-tunes (Egyptian + diacritised + code-switch + Egyptian passage)
profile "egy_en"   : reference-only systems (OmniVoice, VoiceTut): Egyptian sets + English + Egyptian/English passages
profile "egy_main" : the 120 main Egyptian utterances only (VoxCPM2 prompt+transcript variant)
profile "speed"    : first 24 Egyptian jobs, re-run in isolation for GPU timing (<system>-speed/)
profile "pack"     : listening pack, 6 Egyptian sentences x 2 voices, synthesised into <system>-pack/
"""
import json, os, zlib

EVAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = "/home/jovyan/ba-work/bakeoff"


def _jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def texts():
    t = {}
    for fn in ["egy", "egy_diacritised", "codeswitch", "en", "fr", "long"]:
        t[fn] = _jl(f"{EVAL}/texts/{fn}.jsonl")
    return t


def refs():
    return {r["voice_id"]: r for r in _jl(f"{EVAL}/refs.jsonl")}


PACK_TEXTS = ["egy02", "egy13", "egy21", "egy28", "egy36", "cs01"]
PACK_VOICES = ["ar_eg_m1", "ar_eg_f4"]


def seed_of(utt_id):
    return zlib.crc32(utt_id.encode()) % (2**31)


def build(profile="full"):
    T, R = texts(), refs()
    eg = sorted(v for v in R if v.startswith("ar_eg"))      # 6 voices
    en = sorted(v for v in R if v.startswith("en_us"))      # 3
    fr = sorted(v for v in R if v.startswith("fr_fr"))      # 3
    # interleave egyptian voices m/f so rotations mix genders
    m = [v for v in eg if "_m" in v]; f = [v for v in eg if "_f" in v]
    eg = [x for pair in zip(m, f) for x in pair]             # m1 f4 m2 f5 m3 f6
    jobs = []

    def add(cond, t, voice, lang):
        uid = f"{cond}__{t['id']}__{voice}"
        r = R[voice]
        jobs.append(dict(utt_id=uid, condition=cond, text_id=t["id"], lang=lang, category=t.get("category"),
                         text=t["text"], text_spoken=t.get("text_spoken"), voice_id=voice,
                         ref_path=r["path"], ref_transcript=r["transcript"], seed=seed_of(uid)))

    want = {
        "full": {"egy", "diac", "cs", "en", "fr", "cross", "long"},
        "egy": {"egy", "diac", "cs", "long_arz"},
        "egy_en": {"egy", "diac", "cs", "en", "long_arz", "long_en"},
        "egy_main": {"egy"},
        "pack": {"pack"},
        "speed": {"speed"},
    }[profile]
    if "speed" in want:  # isolated speed re-measurement: the first 24 Egyptian main jobs, written to <system>-speed/
        return [j for j in build("full") if j["condition"] == "egy"][:24]
    if "pack" in want:  # listening pack: same 6 Egyptian sentences x 2 voices for every system
        pick = {t["id"]: t for t in T["egy"] + T["codeswitch"]}
        for tid in PACK_TEXTS:
            for v in PACK_VOICES:
                add("pack", pick[tid], v, "arz")
        return jobs
    if "egy" in want:   # 40 texts x 3 of 6 voices, rotating: each voice 20 times
        for i, t in enumerate(T["egy"]):
            for k in range(3):
                add("egy", t, eg[(3 * i + k) % 6], "arz")
    if "diac" in want:  # 10 x 2 voices
        for i, t in enumerate(T["egy_diacritised"]):
            for k in range(2):
                add("diac", t, eg[(2 * i + k) % 6], "arz")
    if "cs" in want:    # 6 x 3 voices
        for i, t in enumerate(T["codeswitch"]):
            for k in range(3):
                add("cs", t, eg[(3 * i + k) % 6], "arz")
    if "en" in want:
        for t in T["en"]:
            for v in en:
                add("en", t, v, "en")
    if "fr" in want:
        for t in T["fr"]:
            for v in fr:
                add("fr", t, v, "fr")
    if "cross" in want:  # 10 en + 10 fr texts x 2 Egyptian voices
        for i, t in enumerate(T["en"][:10]):
            for k in range(2):
                add("cross_en", t, eg[(2 * i + k) % 6], "en")
        for i, t in enumerate(T["fr"][:10]):
            for k in range(2):
                add("cross_fr", t, eg[(2 * i + k + 1) % 6], "fr")
    longs = {t["lang"]: t for t in T["long"]}
    lv = {"arz": [eg[0], eg[1]], "en": [en[0], en[2]], "fr": [fr[0], fr[2]]}
    for lang in ("arz", "en", "fr"):
        if "long" in want or f"long_{lang}" in want:
            for v in lv[lang]:
                add("long", longs[lang], v, lang)
    return jobs


if __name__ == "__main__":
    import collections, sys
    for p in ["full", "egy", "egy_en", "egy_main", "pack"]:
        j = build(p)
        print(p, len(j), dict(collections.Counter(x["condition"] for x in j)))
    j = build("full")
    print(dict(collections.Counter(x["voice_id"] for x in j if x["condition"] == "egy")))
