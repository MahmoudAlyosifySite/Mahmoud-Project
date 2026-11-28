"""Dialect features in the outputs: how the letters jiim (ج) and qaaf (ق) of the Egyptian sentences are pronounced.

Egyptian Arabic pronounces jiim as [g], where MSA and the Levantine and Gulf dialects have [dʒ] or [ʒ]; it pronounces
qaaf as a glottal stop [ʔ], where MSA has [q] (Levantine urban speech also has [ʔ], Gulf speech mostly [g]). Both are
textbook markers of the dialect. A multilingual phone recogniser (wav2vec2 XLSR-53 fine-tuned on CommonVoice
phonemes, espeak phone set; pinned revision) transcribes every clip, and per set of clips we count

  jiim as [g]   recognised [ɡ] / ([ɡ] + [dʒ] + [ʒ]), over the clips whose text contains jiim;
  qaaf as [q]   recognised [q] per qaaf letter of the text, over the clips whose text contains qaaf.

The counts are not aligned to the letters, so they are measured on real speech too (calibration, as for every other
measure in the paper): real colloquial Egyptian (lahgtna-v2 test clips), Egyptian speakers reading mostly MSA text
(FLEURS ar_eg) and synthetic colloquial speech (NileTTS). Clips: the bake-off outputs of every configuration on the 32
non-digit Egyptian sentences (three voices each) and, on the held-out sentences only, BA and plain VoxCPM2. Intervals:
two-way bootstrap over sentences and voices (the same draw for every system, so differences are paired); calibration
sets: bootstrap over clips.

    python eval/scripts/dialect_features.py          # -> eval/results/dialect_features.json and .md
"""
import json
import os
import sys
from math import gcd

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_common import TUNE_EGY      # noqa: E402

BA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL = os.path.join(BA, "eval")
WORK = "/home/jovyan/ba-work"
CACHE = f"{WORK}/paper/dialect_features/phones.jsonl"
MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
REVISION = "2c733782da5604684829819a5eb744c193fe9398"
SYSTEMS = ["cb-v3", "cb-v2", "voxcpm2", "voxcpm2-prompt", "namaa", "lahgtna", "omnivoice", "voicetut", "cb-v3-pypi"]
CALIB = ["lahgtna_eg", "fleurs_ar_eg", "niletts"]
G = {"ɡ", "ɡː", "ɡʲ", "ɡʰ"}
ZH = {"dʒ", "dʒː", "dʒʲ", "ʒ", "ʒʲ", "dZ"}
Q = {"q", "qː"}
N_BOOT = 10000


def jl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def items():
    texts = {t["id"]: t for t in jl(f"{EVAL}/texts/egy.jsonl")}
    main = [i for i, t in texts.items() if t["category"] != "digits"]
    out = []
    for s in SYSTEMS + ["ba"]:
        d = f"{WORK}/engine/full" if s == "ba" else f"{WORK}/bakeoff/{s}"
        for f in sorted(os.listdir(d)):
            if not (f.startswith("egy__") and f.endswith(".wav")):
                continue
            _, tid, voice = f[:-4].split("__")
            if tid in main and (s != "ba" or tid not in TUNE_EGY):
                out.append({"set": s, "text_id": tid, "voice": voice, "path": os.path.join(d, f), "text": texts[tid]["text"]})
    for c in jl(f"{EVAL}/calib.jsonl"):
        if c["set"] in CALIB:
            out.append({"set": c["set"], "text_id": c["calib_id"], "voice": c["calib_id"], "path": c["path"], "text": c["text"]})
    return out


def load16(path):
    sr, y = wavfile.read(path)
    y = y.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(1)
    if np.abs(y).max() > 1.5:          # integer PCM
        y /= 32768.0
    if sr != 16000:
        g = gcd(sr, 16000)
        y = resample_poly(y, 16000 // g, sr // g).astype(np.float32)
    return y


def transcribe(todo):
    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
    # the phone inventory is read from vocab.json: the model's tokenizer class needs the phonemizer package only to
    # turn text into phones, which is not done here
    vocab = json.load(open(hf_hub_download(MODEL, "vocab.json", revision=REVISION)))
    id2tok = {i: t for t, i in vocab.items()}
    pad = vocab["<pad>"]
    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL, revision=REVISION)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL, revision=REVISION).cuda().eval()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "a") as f:
        for i, it in enumerate(todo):
            y = load16(it["path"])
            x = fe(y, sampling_rate=16000, return_tensors="pt").input_values.cuda()
            with torch.no_grad():
                ids = model(x).logits.argmax(-1)[0].cpu().numpy()
            # collapse CTC repeats and blanks token by token, so multi-character phones stay whole
            toks, prev = [], None
            for t in ids:
                if t != prev and t != pad:
                    toks.append(id2tok[int(t)])
                prev = t
            f.write(json.dumps({"path": it["path"], "phones": toks}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0:
                print(i, len(todo), it["set"], " ".join(toks[:20]), flush=True)


def counts(it, phones):
    return {"jiim": it["text"].count("ج"), "qaaf": it["text"].count("ق"),
            "g": sum(p in G for p in phones), "zh": sum(p in ZH for p in phones), "q": sum(p in Q for p in phones)}


def measures(rows, w):
    """jiim-as-[g] share and [q]-per-qaaf rate of a set of clips with weights w (bootstrap multiplicities)."""
    J = np.array([r["jiim"] > 0 for r in rows])
    K = np.array([r["qaaf"] > 0 for r in rows])
    g = np.array([r["g"] for r in rows], float)
    zh = np.array([r["zh"] for r in rows], float)
    q = np.array([r["q"] for r in rows], float)
    nq = np.array([r["qaaf"] for r in rows], float)
    den = (w * J * (g + zh)).sum()
    return {"jiim_g": (w * J * g).sum() / den if den else np.nan,
            "qaaf_q": (w * K * q).sum() / (w * K * nq).sum() if (w * K * nq).sum() else np.nan}


def main():
    its = items()
    done = {r["path"]: r["phones"] for r in jl(CACHE)} if os.path.exists(CACHE) else {}
    todo = [it for it in its if it["path"] not in done]
    if todo:
        transcribe(todo)
        done = {r["path"]: r["phones"] for r in jl(CACHE)}
    rows = [dict(it, **counts(it, done[it["path"]])) for it in its]
    by = {}
    for r in rows:
        by.setdefault(r["set"], []).append(r)

    rng = np.random.default_rng(0)
    gen_sets = SYSTEMS + ["ba"]
    texts = sorted({r["text_id"] for s in gen_sets for r in by[s]})
    voices = sorted({r["voice"] for s in gen_sets for r in by[s]})
    ti, vi = {t: i for i, t in enumerate(texts)}, {v: i for i, v in enumerate(voices)}
    heldout = sorted({r["text_id"] for r in by["ba"]})
    sets = {s: by[s] for s in SYSTEMS + CALIB}
    sets["ba (held-out)"] = by["ba"]
    sets["voxcpm2 (held-out)"] = [r for r in by["voxcpm2"] if r["text_id"] in heldout]
    point = {s: measures(R, np.ones(len(R))) for s, R in sets.items()}
    reps = {s: [] for s in sets}
    diffs = {("voxcpm2", "cb-v3"): [], ("voxcpm2", "cb-v2"): [], ("ba (held-out)", "voxcpm2 (held-out)"): []}
    for _ in range(N_BOOT):
        wt = np.bincount(rng.integers(0, len(texts), len(texts)), minlength=len(texts))
        wv = np.bincount(rng.integers(0, len(voices), len(voices)), minlength=len(voices))
        b = {}
        for s, R in sets.items():
            if s in CALIB:
                w = np.bincount(rng.integers(0, len(R), len(R)), minlength=len(R)).astype(float)
            else:
                w = np.array([wt[ti[r["text_id"]]] * wv[vi[r["voice"]]] for r in R], float)
            b[s] = measures(R, w)
            reps[s].append(b[s])
        for a, c in diffs:
            diffs[(a, c)].append({k: b[a][k] - b[c][k] for k in ("jiim_g", "qaaf_q")})

    def ci(v):
        v = np.array(v, float)
        return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    out = {"model": MODEL, "revision": REVISION, "n_boot": N_BOOT, "sets": {}, "differences": {}}
    for s, R in sets.items():
        out["sets"][s] = {"clips": len(R), "jiim_letters": int(sum(r["jiim"] for r in R)),
                          "qaaf_letters": int(sum(r["qaaf"] for r in R))}
        for k in ("jiim_g", "qaaf_q"):
            out["sets"][s][k] = {"est": float(point[s][k]), "ci": ci([x[k] for x in reps[s]])}
    for (a, c), v in diffs.items():
        out["differences"][f"{a} - {c}"] = {k: {"est": float(point[a][k] - point[c][k]), "ci": ci([x[k] for x in v])}
                                            for k in ("jiim_g", "qaaf_q")}
    os.makedirs(os.path.join(EVAL, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(EVAL, "results", "dialect_features.json"), "w"), indent=1, ensure_ascii=False)
    lines = ["# Dialect features (jiim as [g], [q] per qaaf)", ""]
    for s, v in out["sets"].items():
        lines.append(f"{s:22s} n={v['clips']:3d}  jiim->[g] {v['jiim_g']['est']:.2f} [{v['jiim_g']['ci'][0]:.2f}, "
                     f"{v['jiim_g']['ci'][1]:.2f}]  [q]/qaaf {v['qaaf_q']['est']:.2f} [{v['qaaf_q']['ci'][0]:.2f}, "
                     f"{v['qaaf_q']['ci'][1]:.2f}]")
    for k, v in out["differences"].items():
        lines.append(f"{k}: jiim {v['jiim_g']['est']:+.2f} {v['jiim_g']['ci']}, qaaf {v['qaaf_q']['est']:+.2f} {v['qaaf_q']['ci']}")
    open(os.path.join(EVAL, "results", "dialect_features.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
