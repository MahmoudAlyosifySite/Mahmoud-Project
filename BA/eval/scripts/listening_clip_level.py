"""Clip-level check of the listening test (post hoc, not in the protocol): do the automatic measures of each rated clip
track what the listeners heard in that clip?

For every generated clip of the test (four systems, held-out sentences, six FLEURS voices) the measures are those of the
paper's pipeline, read from the caches: Whisper large-v3 CER (bake-off and engine score files) and P(Egyptian) and the
top label of both dialect classifiers (dialect_classifier2.py cache). The listener side is the mean Egyptian-ness and
naturalness rating of the clip over the listeners kept by the pre-specified quality checks (listening_analysis.py).

Reported (aggregates only; no per-clip or per-listener data leave the private export):
  - Spearman correlation across clips between each measure and the mean rating, with a two-way bootstrap interval
    (listeners and sentences resampled, the per-clip means recomputed with the bootstrap weights);
  - the same correlation within systems (each measure and each rating centred on its system mean), which asks whether
    a measure follows the listeners beyond the differences between systems;
  - the mean rating of clips that each classifier labels Egyptian and of the other clips.

    python eval/scripts/listening_clip_level.py      # -> eval/results/listening_clips.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from listening_analysis import RES, load, quality_control      # noqa: E402

WORK = "/home/jovyan/ba-work"
C2_CACHE = f"{WORK}/paper/dialect2/scores.jsonl"
N_BOOT = 10000


def jl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def cer_of(src):
    """CER of one source wav from the score file of its run (bake-off system or engine run)."""
    d, name = os.path.split(src)
    key = name[:-4]
    if "/bakeoff/" in src:
        path = f"{WORK}/bakeoff/scores/{os.path.basename(d)}.jsonl"
        recs = [r for r in jl(path) if r["key"] == key]
    else:
        recs = [r for r in jl(os.path.join(d, "scores.jsonl")) if r["key"] == key]
    return recs[0]["cer"] if recs else None


def spearman_w(x, y, w):
    """Weighted Spearman: Pearson correlation of weighted ranks (weights = bootstrap multiplicities)."""
    m = w > 0
    x, y, w = x[m], y[m], w[m]
    rx, ry = rankdata(x), rankdata(y)
    mx, my = np.average(rx, weights=w), np.average(ry, weights=w)
    cov = np.average((rx - mx) * (ry - my), weights=w)
    return cov / np.sqrt(np.average((rx - mx) ** 2, weights=w) * np.average((ry - my) ** 2, weights=w))


def main():
    d, x, clips, lists, sessions, rows, dropped = load()
    qc = quality_control(sessions, rows, lists)
    keep = {s for s, q in qc.items() if q["keep"]}
    gen = [r for r in rows if r["listener"] in keep and r["kind"] == "generated" and None not in (r["egy"], r["nat"])]
    c2 = {r["path"]: r for r in jl(C2_CACHE)}

    keys = sorted({r["clip"] for r in gen})
    meas = {}
    for k in keys:
        src = clips[k]["src"]
        rec = c2.get(src)
        meas[k] = {"system": clips[k]["system"], "text": clips[k]["text_id"], "cer": cer_of(src),
                   "p1": rec["c1"]["p_egy"] if rec else None, "p2": rec["c2"]["p_egy"] if rec else None,
                   "egy1": float(rec["c1"]["label"] == "Egyptian") if rec else None,
                   "egy2": float(rec["c2"]["label"] == "EGY") if rec else None}
    missing = [k for k, m in meas.items() if None in (m["cer"], m["p1"], m["p2"])]
    if missing:
        sys.exit(f"{len(missing)} clips without measures, e.g. {clips[missing[0]]['src']}")

    listeners = sorted(keep)
    texts = sorted({r["text"] for r in gen})
    li = {l: i for i, l in enumerate(listeners)}
    ti = {t: i for i, t in enumerate(texts)}
    ki = {k: i for i, k in enumerate(keys)}
    R_clip = np.array([ki[r["clip"]] for r in gen])
    R_l = np.array([li[r["listener"]] for r in gen])
    R_t = np.array([ti[r["text"]] for r in gen])
    sys_names = sorted({m["system"] for m in meas.values()})
    clip_sys = np.array([sys_names.index(meas[k]["system"]) for k in keys])
    M = {q: np.array([meas[k][q] for k in keys], dtype=float) for q in ("cer", "p1", "p2", "egy1", "egy2")}
    Y = {s: np.array([float(r[s]) for r in gen]) for s in ("egy", "nat")}

    def clip_means(w_rows, y):
        num = np.bincount(R_clip, weights=w_rows * y, minlength=len(keys))
        den = np.bincount(R_clip, weights=w_rows, minlength=len(keys))
        with np.errstate(invalid="ignore", divide="ignore"):
            return num / den, den

    def within(v, wc):
        out = v.copy()
        for j in range(len(sys_names)):
            m = (clip_sys == j) & (wc > 0)
            if m.any():
                out[clip_sys == j] = v[clip_sys == j] - np.average(v[m], weights=wc[m])
        return out

    def stats(w_rows):
        res = {}
        for s in ("egy", "nat"):
            ym, wc = clip_means(w_rows, Y[s])
            ok = wc > 0
            for q in ("cer", "p1", "p2"):
                res[f"rho:{s}:{q}"] = spearman_w(M[q][ok], ym[ok], wc[ok])
                res[f"rhow:{s}:{q}"] = spearman_w(within(M[q], wc)[ok], within(np.nan_to_num(ym), wc)[ok], wc[ok])
            for q in ("egy1", "egy2"):
                lab = M[q] == 1
                res[f"mean:{s}:{q}:yes"] = np.average(ym[ok & lab], weights=wc[ok & lab])
                res[f"mean:{s}:{q}:no"] = np.average(ym[ok & ~lab], weights=wc[ok & ~lab])
                res[f"diff:{s}:{q}"] = res[f"mean:{s}:{q}:yes"] - res[f"mean:{s}:{q}:no"]
        return res

    point = stats(np.ones(len(gen)))
    rng = np.random.default_rng(0)
    reps = {k: [] for k in point}
    for _ in range(N_BOOT):
        wl = np.bincount(rng.integers(0, len(listeners), len(listeners)), minlength=len(listeners))
        wt = np.bincount(rng.integers(0, len(texts), len(texts)), minlength=len(texts))
        b = stats((wl[R_l] * wt[R_t]).astype(float))
        for k, v in b.items():
            reps[k].append(v)
    out = {"note": "post hoc; not in the pre-specified protocol", "listeners": len(listeners), "clips": len(keys),
           "ratings": len(gen), "ratings_per_clip": len(gen) / len(keys),
           "clips_per_system": {s: int((clip_sys == j).sum()) for j, s in enumerate(sys_names)},
           "share_labelled_egyptian": {"c1": float(M["egy1"].mean()), "c2": float(M["egy2"].mean())}, "stats": {}}
    for k, v in point.items():
        r = np.array(reps[k], dtype=float)
        lo, hi = np.nanpercentile(r, [2.5, 97.5])
        out["stats"][k] = {"est": float(v), "lo": float(lo), "hi": float(hi)}
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "listening_clips.json"), "w") as f:
        json.dump(out, f, indent=1)
    for k, v in out["stats"].items():
        print(f"{k:24s} {v['est']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]")
    print({k: out[k] for k in ("listeners", "clips", "ratings", "ratings_per_clip", "clips_per_system")})


if __name__ == "__main__":
    main()
