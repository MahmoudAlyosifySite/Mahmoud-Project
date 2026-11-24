"""Shared helpers for the paper experiments (step A of BA/paper/PLAN.md). Stdlib + numpy + scipy.

Statistics
  two_way_bootstrap  95% percentile CI of a mean over utterances, resampling texts AND voices
                     independently with replacement (two-way random-effects bootstrap): an utterance
                     with text t and voice v gets weight (#times t drawn) x (#times v drawn). Seeds of
                     the same (text, voice) cell stay together, so seed variation is inside the cell.
                     B = 10,000 resamples; generator seeded so every run gives the same intervals.
  paired             difference between two systems on the same (text, voice) cells (each system's cell
                     value = mean over its seeds): two-way bootstrap CI of the mean difference, two-sided
                     bootstrap p (2 x the smaller tail share, floored at 1/B), Wilcoxon signed-rank on the
                     cell differences (ignores clustering; reported next to the bootstrap), Cohen's d_z,
                     matched-pairs rank-biserial r and, for 0/1 metrics, Cohen's h on the pooled shares.
  cohen_kappa        two raters, binary labels.

Row loaders return dicts with: text_id, voice_id, cond (egy_main / egy_digits / diac / cs / en / ...),
seed_index, cer_best (CER, best of spelled/digit reference), wer_best, err (CER for Arabic text, WER
otherwise), sim_ecapa, sim_wavlm, dialect (classifier 1 top label), mos_p808, fail, wav.
"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from engine_common import ENGINE_ROOT, RESULTS, EVAL, TUNE_IDS, jl  # noqa: E402

PAPER_WORK = "/home/jovyan/ba-work/paper"
PAPER_RESULTS = os.path.join(os.path.dirname(EVAL), "paper", "results")
BAKEOFF_ROOT = "/home/jovyan/ba-work/bakeoff"
B_DEFAULT = 10000
RNG_SEED = 20261115


def cond_of(condition, category, lang):
    if condition == "egy":
        return "egy_digits" if category == "digits" else "egy_main"
    if condition == "long":
        return f"long_{lang}"
    return condition


# ------------------------------------------------------------------ loaders
def bakeoff_rows(system):
    """Phase 1 bake-off utterances of one system (seed index 0), via engine_report.bakeoff_rows."""
    from engine_report import bakeoff_rows as _br
    out = []
    for r in _br(system):
        d = dict(r)
        d["cond"] = cond_of(r["condition"], r["category"], r["lang"])
        d["seed_index"] = 0
        d["system"] = system
        d["wav"] = f"{BAKEOFF_ROOT}/{system}/{r['utt_id']}.wav"
        out.append(d)
    return out


def engine_rows(exp, system=None):
    """Rows of an engine-style experiment (meta/manifest/scores in ENGINE_ROOT/<exp>), via analyze_engine.load."""
    from analyze_engine import load
    out = []
    for r in load(exp):
        d = dict(r)
        d["cond"] = cond_of(r.get("condition", r.get("cond")), r.get("category"), r["lang"])
        d["seed_index"] = r.get("seed_index", 0)
        d["system"] = system or r.get("system")
        d["wav"] = f"{ENGINE_ROOT}/{exp}/{r['utt_id']}.wav"
        out.append(d)
    return out


def held_out(rows):
    return [r for r in rows if r["text_id"] not in TUNE_IDS]


_DIALECT2 = {}


def attach_dialect(rows):
    """Adds c1 (classifier 1 Egyptian, 0/1 from the stored top label), c2 (classifier 2 top label == EGY) and
    c2_p (classifier 2 P(EGY)) from dialect_classifier2.py's cache, plus failed (0/1)."""
    if not _DIALECT2:
        _DIALECT2.update({r["path"]: r for r in jl(f"{PAPER_WORK}/dialect2/scores.jsonl")})
    for r in rows:
        r["c1"] = egyptian(r.get("dialect")) if r.get("lang") == "arz" else None
        rec = _DIALECT2.get(r["wav"])
        r["c2"] = None if rec is None or "c2" not in rec else float(rec["c2"]["label"] == "EGY")
        r["c2_p"] = None if rec is None or "c2" not in rec else rec["c2"]["p_egy"]
        r["failed"] = float(bool(r.get("fail")))
    return rows


def add_canonical_cer(rows):
    """Code-switch rows: CER with Latin loanwords mapped to Arabic script on both sides (engine_report.py's
    cer_canonical, via analyze_engine.canonical)."""
    import jiwer
    from analyze_engine import canonical
    from engine_common import _text_map
    T = _text_map()
    for r in rows:
        if r["cond"] == "cs":
            hyp = canonical(r.get("asr") or "")
            r["cer_canonical"] = float(jiwer.cer(canonical(T[r["text_id"]]["text"]), hyp)) if hyp else 1.0
    return rows


def holm(pvals):
    """Holm-Bonferroni adjusted p-values for a dict {name: p}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = round(run, 5)
    return out


def egyptian(label):
    """Classifier 1 label -> 0/1 (None if no label)."""
    return None if label is None else float(label == "Egyptian")


# ------------------------------------------------------------------ bootstrap
def _index(values):
    keys = sorted(set(values))
    pos = {k: i for i, k in enumerate(keys)}
    return np.array([pos[v] for v in values]), len(keys)


def _draws(rng, n, B):
    return rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float64)


def two_way_bootstrap(values, texts, voices, B=B_DEFAULT, seed=RNG_SEED):
    """values, texts, voices: equal-length lists (None values dropped). Returns dict(mean, lo, hi, n, ...)."""
    keep = [i for i, v in enumerate(values) if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not keep:
        return dict(mean=None, lo=None, hi=None, n=0)
    x = np.array([values[i] for i in keep], dtype=np.float64)
    ti, nt = _index([texts[i] for i in keep])
    vi, nv = _index([voices[i] for i in keep])
    rng = np.random.default_rng(seed)
    Ct, Cv = _draws(rng, nt, B), _draws(rng, nv, B)
    W = Ct[:, ti] * Cv[:, vi]
    tot = W.sum(1)
    ok = tot > 0
    m = (W[ok] @ x) / tot[ok]
    lo, hi = np.percentile(m, [2.5, 97.5])
    return dict(mean=round(float(x.mean()), 4), lo=round(float(lo), 4), hi=round(float(hi), 4), n=len(x),
                n_texts=nt, n_voices=nv, B=int(ok.sum()))


def cell_means(rows, key):
    """{(text_id, voice_id): mean of key over the rows (seeds) of that cell}."""
    acc = {}
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        acc.setdefault((r["text_id"], r["voice_id"]), []).append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def paired(rows_a, rows_b, key, B=B_DEFAULT, seed=RNG_SEED, binary=False):
    """A minus B on the shared (text, voice) cells; see module docstring."""
    from scipy import stats
    ca, cb = cell_means(rows_a, key), cell_means(rows_b, key)
    cells = sorted(set(ca) & set(cb))
    if not cells:
        return None
    d = np.array([ca[c] - cb[c] for c in cells])
    texts, voices = [c[0] for c in cells], [c[1] for c in cells]
    bs_rng = np.random.default_rng(seed)
    ti, nt = _index(texts)
    vi, nv = _index(voices)
    Ct, Cv = _draws(bs_rng, nt, B), _draws(bs_rng, nv, B)
    W = Ct[:, ti] * Cv[:, vi]
    tot = W.sum(1)
    ok = tot > 0
    m = (W[ok] @ d) / tot[ok]
    lo, hi = np.percentile(m, [2.5, 97.5])
    p_boot = min(1.0, max(2 * min(float(np.mean(m <= 0)), float(np.mean(m >= 0))), 1.0 / ok.sum()))
    nz = d[d != 0]
    if len(nz) >= 1:
        w = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        p_w = float(w.pvalue)
        rk = stats.rankdata(np.abs(nz))
        rb = float((rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum())
    else:
        p_w, rb = 1.0, 0.0
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else float("nan")
    out = dict(metric=key, n_cells=len(cells), n_texts=nt, n_voices=nv,
               mean_a=round(float(np.mean([ca[c] for c in cells])), 4), mean_b=round(float(np.mean([cb[c] for c in cells])), 4),
               diff=round(float(d.mean()), 4), ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4),
               p_bootstrap=round(p_boot, 5), p_wilcoxon=round(p_w, 6), d_z=None if not sd or math.isnan(sd) else round(float(d.mean() / sd), 3),
               rank_biserial=round(rb, 3), a_better_cells=None, B=int(ok.sum()))
    if binary:
        pa, pb = out["mean_a"], out["mean_b"]
        out["cohens_h"] = round(2 * math.asin(math.sqrt(min(max(pa, 0), 1))) - 2 * math.asin(math.sqrt(min(max(pb, 0), 1))), 3)
    out["a_higher_cells"] = int((d > 0).sum())
    out["b_higher_cells"] = int((d < 0).sum())
    out["tied_cells"] = int((d == 0).sum())
    del out["a_better_cells"]
    return out


def cohen_kappa(a, b):
    """a, b: equal-length lists of 0/1. Returns dict(kappa, observed, expected, table)."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    n = len(a)
    if n == 0:
        return dict(kappa=None, n=0)
    po = float(np.mean(a == b))
    pa, pb = a.mean(), b.mean()
    pe = float(pa * pb + (1 - pa) * (1 - pb))
    k = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    tab = {"both_egyptian": int(((a == 1) & (b == 1)).sum()), "only_first": int(((a == 1) & (b == 0)).sum()),
           "only_second": int(((a == 0) & (b == 1)).sum()), "neither": int(((a == 0) & (b == 0)).sum())}
    return dict(kappa=round(k, 4), observed_agreement=round(po, 4), expected_agreement=round(pe, 4), n=n, table=tab)


def fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def ci(d, nd=3):
    """'0.123 [0.100, 0.150]' from a two_way_bootstrap dict."""
    if not d or d.get("mean") is None:
        return "-"
    return f"{d['mean']:.{nd}f} [{d['lo']:.{nd}f}, {d['hi']:.{nd}f}]"


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)
