r"""Build every number, table and figure of the paper from results/.

    python scripts/make_assets.py            # writes paper/generated/*.tex and paper/figures/*.pdf

Nothing in the paper is typed by hand: paper/generated/numbers.tex defines \R{<config>:<protocol>} for
every configuration, and the tables are written as tabular bodies. A cell whose runs are not finished
is printed as "--".

Protocols (columns)
    lp50   linear probe trained with all training labels (the oracle-label probe)
    knn    weighted kNN with all training labels as memory bank
    lp5    linear probe trained with the labeled subset only
    ft     fine-tuning on the labeled subset, default recipe (dlt.finetune)
    ftt    fine-tuning on the labeled subset, recipe selected on a split of the labeled subset
    ftx    the recipe that the labeled subset selects for SimCLR, applied to another encoder
    fts    "selected recipe": ftt for label-free encoders, ftx for encoders that were trained with the labels
           (their own labeled images cannot serve as a validation set)
    fto    recipe chosen on the TEST set of seed 0 (an upper bound), evaluated on the other seeds
    fm     FixMatch initialised from the encoder
    dist   SimCLR v2-style fine-tune + self-distillation
    clf    accuracy of the method's own classifier (cross-entropy, FixMatch, MixMatch)
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
from scipy import stats as sps      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import stats as st                  # noqa: E402

RUNS = os.path.join(ROOT, "results", "runs")
GEN = os.path.join(ROOT, "paper", "generated")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(GEN, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "lines.linewidth": 1.4,
                     "lines.markersize": 4, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
                     "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.5})
# Okabe-Ito colours; every series also has its own marker and line style
STYLE = {"simclr": ("#0072B2", "o", "-", "SimCLR (all images, no labels)"),
         "sub": ("#56B4E9", "s", "--", "SimCLR (labeled images only)"),
         "supcon": ("#D55E00", "^", "-", "SupCon (labeled images only)"),
         "union": ("#CC79A7", "D", "-.", "SupCon+U (all images)")}
ARMS = {"simclr": "simclr_all_e{E}", "sub": "simclr_sub_f0.1_e{E}", "supcon": "supcon_lab_f0.1_e{E}",
        "union": "union_f0.1_e{E}"}
ARM_TEX = {"simclr": r"\simclr", "sub": r"\simclrsub", "supcon": r"\supcon", "union": r"\union"}

# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
RUN = {}
for path in glob.glob(os.path.join(RUNS, "*", "result.json")):
    d = os.path.dirname(path)
    name = os.path.basename(d)
    r = json.load(open(path))
    pj = os.path.join(d, "probes.json")
    r["probes"] = json.load(open(pj)) if os.path.exists(pj) else {}
    r["dir"] = d
    RUN[name] = r
CONFIGS = sorted({re.sub(r"_s\d+$", "", n) for n in RUN if re.search(r"_s\d+$", n)})


def seeds(cfg):
    return sorted(int(m.group(1)) for n in RUN for m in [re.fullmatch(re.escape(cfg) + r"_s(\d+)", n)] if m)


def companion(name, kind):
    """Name of the run that post-processes encoder ``name`` (fine-tuning, FixMatch, ...)."""
    if kind == "fm" and name.startswith("stl_"):
        return "stl_fixmatch_from_" + name[4:]
    return {"ft": "ft_", "ftt": "ftsearch_", "ftd": "ftsearch_", "fm": "fixmatch_from_", "dist": "distill_",
            "fto": "ftoracle_", "fto0": "ftoracle_", "ftx": "fttransfer_"}[kind] + name


def label_free(name):
    return name.startswith(("simclr_all", "byol_all", "simsiam_all", "stl_simclr_all"))


def metric(name, proto):
    r = RUN.get(name)
    if r is None:
        return None
    m, p = r["metrics"], r["probes"]
    stl = name.startswith("stl_")
    if proto == "lp50":
        return None if stl else m.get("linear_acc")
    if proto == "knn":
        return m.get("knn_acc")
    if proto == "lp5":
        return m.get("linear_acc") if stl else m.get("linear_lab_acc", p.get("linear_0.1"))
    if proto.startswith("lpf:"):
        return p.get("linear_" + proto[4:])
    if proto == "fts":
        proto = "ftt" if label_free(name) else "ftx"
    if proto in ("ft", "ftt", "ftd", "fm", "dist", "fto", "fto0", "ftx"):
        c = RUN.get(companion(name, proto))
        key = {"ft": "finetune_acc", "ftt": "finetune_tuned_acc", "ftd": "finetune_default_acc",
               "fm": "classifier_acc", "dist": "classifier_acc", "fto": "finetune_tuned_acc",
               "fto0": "finetune_oracle_acc", "ftx": "finetune_tuned_acc"}[proto]
        return None if c is None else c["metrics"].get(key)
    if proto == "clf":
        return m.get("classifier_acc")
    if proto == "cdnv_lab":
        return m.get("seen_cdnv") if stl else p.get("cdnv_labeled_0.1")
    if proto == "cdnv_rest":
        return p.get("cdnv_rest_0.1")
    if proto == "cdnv_test":
        return m.get("test_cdnv") if stl else p.get("cdnv_test_0.1")
    if proto == "lp5_train":
        return p.get("linear_0.1_train")
    if proto == "etf_lab":
        return p.get("etf_dev_labeled_0.1")
    if proto == "time":
        return r.get("train_time_s", 0) / 60
    return m.get(proto, p.get(proto))


def V(cfg, proto, only=None):
    out = [metric(f"{cfg}_s{s}", proto) for s in seeds(cfg) if only is None or s in only]
    return np.array([v for v in out if v is not None], dtype=float)


def mean(cfg, proto):
    v = V(cfg, proto)
    return float(v.mean()) if len(v) else float("nan")


def MS(cfg, proto, d=1, bold=False):
    v = V(cfg, proto)
    if len(v) == 0:
        return "--"
    s = f"{v.mean():.{d}f}"
    if bold:
        s = r"\textbf{" + s + "}"
    if len(v) > 1:
        s += rf"{{\scriptsize$\,\pm\,${v.std(ddof=1):.{d}f}}}"
    return s


def nseeds(cfg, proto="knn"):
    return len(V(cfg, proto))


def fmt(x, d=1, sign=False):
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def write(name, text):
    with open(os.path.join(GEN, name), "w") as f:
        f.write("% generated by scripts/make_assets.py -- do not edit\n" + text.rstrip() + "\n")


M = {}   # macros: key -> text, used in the paper as \R{key}


def best_of(cfgs, proto):
    vals = {c: mean(c, proto) for c in cfgs}
    vals = {c: v for c, v in vals.items() if np.isfinite(v)}
    return max(vals, key=vals.get) if vals else None


# ----------------------------------------------------------------------------
# macros for every configuration and protocol
# ----------------------------------------------------------------------------
PROTOS = ["lp50", "knn", "lp5", "ft", "ftt", "ftd", "fts", "ftx", "fto", "fto0", "fm", "dist", "clf", "cdnv_lab", "cdnv_rest", "cdnv_test",
          "lp5_train", "seen_cdnv", "heldout_cdnv", "test_cdnv", "linear_acc_gray", "cka_hist", "cka_label"]
for cfg in CONFIGS:
    for proto in PROTOS:
        v = V(cfg, proto)
        if len(v) == 0:
            continue
        d = 2 if "cdnv" in proto or "cka" in proto else 1
        M[f"{cfg}:{proto}"] = f"{v.mean():.{d}f}"
        M[f"{cfg}:{proto}:pm"] = f"{v.mean():.{d}f}" + (rf"\,$\pm$\,{v.std(ddof=1):.{d}f}" if len(v) > 1 else "")
        M[f"{cfg}:{proto}:n"] = str(len(v))
    t = V(cfg, "time")
    if len(t):
        M[f"{cfg}:time"] = f"{t.min():.0f}"


def diffmacro(tag, a, pa, b, pb, d=1):
    x, y = mean(a, pa), mean(b, pb)
    if np.isfinite(x) and np.isfinite(y):
        M[f"d:{tag}"] = f"{abs(x - y):.{d}f}"
        M[f"ds:{tag}"] = f"{x - y:+.{d}f}"


# ----------------------------------------------------------------------------
# statistics: hierarchical bootstrap, Holm correction, TOST
# ----------------------------------------------------------------------------
PRED = {"lp50": [("{c}", "linear")], "knn": [("{c}", "knn")], "lp5": [("{c}", "linear_lab"), ("{c}", "linear_0.1")],
        "ft": [("ft_{c}", "finetune")], "ftt": [("ftsearch_{c}", "finetune_tuned")],
        "ftx": [("fttransfer_{c}", "finetune_tuned")],
        "fm": [("fixmatch_from_{c}", "classifier")], "clf": [("{c}", "classifier")]}
_Y = {}


def correct(cfg, proto, dataset="cifar10"):
    if dataset not in _Y:
        _Y[dataset] = st.test_labels(dataset)
    if proto == "fts":
        proto = "ftt" if label_free(cfg) else "ftx"
    if proto == "fto":      # seed 0 chose the recipe on its test set and is left out
        c = st.load_correct(f"ftoracle_{cfg}", "finetune_tuned", _Y[dataset], RUNS)
        return {k: v for k, v in c.items() if k != 0}
    for tmpl, key in PRED[proto]:
        c = st.load_correct(tmpl.format(c=cfg), key, _Y[dataset], RUNS)
        if c:
            return c
    return {}


N_BOOT = 100000
CACHE_PATH = os.path.join(ROOT, "results", "stats_cache.json")
CACHE = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}


def cmp(a, pa, b, pb=None, dataset="cifar10"):
    ca, cb = correct(a, pa, dataset), correct(b, pb or pa, dataset)
    if len(ca) < 2 or len(cb) < 2:
        return None
    key = f"{a}|{pa}|{sorted(ca)}|{b}|{pb or pa}|{sorted(cb)}|{N_BOOT}"
    if key not in CACHE:
        CACHE[key] = {k: (float(v) if not isinstance(v, (bool, int)) else v)
                      for k, v in st.hierarchical_bootstrap(ca, cb, n_boot=N_BOOT).items()}
    return dict(CACHE[key])


BUDGETS = [(100, "9,700"), (200, "19,400"), (400, "38,800"), (1000, "97,000")]
FAMILY = []   # (tag, group label, text of the comparison, a, pa, b, pb)
for E, steps in BUDGETS:
    s, c, u, sub = (ARMS[k].format(E=E) for k in ("simclr", "supcon", "union", "sub"))
    for proto, pname in (("lp50", "LP-50k"), ("lp5", "LP-5k"), ("ft", "FT-5k")):
        FAMILY.append((f"simclr-supcon:{proto}:e{E}", f"{pname}, {steps} steps", r"\simclr $-$ \supcon", s, proto, c, proto))
        FAMILY.append((f"simclr-union:{proto}:e{E}", f"{pname}, {steps} steps", r"\simclr $-$ \union", s, proto, u, proto))
    FAMILY.append((f"union-supcon:lp50:e{E}", f"LP-50k, {steps} steps", r"\union $-$ \supcon", u, "lp50", c, "lp50"))
    FAMILY.append((f"simclr-sub:lp50:e{E}", f"LP-50k, {steps} steps", r"\simclr $-$ \simclrsub", s, "lp50", sub, "lp50"))
    FAMILY.append((f"supcon-sub:lp50:e{E}", f"LP-50k, {steps} steps", r"\supcon $-$ \simclrsub", c, "lp50", sub, "lp50"))
for E, steps in ((10, "970"), (25, "2,425"), (50, "4,850")):
    FAMILY.append((f"simclr-supcon:lp50:e{E}", f"LP-50k, {steps} steps", r"\simclr $-$ \supcon",
                   ARMS["simclr"].format(E=E), "lp50", ARMS["supcon"].format(E=E), "lp50"))
for _p, _n in (("fts", "FT-5k (selected recipe)"), ("fto", "FT-5k (oracle recipe)")):
    FAMILY.append((f"simclr-supcon:{_p}:e200", f"{_n}, 19,400 steps", r"\simclr $-$ \supcon", "simclr_all_e200", _p,
                   "supcon_lab_f0.1_e200", _p))
    FAMILY.append((f"simclr-union:{_p}:e200", f"{_n}, 19,400 steps", r"\simclr $-$ \union", "simclr_all_e200", _p,
                   "union_f0.1_e200", _p))
FAMILY.append(("simclr-supcon:fm:e200", "FixMatch from encoder, 19,400 steps", r"\simclr $-$ \supcon", "simclr_all_e200", "fm",
               "supcon_lab_f0.1_e200", "fm"))
FAMILY.append(("simclr-union:fm:e200", "FixMatch from encoder, 19,400 steps", r"\simclr $-$ \union", "simclr_all_e200", "fm",
               "union_f0.1_e200", "fm"))
FAMILY.append(("twostage-simclr:lp5:e200", "LP-5k, 19,400 steps", r"\simclr{}$\to$\supcon $-$ \simclr",
               "twostage_simclr200_supcon5", "lp5", "simclr_all_e200", "lp5"))
FAMILY.append(("simclrft-ce:e200", "label-budget pipelines, 19,400 steps", r"\simclr{}+FT $-$ CE on 5k", "simclr_all_e200", "ft",
               "ce_f0.1_e200", "clf"))
FAMILY.append(("supconft-ce:e200", "label-budget pipelines, 19,400 steps", r"\supcon{}+FT $-$ CE on 5k", "supcon_lab_f0.1_e200", "ft",
               "ce_f0.1_e200", "clf"))
STATS = []
for tag, group, text, a, pa, b, pb in FAMILY:
    r = cmp(a, pa, b, pb)
    if r is not None:
        STATS.append((tag, group, text, r))
json.dump(CACHE, open(CACHE_PATH, "w"), indent=0)
_first = {}
for i, (_t, g, *_r) in enumerate(STATS):
    _first.setdefault(g, i)
STATS.sort(key=lambda x: _first[x[1]])
if STATS:
    adj = st.holm([r["p"] for *_x, r in STATS])
    for (tag, group, text, r), pa_ in zip(STATS, adj):
        r["p_holm"] = float(pa_)

        def pf(p):
            return "$<$0.001" if p < 0.001 else f"{p:.3f}"
        M[f"cmp:{tag}:diff"] = f"{abs(r['diff']):.1f}"
        M[f"cmp:{tag}:sdiff"] = f"{r['diff']:+.1f}"
        M[f"cmp:{tag}:ci"] = f"[{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]"
        M[f"cmp:{tag}:p"] = pf(r["p_holm"])
        M[f"cmp:{tag}:tost"] = pf(r["p_tost"])
    rows, last = [], None
    for tag, group, text, r in STATS:
        if group != last:
            rows.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{group}}}}} \\")
            last = group
        eq = "yes" if r["p_tost"] < 0.05 else "no"
        rows.append(f"\\quad {text} & {r['n_a']}/{r['n_b']} & {r['diff']:+.1f} & [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}] & "
                    f"{M[f'cmp:{tag}:p']} & {eq} \\\\")
    write("tab_stats.tex", "\\begin{tabular}{lccccc}\n\\toprule\ncomparison & seeds & $\\Delta$ (points) & 95\\% CI & "
          "$p$ (Holm) & equivalent within $\\pm1$? \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    M["stats:ntests"] = str(len(STATS))

# ----------------------------------------------------------------------------
# Table: main results on CIFAR-10 (four arms x budgets x protocols)
# ----------------------------------------------------------------------------
rows = []
for E, steps in BUDGETS:
    cfgs = {k: ARMS[k].format(E=E) for k in ("sub", "simclr", "supcon", "union")}
    if not any(nseeds(c) for c in cfgs.values()):
        continue
    rows.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{steps} steps}}}} \\")
    best = {p: best_of(cfgs.values(), p) for p in ("lp50", "knn", "lp5", "ft")}
    for k, c in cfgs.items():
        if not nseeds(c):
            continue
        rows.append(f"\\quad {ARM_TEX[k]} & {nseeds(c)} & " + " & ".join(MS(c, p, bold=(best[p] == c)) for p in ("lp50", "knn", "lp5", "ft")) + r" \\")
write("tab_main.tex", "\\begin{tabular}{lccccc}\n\\toprule\n & & \\multicolumn{2}{c}{oracle labels (50k)} & "
      "\\multicolumn{2}{c}{label budget (5k)} \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
      "method & seeds & LP-50k & $k$NN-50k & LP-5k & FT-5k \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# all budgets (appendix)
rows = []
for E in (10, 25, 50, 100, 200, 400, 1000):
    line = [f"{97 * E:,}"]
    for k in ("simclr", "sub", "supcon", "union"):
        c = ARMS[k].format(E=E)
        line += [MS(c, "lp50"), MS(c, "lp5")]
    rows.append(" & ".join(line) + r" \\")
write("tab_budgets.tex", "\\begin{tabular}{rcccccccc}\n\\toprule\n & \\multicolumn{2}{c}{\\simclr} & \\multicolumn{2}{c}{\\simclrsub} & "
      "\\multicolumn{2}{c}{\\supcon} & \\multicolumn{2}{c}{\\union} \\\\\n"
      "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
      "steps & LP-50k & LP-5k & LP-50k & LP-5k & LP-50k & LP-5k & LP-50k & LP-5k \\\\\n\\midrule\n" + "\n".join(rows)
      + "\n\\bottomrule\n\\end{tabular}")

# ----------------------------------------------------------------------------
# Figure 1: accuracy against the number of steps, three protocols
# ----------------------------------------------------------------------------
ES = (10, 25, 50, 100, 200, 400, 1000)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)
for ax, (proto, title) in zip(axes, (("lp50", "(a) LP-50k (oracle labels)"), ("lp5", "(b) LP-5k"), ("ft", "(c) FT-5k"))):
    for k, tmpl in ARMS.items():
        xs, ys, es = [], [], []
        for E in ES:
            v = V(tmpl.format(E=E), proto)
            if len(v):
                xs.append(97 * E), ys.append(v.mean()), es.append(v.std(ddof=1) if len(v) > 1 else 0)
        if xs:
            col, mk, ls, lab = STYLE[k]
            ax.errorbar(xs, ys, yerr=es, color=col, marker=mk, ls=ls, label=lab, capsize=2, elinewidth=0.8)
    ax.set_xscale("log")
    ax.set_xticks([970, 2425, 4850, 9700, 19400, 38800, 97000])
    ax.set_xticklabels(["970", "2.4k", "4.9k", "9.7k", "19.4k", "38.8k", "97k"], rotation=40)
    ax.minorticks_off()
    ax.set_xlabel("gradient steps")
    ax.set_title(title, loc="left")
axes[0].set_ylabel("CIFAR-10 test accuracy (%)")
axes[0].set_ylim(60, 93)
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.22), columnspacing=1.2, handlelength=2.6)
fig.savefig(os.path.join(FIG, "fig_budget.pdf"))
plt.close(fig)

# ----------------------------------------------------------------------------
# Figure 2: which method wins, as a function of steps and label fraction
# ----------------------------------------------------------------------------
FRS = (0.01, 0.05, 0.1, 0.2, 0.5, 1.0)
GES = (25, 100, 400)


def frac_cfg(kind, f, E):
    if kind == "simclr":
        return f"simclr_all_e{E}"
    return f"{'supcon_lab' if kind == 'supcon' else 'union'}_f{f}_e{E}"


def frac_val(kind, f, E, proto):
    """proto: 'oracle' (LP with all labels) or 'budget' (LP with the f labels)."""
    cfg = frac_cfg(kind, f, E)
    if proto == "oracle" or f >= 1.0:
        return mean(cfg, "lp50")
    v = V(cfg, "linear_lab_acc") if kind != "simclr" else V(cfg, f"lpf:{f}")
    if len(v) == 0:
        v = V(cfg, f"lpf:{f}")
    return float(v.mean()) if len(v) else float("nan")


fig, axes = plt.subplots(2, 2, figsize=(7.2, 3.5), sharey=True, sharex=True, gridspec_kw={"hspace": 0.32, "wspace": 0.06})
panels = [("supcon", "oracle", r"(a) SupCon $-$ SimCLR, probe with all labels (LP-all)"),
          ("supcon", "budget", r"(b) SupCon $-$ SimCLR, probe with the $f$ labels (LP-$f$)"),
          ("union", "oracle", r"(c) SupCon+U $-$ SimCLR, LP-all"), ("union", "budget", r"(d) SupCon+U $-$ SimCLR, LP-$f$")]
rows_frac = []
for ax, (kind, proto, title) in zip(axes.ravel(), panels):
    G = np.full((len(GES), len(FRS)), np.nan)
    for i, E in enumerate(GES):
        for j, f in enumerate(FRS):
            if kind == "union" and f >= 1.0:
                continue        # with all labels SupCon+U is SupCon
            G[i, j] = frac_val(kind, f, E, proto) - frac_val("simclr", f, E, proto)
    im = ax.imshow(G, cmap="RdBu_r", vmin=-12, vmax=12, aspect="auto", origin="lower")
    for i in range(len(GES)):
        for j in range(len(FRS)):
            if np.isfinite(G[i, j]):
                ax.text(j, i, f"{G[i, j]:+.1f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(G[i, j]) > 7 else "black")
    ax.set_xticks(range(len(FRS)))
    ax.set_xticklabels([f"{100 * f:g}" for f in FRS])
    ax.set_yticks(range(len(GES)))
    ax.set_yticklabels([f"{97 * E:,}" for E in GES])
    ax.set_title(title, loc="left", fontsize=7.5)
    ax.grid(False)
for ax in axes[1]:
    ax.set_xlabel("labeled fraction $f$ (%)")
for ax in axes[:, 0]:
    ax.set_ylabel("gradient steps")
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cb.set_label("difference (points)")
fig.savefig(os.path.join(FIG, "fig_wins.pdf"))
plt.close(fig)
rows = []
for E in GES:
    for f in FRS:
        line = [f"{97 * E:,}", f"{100 * f:g}\\%"]
        for kind in ("simclr", "supcon", "union"):
            for proto in ("oracle", "budget"):
                x = float("nan") if (kind == "union" and f >= 1) else frac_val(kind, f, E, proto)
                line.append(fmt(x))
        rows.append(" & ".join(line) + r" \\")
    rows.append(r"\addlinespace")
write("tab_fraction.tex", "\\begin{tabular}{rrcccccc}\n\\toprule\n & & \\multicolumn{2}{c}{\\simclr} & \\multicolumn{2}{c}{\\supcon} & "
      "\\multicolumn{2}{c}{\\union} \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n"
      "steps & $f$ & LP-all & LP-$f$ & LP-all & LP-$f$ & LP-all & LP-$f$ \\\\\n\\midrule\n" + "\n".join(rows[:-1])
      + "\n\\bottomrule\n\\end{tabular}")

# ----------------------------------------------------------------------------
# Table: other datasets and architectures
# ----------------------------------------------------------------------------
BLOCKS = [("CIFAR-100, 10\\% labels (5,000), ResNet-18", "c100_", "_f0.1", [(100, "9,700"), (200, "19,400")]),
          ("STL-10, 5,000 labels + 100,000 unlabeled images, 96 px, ResNet-18", "stl_", "", [(100, "20,500"), (200, "41,000")]),
          ("CIFAR-10, 10\\% labels, ResNet-50", "r50_", "_f0.1", [(100, "9,700"), (200, "19,400")])]
rows = []
for title, pre, suf, buds in BLOCKS:
    rows.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{title}}}}} \\")
    for E, steps in buds:
        cfgs = {"sub": f"{pre}simclr_sub{suf}_e{E}", "simclr": f"{pre}simclr_all_e{E}", "supcon": f"{pre}supcon_lab{suf}_e{E}",
                "union": f"{pre}union{suf}_e{E}"}
        best = {p: best_of(cfgs.values(), p) for p in ("lp50", "lp5", "ft")}
        for k, c in cfgs.items():
            if nseeds(c):
                rows.append(f"\\quad {ARM_TEX[k]} & {steps} & {nseeds(c)} & "
                            + " & ".join(MS(c, p, bold=(best[p] == c)) for p in ("lp50", "lp5", "ft")) + r" \\")
    rows.append(r"\addlinespace")
write("tab_datasets.tex", "\\begin{tabular}{lrcccc}\n\\toprule\nmethod & steps & seeds & LP-all & LP-budget & FT-budget \\\\\n\\midrule\n"
      + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}")

# ----------------------------------------------------------------------------
# Table: the protocol decides the ranking (encoders at 19,400 steps)
# ----------------------------------------------------------------------------
ENC = [(r"\simclr", "simclr_all_e200"), ("BYOL", "byol_all_e200"), (r"\simclrsub", "simclr_sub_f0.1_e200"),
       (r"\supcon", "supcon_lab_f0.1_e200"), (r"\union", "union_f0.1_e200"),
       (r"\simclr{}$\to$\supcon", "twostage_simclr200_supcon5")]
PCOLS = ("lp50", "knn", "lp5", "ft", "fts", "fto", "fm")


def ranks(cfgs, proto):
    vals = {c: mean(c, proto) for c in cfgs}
    order = sorted([c for c in vals if np.isfinite(vals[c])], key=lambda c: -vals[c])
    return {c: i + 1 for i, c in enumerate(order)}


RK = {p: ranks([c for _, c in ENC], p) for p in PCOLS}
rows = []
for name, c in ENC:
    cells = []
    for p in PCOLS:
        s = MS(c, p)
        if c in RK[p]:
            s += rf"\,\textsuperscript{{({RK[p][c]})}}"
        cells.append(s)
    rows.append(f"{name} & " + " & ".join(cells) + r" \\")
write("tab_protocol.tex", "\\begin{tabular}{lccccccc}\n\\toprule\n & \\multicolumn{2}{c}{oracle labels} & \\multicolumn{5}{c}{label budget (5,000 labels)} \\\\\n"
      "\\cmidrule(lr){2-3}\\cmidrule(lr){4-8}\n & & & & \\multicolumn{3}{c}{FT-5k, by recipe} & \\\\\n\\cmidrule(lr){5-7}\n"
      "encoder & LP-50k & $k$NN-50k & LP-5k & default & selected & oracle & FixMatch-init \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
# rank agreement between the oracle probe and each label-budget protocol
main_enc = [c for _, c in ENC[:5]]
for p in ("lp5", "ft", "fts", "fto", "fm"):
    a = [mean(c, "lp50") for c in main_enc]
    b = [mean(c, p) for c in main_enc]
    ok = [i for i in range(len(a)) if np.isfinite(a[i]) and np.isfinite(b[i])]
    if len(ok) >= 4:
        M[f"kendall:{p}"] = f"{sps.kendalltau([a[i] for i in ok], [b[i] for i in ok]).statistic:.2f}"

# fine-tuning search: label-split selection and oracle selection side by side
REGN = {"none": "none", "ls": "label smoothing", "mixup": "mixup"}


def recipe(b):
    return f"{'LP' if b['head'] == 'lp' else 'random'} head, lr {b['lr']:g}, {b['epochs']} passes, {REGN[b['reg']]}"


rows = []
for name, c in ENC:
    a, o = RUN.get(f"ftsearch_{c}_s0"), RUN.get(f"ftoracle_{c}_s0")
    if a and a.get("grid"):
        g = a["grid"]
        vals = [x["val_acc"] for x in g]
        M[f"ftgrid:{c}:valmin"], M[f"ftgrid:{c}:valmax"] = f"{min(vals):.1f}", f"{max(vals):.1f}"
        M[f"ftgrid:{c}:valrange"] = f"{max(vals) - min(vals):.1f}"
        rows.append(f"{name} & label split & {recipe(a['best'])} & {min(vals):.1f}--{max(vals):.1f} & {MS(c, 'ftt')} \\\\")
    if o and o.get("grid"):
        g = o["grid"]
        tv = [x["test_acc"] for x in g]
        M[f"ftgrid:{c}:testmin"], M[f"ftgrid:{c}:testmax"] = f"{min(tv):.1f}", f"{max(tv):.1f}"
        for head in ("lp", "random"):
            M[f"ftgrid:{c}:{head}"] = f"{max(x['test_acc'] for x in g if x['head'] == head):.1f}"
        for reg in ("none", "ls", "mixup"):
            M[f"ftgrid:{c}:{reg}"] = f"{max(x['test_acc'] for x in g if x['reg'] == reg):.1f}"
        M[f"ftgrid:{c}:recipe"] = recipe(o["best"])
        rows.append(f"{name} & test set (oracle) & {recipe(o['best'])} & {min(tv):.1f}--{max(tv):.1f} & {MS(c, 'fto')} \\\\")
    if nseeds("fttransfer_" + c, "finetune_tuned_acc") or len(V(c, "ftx")):
        rows.append(f"{name} & recipe of \\simclr & -- & -- & {MS(c, 'ftx')} \\\\")
    rows.append(r"\addlinespace")
write("tab_ftsearch.tex", "\\begin{tabular}{lllcc}\n\\toprule\nencoder & recipe chosen on & chosen recipe & range of the selection score over the grid & "
      "test accuracy \\\\\n\\midrule\n" + ("\n".join(rows[:-1]) if rows else r"\multicolumn{5}{c}{--} \\")
      + "\n\\bottomrule\n\\end{tabular}")

# ----------------------------------------------------------------------------
# Table: where the labels should go (pipelines, CIFAR-10, 5,000 labels)
# ----------------------------------------------------------------------------
PIPE = [
    ("\\emph{labels inside the contrastive loss from the first step}", None, None, None),
    (r"\supcon, then FT-5k", "inside the loss", "supcon_lab_f0.1_e200", "ft"),
    (r"\supcon, then FT-5k (oracle recipe)", "inside the loss", "supcon_lab_f0.1_e200", "fto"),
    (r"\union, then FT-5k", "inside the loss", "union_f0.1_e200", "ft"),
    (r"\union, then FT-5k (oracle recipe)", "inside the loss", "union_f0.1_e200", "fto"),
    ("\\emph{labels after label-free pretraining}", None, None, None),
    (r"\simclr, then LP-5k", "after", "simclr_all_e200", "lp5"),
    (r"\simclr, then FT-5k", "after", "simclr_all_e200", "ft"),
    (r"\simclr, then FT-5k (selected recipe)", "after", "simclr_all_e200", "ftt"),
    (r"\simclr, then FT-5k (oracle recipe)", "after", "simclr_all_e200", "fto"),
    (r"BYOL, then FT-5k", "after", "byol_all_e200", "ft"),
    (r"\simclr, then \supcon on the 5k (485 steps), then LP-5k", "after, contrastive", "twostage_simclr200_supcon5", "lp5"),
    (r"\simclr, then \supcon on the 5k (485 steps), then FT-5k", "after, contrastive", "twostage_simclr200_supcon5", "ft"),
    (r"\simclr, then FT-5k, then self-distillation", "after", "simclr_all_e200", "dist"),
    ("\\emph{labels in a semi-supervised objective}", None, None, None),
    (r"FixMatch from scratch", "pseudo-labels", "fixmatch_f0.1", "clf"),
    (r"FixMatch from scratch, 3$\times$ budget", "pseudo-labels", "fixmatch_b3_f0.1", "clf"),
    (r"MixMatch from scratch", "pseudo-labels", "mixmatch_f0.1", "clf"),
    (r"\simclr, then FixMatch", "after, pseudo-labels", "simclr_all_e200", "fm"),
    (r"BYOL, then FixMatch", "after, pseudo-labels", "byol_all_e200", "fm"),
    (r"\supcon, then FixMatch", "both", "supcon_lab_f0.1_e200", "fm"),
    (r"\union, then FixMatch", "both", "union_f0.1_e200", "fm"),
    (r"\simclr, then \supcon (2,425 steps), then FixMatch", "both", "twostage_simclr200_supcon25", "fm"),
    ("\\emph{references}", None, None, None),
    (r"cross-entropy on the 5k, 9,700 steps", "supervised only", "ce_f0.1_e100", "clf"),
    (r"cross-entropy on the 5k, 19,400 steps", "supervised only", "ce_f0.1_e200", "clf"),
    (r"cross-entropy on all 50k labels", "all labels", "ce_f1.0_e100", "clf"),
]
rows = []
for name, where, c, p in PIPE:
    if c is None:
        rows.append(rf"\multicolumn{{4}}{{l}}{{{name}}} \\")
    else:
        rows.append(f"\\quad {name} & {where} & {len(V(c, p)) or '--'} & {MS(c, p)} \\\\")
write("tab_where.tex", "\\begin{tabular}{llcc}\n\\toprule\npipeline & where the labels enter & seeds & test accuracy \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# ----------------------------------------------------------------------------
# Table + figure: geometry on the labeled images, the other training images and the test set
# ----------------------------------------------------------------------------
GEO = [(r"\simclr", "simclr_all_e200", "seen, no label"), (r"\simclrsub", "simclr_sub_f0.1_e200", "not seen"),
       (r"\supcon", "supcon_lab_f0.1_e200", "not seen"), (r"\union", "union_f0.1_e200", "seen, no label"),
       ("cross-entropy on 5k", "ce_f0.1_e200", "not seen"), (r"\simclr{}$\to$\supcon", "twostage_simclr200_supcon5", "seen, no label")]
rows = []
for name, c, rest in GEO:
    a, b, t = (V(c, k) for k in ("cdnv_lab", "cdnv_rest", "cdnv_test"))
    ratio = f"{(t / a).mean():.1f}" if len(a) and len(a) == len(t) and (a > 1e-6).all() else ("$>$100" if len(a) and len(t) else "--")
    rows.append(f"{name} & {MS(c, 'cdnv_lab', 2)} & {MS(c, 'cdnv_rest', 2)} & {MS(c, 'cdnv_test', 2)} & {ratio} & "
                f"{MS(c, 'lp5_train')} & {MS(c, 'lp5')} \\\\")
    if ratio not in ("--",):
        M[f"{c}:cdnv_ratio"] = ratio
write("tab_geometry.tex", "\\begin{tabular}{lcccccc}\n\\toprule\n & \\multicolumn{3}{c}{CDNV ($\\downarrow$ tighter classes)} & & \\multicolumn{2}{c}{LP-5k accuracy} \\\\\n"
      "\\cmidrule(lr){2-4}\\cmidrule(lr){6-7}\nencoder (19,400 steps) & labeled 5k & other 45k & test 10k & test / labeled & on its 5k & test \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.28})
ax = axes[0]
labs = ["SimCLR", "SimCLR\n-sub", "SupCon", "SupCon\n+U", "CE\non 5k", "SimCLR\n→SupCon"]
w = 0.26
for j, (key, lab, col, hatch) in enumerate((("cdnv_lab", "labeled 5k", "#555555", None), ("cdnv_rest", "other 45k training images", "#aaaaaa", "//"),
                                            ("cdnv_test", "test images", "#e6e6e6", None))):
    ms_ = [(V(c, key).mean() if len(V(c, key)) else np.nan) for _, c, _ in GEO]
    sd_ = [(V(c, key).std(ddof=1) if len(V(c, key)) > 1 else 0) for _, c, _ in GEO]
    ms_ = [max(m_, 2e-3) if np.isfinite(m_) else m_ for m_ in ms_]
    ax.bar(np.arange(len(GEO)) + (j - 1) * w, ms_, w, yerr=sd_, label=lab, color=col, edgecolor="black", linewidth=0.4,
           hatch=hatch, error_kw={"elinewidth": 0.7, "capsize": 1.5})
ax.set_yscale("log")
ax.set_xticks(range(len(GEO)))
ax.set_xticklabels(labs, fontsize=6.5)
ax.set_ylabel("CDNV (log scale)")
ax.set_title("(a) CDNV on three sets of images", loc="left")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=6.5, columnspacing=1.0)
ax.grid(axis="x", visible=False)
ax = axes[1]
for k, cands, a_key in (("supcon", ["supcon_lab_f0.1_e1000", "supcon_lab_f0.1_e400", "supcon_lab_f0.1_e200"], "pool_cdnv"),
                        ("union", ["union_f0.1_e1000", "union_f0.1_e400", "union_f0.1_e200"], "labeled_cdnv"),
                        ("simclr", ["simclr_all_e1000", "simclr_all_e400", "simclr_all_e200"], "pool_cdnv")):
    for c in cands:
        curves = [RUN[f"{c}_s{s}"].get("geom_curve") for s in seeds(c)]
        curves = [g for g in curves if g and a_key in g[0]]
        if curves:
            xs = [p["step"] for p in curves[0]]
            R = np.array([[p["test_cdnv"] / max(p[a_key], 1e-9) for p in g] for g in curves if len(g) == len(xs)])
            col, mk, ls, lab = STYLE[k]
            ax.errorbar(xs, R.mean(0), yerr=R.std(0, ddof=1) if len(R) > 1 else None, color=col, marker=mk, ls=ls,
                        label=lab.split(" (")[0], capsize=2, elinewidth=0.8, markersize=3)
            M[f"geomcurve:{k}"] = c
            break
ax.set_xscale("log")
ax.minorticks_off()
tk = [t for t in (2425, 4850, 9700, 19400, 38800, 97000) if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
ax.set_xticks(tk)
ax.set_xticklabels([f"{t / 1000:.1f}k".replace(".0k", "k") for t in tk])
ax.axhline(1, color="black", lw=0.6, ls=":")
ax.set_xlabel("gradient steps")
ax.set_ylabel("CDNV(test) / CDNV(labeled)")
ax.set_title("(b) the gap during training", loc="left")
ax.legend(frameon=False, fontsize=6.5)
fig.savefig(os.path.join(FIG, "fig_geometry.pdf"))
plt.close(fig)

# ----------------------------------------------------------------------------
# Coverage intervention: pools of fixed size whose coverage is varied
# ----------------------------------------------------------------------------
DIV = {}
dj = os.path.join(ROOT, "results", "diversity.json")
if os.path.exists(dj):
    DIV = json.load(open(dj))
KINDS = [("random", "random", ""), ("mixkcenter25", "25% k-center", "mixkcenter25_"), ("mixkcenter50", "50% k-center", "mixkcenter50_"),
         ("mixkcenter75", "75% k-center", "mixkcenter75_"), ("kcenter", "k-center", "kcenter_"),
         ("mixcluster5025", "25% redundant", "mixcluster5025_"), ("mixcluster5050", "50% redundant", "mixcluster5050_"),
         ("mixcluster5075", "75% redundant", "mixcluster5075_"), ("cluster10", "redundant (k=10)", "cluster10_"),
         ("cluster50", "redundant (k=50)", "cluster50_")]


def pool_stat(space, kind, seed, key):
    try:
        return DIV[space]["pools"][f"{kind}_f0.1_s{seed}"][key]
    except KeyError:
        return None


PTS = []   # one point per (kind, seed): pool statistics and accuracies
for kind, label, infix in KINDS:
    for s in range(5):
        row = {"kind": kind, "label": label, "seed": s}
        for space in ("simclr", "supervised", "pixel"):
            for key in ("radius", "spread"):
                row[f"{key}_{space}"] = pool_stat(space, kind, s, key)
        row["supcon"] = metric(f"supcon_lab_{infix}f0.1_e100_s{s}", "lp50")
        row["supcon_lp5"] = metric(f"supcon_lab_{infix}f0.1_e100_s{s}", "lp5")
        row["sub"] = metric(f"simclr_sub_{infix}f0.1_e100_s{s}", "lp50")
        if row["radius_simclr"] is not None and (row["supcon"] is not None or row["sub"] is not None):
            PTS.append(row)


def agg(kind, key):
    v = [p[key] for p in PTS if p["kind"] == kind and p.get(key) is not None]
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v)) if v else None


rows = []
for kind, label, _ in KINDS:
    cells = []
    for key, d in (("radius_simclr", 3), ("radius_supervised", 3), ("spread_simclr", 3), ("supcon", 1), ("supcon_lp5", 1), ("sub", 1)):
        a = agg(kind, key)
        cells.append("--" if a is None else (f"{a[0]:.{d}f}" + (rf"{{\scriptsize$\,\pm\,${a[1]:.{d}f}}}" if a[2] > 1 and d == 1 else "")))
    n = agg(kind, "supcon")
    if any(c != "--" for c in cells[3:]):
        rows.append(f"{label} & {n[2] if n else '--'} & " + " & ".join(cells) + r" \\")
write("tab_coverage.tex", "\\begin{tabular}{lccccccc}\n\\toprule\n & & \\multicolumn{2}{c}{coverage radius $\\downarrow$} & spread & "
      "\\multicolumn{2}{c}{\\supcon on the pool} & \\simclr on the pool \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){6-7}\n"
      "pool (5,000 images) & seeds & \\simclr space & supervised space & & LP-50k & LP-5k & LP-50k \\\\\n\\midrule\n"
      + ("\n".join(rows) if rows else r"\multicolumn{8}{c}{--} \\") + "\n\\bottomrule\n\\end{tabular}")


def spearman(xk, yk, pts):
    x = [p[xk] for p in pts if p.get(xk) is not None and p.get(yk) is not None]
    y = [p[yk] for p in pts if p.get(xk) is not None and p.get(yk) is not None]
    if len(x) < 5:
        return None
    r = sps.spearmanr(x, y)
    return r.statistic, r.pvalue, len(x)


for subset, pts in (("all", PTS), ("nokc", [p for p in PTS if "kcenter" not in p["kind"]]),
                    ("red", [p for p in PTS if "kcenter" not in p["kind"]]),
                    ("kc", [p for p in PTS if "cluster" not in p["kind"]])):
    for xk in ("radius_simclr", "radius_supervised", "spread_simclr"):
        for yk in ("supcon", "sub", "supcon_lp5"):
            r = spearman(xk, yk, pts)
            if r:
                M[f"rho:{subset}:{xk}:{yk}"] = f"{r[0]:+.2f}"
                M[f"rho:{subset}:{xk}:{yk}:p"] = "$<$0.001" if r[1] < 0.001 else f"{r[1]:.3f}"
                M[f"rho:{subset}:{xk}:{yk}:n"] = str(r[2])

# partial rank correlation between coverage radius and accuracy across pool SIZES, controlling for log n
if DIV:
    xs, ys, ns = [], [], []
    for f in (0.01, 0.05, 0.1, 0.2):
        for s in range(5):
            rad = None
            try:
                rad = DIV["simclr"]["pools"][f"random_f{f}_s{s}"]["radius"]
            except KeyError:
                pass
            acc = metric(f"supcon_lab_f{f}_e100_s{s}", "lp50")
            if rad is not None and acc is not None:
                xs.append(rad), ys.append(acc), ns.append(np.log(f))
    for p in PTS:
        if p["kind"] != "random" and p["supcon"] is not None:
            xs.append(p["radius_simclr"]), ys.append(p["supcon"]), ns.append(np.log(0.1))
    if len(xs) >= 8:
        rx, ry, rn = (sps.rankdata(v) for v in (xs, ys, ns))
        cxy, cxn, cyn = np.corrcoef(rx, ry)[0, 1], np.corrcoef(rx, rn)[0, 1], np.corrcoef(ry, rn)[0, 1]
        M["rho:partial"] = f"{(cxy - cxn * cyn) / np.sqrt((1 - cxn ** 2) * (1 - cyn ** 2)):+.2f}"
        M["rho:partial:raw"] = f"{cxy:+.2f}"
        M["rho:partial:n"] = str(len(xs))

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharex=True)
fam = {"random": ("#000000", "o"), "kcenter": ("#009E73", "^"), "cluster": ("#D55E00", "s")}
for ax, (yk, title) in zip(axes, (("supcon", "(a) SupCon trained on the pool"), ("sub", "(b) SimCLR trained on the pool"))):
    for kind, label, _ in KINDS:
        a, r = agg(kind, yk), agg(kind, "radius_simclr")
        if a is None or r is None:
            continue
        col, mk = fam["kcenter" if "kcenter" in kind else "cluster" if "cluster" in kind else "random"]
        ax.errorbar(r[0], a[0], yerr=a[1], xerr=r[1], color=col, marker=mk, capsize=2, elinewidth=0.8, ls="none")
        ax.annotate(label, (r[0], a[0]), textcoords="offset points", xytext=(4, 3), fontsize=6, color=col)
    ax.set_xlabel("coverage radius of the pool (larger = worse coverage)")
    ax.set_title(title, loc="left")
axes[0].set_ylabel("LP-50k accuracy (%)")
fig.savefig(os.path.join(FIG, "fig_coverage.pdf"))
plt.close(fig)

# ----------------------------------------------------------------------------
# Table: labels make colour jitter less necessary
# ----------------------------------------------------------------------------
rows = []
for name, full, nojit in ((r"\simclr", "simclr_all_e100", "simclr_exp6_e100"), (r"\supcon", "supcon_lab_f0.1_e100", "supcon_lab_f0.1_exp6_e100"),
                          (r"\union", "union_f0.1_e100", "union_f0.1_exp6_e100")):
    common = sorted(set(seeds(full)) & set(seeds(nojit)))
    a, b = V(full, "lp50", common), V(nojit, "lp50", common)
    if len(a) and len(a) == len(b):
        dlt_ = b - a
        M[f"jitter:{full}"] = f"{abs(dlt_.mean()):.1f}"
        M[f"jitter:{full}:n"] = str(len(a))
        sd = lambda v: (rf"{{\scriptsize$\,\pm\,${v.std(ddof=1):.1f}}}" if len(v) > 1 else "")   # noqa: E731
        rows.append(f"{name} & {len(a)} & {a.mean():.1f}{sd(a)} & {b.mean():.1f}{sd(b)} & {dlt_.mean():+.1f}{sd(dlt_)} \\\\")
write("tab_jitter.tex", "\\begin{tabular}{lcccc}\n\\toprule\nmethod & seeds & full augmentation & without colour jitter & change \\\\\n\\midrule\n"
      + ("\n".join(rows) if rows else r"\multicolumn{5}{c}{--} \\") + "\n\\bottomrule\n\\end{tabular}")

# ----------------------------------------------------------------------------
# Table: is the SupCon plateau a regularisation failure? (38,800 steps)
# ----------------------------------------------------------------------------
REG = [("default (weight decay $10^{-4}$)", "supcon_lab_f0.1_e400"), ("weight decay $5\\times10^{-4}$", "supcon_lab_f0.1_wd0.0005_e400"),
       ("weight decay $10^{-3}$", "supcon_lab_f0.1_wd0.001_e400"), ("stronger augmentation", "supcon_lab_f0.1_strongaug_e400"),
       ("weight decay $5\\times10^{-4}$ + stronger augmentation", "supcon_lab_f0.1_wd0.0005_strongaug_e400")]
rows = []
for name, c in REG:
    if not nseeds(c):
        continue
    best_knn = [max(a for _, a in RUN[f"{c}_s{s}"]["knn_curve"]) for s in seeds(c)]
    M[f"{c}:bestknn"] = f"{np.mean(best_knn):.1f}"
    rows.append(f"{name} & {nseeds(c)} & {MS(c, 'lp50')} & {MS(c, 'lp5')} & {MS(c, 'knn')} & {np.mean(best_knn):.1f} \\\\")
write("tab_reg.tex", "\\begin{tabular}{lccccc}\n\\toprule\n\\supcon, 38,800 steps & seeds & LP-50k & LP-5k & $k$NN-50k (final) & $k$NN-50k (best checkpoint) \\\\\n\\midrule\n"
      + ("\n".join(rows) if rows else r"\multicolumn{6}{c}{--} \\") + "\n\\bottomrule\n\\end{tabular}")
for k, tmpl in ARMS.items():      # best checkpoint along the longest run of every arm (an upper bound on early stopping)
    for E in (1000, 400, 200):
        c = tmpl.format(E=E)
        if nseeds(c):
            M[f"bestknn:{k}"] = f"{np.mean([max(a for _, a in RUN[f'{c}_s{s}']['knn_curve']) for s in seeds(c)]):.1f}"
            M[f"bestknn:{k}:E"] = f"{97 * E:,}"
            break

# ----------------------------------------------------------------------------
# Table: steps, architecture and learning rate (ResNet-18 vs ResNet-50)
# ----------------------------------------------------------------------------
R50 = [("ResNet-18", r"\supcon", "900", "0.25", "supcon_lab_f0.1_steps900"), ("ResNet-18", r"\supcon", "970", "0.25", "supcon_lab_f0.1_e10"),
       ("ResNet-18", r"\supcon", "9,700", "0.25", "supcon_lab_f0.1_e100"), ("ResNet-18", r"\supcon", "19,400", "0.25", "supcon_lab_f0.1_e200"),
       ("ResNet-18", r"\simclr", "970", "0.25", "simclr_all_e10"), ("ResNet-18", r"\simclr", "9,700", "0.25", "simclr_all_e100"),
       ("ResNet-18", r"\simclr", "19,400", "0.25", "simclr_all_e200"),
       ("ResNet-50", r"\supcon", "900", "0.25", "r50_supcon_lab_f0.1_steps900"), ("ResNet-50", r"\supcon", "900", "0.05", "r50_supcon_lab_f0.1_steps900_lr0.05"),
       ("ResNet-50", r"\supcon", "9,700", "0.25", "r50_supcon_lab_f0.1_e100"), ("ResNet-50", r"\supcon", "19,400", "0.25", "r50_supcon_lab_f0.1_e200"),
       ("ResNet-50", r"\simclr", "970", "0.25", "r50_simclr_all_e10"), ("ResNet-50", r"\simclr", "9,700", "0.25", "r50_simclr_all_e100"),
       ("ResNet-50", r"\simclr", "19,400", "0.25", "r50_simclr_all_e200")]
rows = [f"{a} & {m} & {s} & {lr} & {nseeds(c)} & {MS(c, 'lp50')} & {MS(c, 'lp5')} \\\\" for a, m, s, lr, c in R50 if nseeds(c)]
write("tab_r50.tex", "\\begin{tabular}{llrcccc}\n\\toprule\nencoder & method & steps & learning rate & seeds & LP-50k & LP-5k \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# ----------------------------------------------------------------------------
# Table: compute accounting (per step and for the reference budget)
# ----------------------------------------------------------------------------
COMP = [(r"\simclr, \supcon, \union", "1,024", "1,024", "1,024", "9,700", "simclr_all_e100"),
        ("BYOL", "1,024", "2,048", "1,024", "9,700", "byol_all_e100"),
        ("SimSiam", "1,024", "1,024", "1,024", "9,700", "simsiam_all_e100"),
        ("cross-entropy", "512", "512", "512", "9,700", "ce_f0.1_e100"),
        ("FixMatch", "64 + 448 + 448", "960", "960", "10,347", "fixmatch_f0.1"),
        ("MixMatch", "64 + 2$\\times$64", "320", "192", "51,733", "mixmatch_f0.1"),
        ("fine-tuning on the 5k (FT-5k)", "128", "128", "128", "2,344", "ft_simclr_all_e100")]
rows = []
for name, views, fw, bw, steps, c in COMP:
    t = V(c, "time")
    tot = float(fw.replace(",", "")) * float(steps.replace(",", "")) / 1e6
    rows.append(f"{name} & {views} & {fw} & {bw} & {steps} & {tot:.1f} & {t.min():.0f} \\\\" if len(t) else
                f"{name} & {views} & {fw} & {bw} & {steps} & {tot:.1f} & -- \\\\")
write("tab_compute.tex", "\\begin{tabular}{lcccrcc}\n\\toprule\n & \\multicolumn{3}{c}{per step} & & \\multicolumn{2}{c}{reference budget} \\\\\n"
      "\\cmidrule(lr){2-4}\\cmidrule(lr){6-7}\nmethod & augmented views & forward passes & backward passes & steps & forward passes ($10^6$) & minutes \\\\\n"
      "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# ----------------------------------------------------------------------------
# Appendix tables: colour reliance of SimCLR, and the validation-split choice of the SupCon+U variant
# ----------------------------------------------------------------------------
COL = [("crop + flip", "simclr_exp2_e100"), ("crop + flip + colour jitter", "simclr_exp5_e100"),
       ("crop + flip + grayscale", "simclr_exp6_e100"), ("crop + flip + colour jitter + grayscale (default)", "simclr_all_e100")]
rows = []
for name, c in COL:
    if nseeds(c):
        rows.append(f"{name} & {nseeds(c)} & {MS(c, 'lp50')} & {MS(c, 'linear_acc_gray')} & {fmt(mean(c, 'cka_hist'), 2)} & "
                    f"{fmt(mean(c, 'cka_label'), 2)} & {fmt(mean(c, 'hist_r2'), 2)} \\\\")
hp = os.path.join(ROOT, "results", "hist_probe.json")
if os.path.exists(hp):
    h = json.load(open(hp))
    if "cifar10_bins8" in h:
        M["histprobe"] = f"{h['cifar10_bins8']['acc']:.1f}"
write("tab_color.tex", "\\begin{tabular}{lcccccc}\n\\toprule\n & & \\multicolumn{2}{c}{LP-50k accuracy} & \\multicolumn{3}{c}{colour reliance} \\\\\n"
      "\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}\n\\simclr augmentation & seeds & colour test & gray test & CKA(histogram) & CKA(label) & histogram $R^2$ \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
PIL = [(r"\simclr, $\tau=0.5$", "pilot_simclr_t0.5"), (r"\simclr, $\tau=0.2$", "pilot_simclr_t0.2"), (r"\simclr, $\tau=0.1$", "pilot_simclr_t0.1"),
       (r"\supcon, $\tau=0.1$", "pilot_supconL_t0.1"), (r"\supcon, $\tau=0.5$", "pilot_supconL_t0.5"),
       (r"\union, $\tau=0.1$ for all anchors", "pilot_union_t0.1"), (r"\union, $\tau=0.5$ for all anchors", "pilot_union_t0.5"),
       (r"\union, $\tau_\ell=0.1$, $\tau_u=0.5$, uniform batches", "pilot_union_mixed"),
       (r"\union, $\tau_\ell=0.1$, $\tau_u=0.5$, 50\% labeled per batch", "pilot_union_mixed_sh0.5"),
       (r"\union, $\tau_\ell=0.1$, $\tau_u=0.5$, 25\% labeled per batch (chosen)", "pilot_union_mixed_sh0.25")]
rows = []
for name, c in PIL:
    r = RUN.get(c) or RUN.get(c + "_s0")
    if r:
        m = r["metrics"]
        rows.append(f"{name} & {m['linear_acc']:.1f} & {m['knn_acc']:.1f} & {fmt(m.get('linear_lab_acc', float('nan')))} \\\\")
write("tab_pilot.tex", "\\begin{tabular}{lccc}\n\\toprule\nconfiguration (validation split, 4,850 steps) & LP (all labels) & $k$NN & LP (10\\% labels) \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# ----------------------------------------------------------------------------
# a few derived numbers used in the text
# ----------------------------------------------------------------------------
for E in (10, 25, 50, 100, 200, 400, 1000):
    s, c, u, sub = (ARMS[k].format(E=E) for k in ("simclr", "supcon", "union", "sub"))
    for p in ("lp50", "lp5", "ft", "fts", "fto", "fm", "knn"):
        diffmacro(f"simclr-supcon:{p}:e{E}", s, p, c, p)
        diffmacro(f"simclr-union:{p}:e{E}", s, p, u, p)
        diffmacro(f"union-supcon:{p}:e{E}", u, p, c, p)
        diffmacro(f"simclr-sub:{p}:e{E}", s, p, sub, p)
        diffmacro(f"supcon-sub:{p}:e{E}", c, p, sub, p)
for c in ("simclr_all_e200", "supcon_lab_f0.1_e200", "union_f0.1_e200", "byol_all_e200"):
    diffmacro(f"ftt-ft:{c}", c, "ftt", c, "ft")
    diffmacro(f"ftt-ftd:{c}", c, "ftt", c, "ftd")
    diffmacro(f"fto-ft:{c}", c, "fto", c, "ft")
    diffmacro(f"ftx-ft:{c}", c, "ftx", c, "ft")
    diffmacro(f"fm-ft:{c}", c, "fm", c, "ft")
diffmacro("twostage-simclr:lp5", "twostage_simclr200_supcon5", "lp5", "simclr_all_e200", "lp5")
diffmacro("twostage-simclr:lp50", "twostage_simclr200_supcon5", "lp50", "simclr_all_e200", "lp50")
diffmacro("twostage-simclr:ft", "twostage_simclr200_supcon5", "ft", "simclr_all_e200", "ft")
for pre, tag in (("c100_", "c100"), ("r50_", "r50")):
    for E in (100, 200):
        for p in ("lp50", "lp5", "ft"):
            diffmacro(f"{tag}:simclr-supcon:{p}:e{E}", f"{pre}simclr_all_e{E}", p, f"{pre}supcon_lab_f0.1_e{E}", p)
            diffmacro(f"{tag}:simclr-union:{p}:e{E}", f"{pre}simclr_all_e{E}", p, f"{pre}union_f0.1_e{E}", p)
for E in (100, 200):
    for p in ("lp5", "ft", "knn", "fm"):
        diffmacro(f"stl:simclr-supcon:{p}:e{E}", f"stl_simclr_all_e{E}", p, f"stl_supcon_lab_e{E}", p)
        diffmacro(f"stl:simclr-union:{p}:e{E}", f"stl_simclr_all_e{E}", p, f"stl_union_e{E}", p)
        diffmacro(f"stl:simclr-sub:{p}:e{E}", f"stl_simclr_all_e{E}", p, f"stl_simclr_sub_e{E}", p)
diffmacro("steps900", "supcon_lab_f0.1_e100", "lp50", "supcon_lab_f0.1_steps900", "lp50")
diffmacro("twostage-supcon:lp5", "twostage_simclr200_supcon5", "lp5", "supcon_lab_f0.1_e200", "lp5")
# the longest budget with at least two seeds for both SimCLR and SupCon
for E in (1000, 400, 200):
    if nseeds(ARMS["simclr"].format(E=E)) >= 2 and nseeds(ARMS["supcon"].format(E=E)) >= 2:
        M["long:steps"] = f"{97 * E:,}"
        for k in ARMS:
            if nseeds(ARMS[k].format(E=E)):
                M[f"long:{k}"] = f"{mean(ARMS[k].format(E=E), 'lp50'):.1f}"
        for p in ("lp50", "lp5", "ft"):
            x, y = mean(ARMS["simclr"].format(E=E), p), mean(ARMS["supcon"].format(E=E), p)
            if np.isfinite(x) and np.isfinite(y):
                M["long:d:simclr-supcon" + ("" if p == "lp50" else ":" + p)] = f"{abs(x - y):.1f}"
        break
regs = [mean(c, "lp50") for _, c in REG[1:] if nseeds(c)]
if regs:
    M["reg:best"] = f"{max(regs):.1f}"
M["nruns"] = str(len(RUN))
M["gpuhours"] = f"{sum(r.get('train_time_s', 0) for r in RUN.values()) / 3600:.0f}"

lines = [r"\newcommand{\R}[1]{\ifcsname R@#1\endcsname\csname R@#1\endcsname\else\textbf{??}\fi}"]
for k in sorted(M):
    lines.append(rf"\expandafter\def\csname R@{k}\endcsname{{{M[k]}}}")
write("numbers.tex", "\n".join(lines))
print(f"{len(RUN)} runs, {len(CONFIGS)} configurations, {len(M)} macros, {len(STATS)} tests")
