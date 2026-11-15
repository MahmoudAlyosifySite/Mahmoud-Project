"""Build every number, table and figure of the paper from results/.

    python scripts/make_assets.py

Writes
    paper/generated/numbers.tex   \\newcommand macros for every number quoted in the text
    paper/generated/*.tex         tables
    paper/figures/*.pdf           figures
    results/summary.csv           one row per run
Missing runs are skipped (their macros print "--"), so the script can be run at any time.
"""
import glob
import json
import os
import re
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from stats import compare, test_labels   # noqa: E402

RUNS = os.path.join(ROOT, "results", "runs")
GEN = os.path.join(ROOT, "paper", "generated")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(GEN, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# entity colours (fixed per method, never by rank)
COL = {"simclr": "#2a78d6", "supcon": "#eb6834", "union": "#1baf7a", "sub": "#7f7f7f",
       "ce": "#6250d6", "fixmatch": "#e87ba4", "mixmatch": "#008300", "byol": "#eda100", "simsiam": "#e34948"}
INK, MUTED, GRID = "#1f1f1f", "#5f5f5f", "#e6e6e6"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "lines.linewidth": 1.6, "lines.markersize": 5, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------

def load():
    rows = []
    for path in sorted(glob.glob(os.path.join(RUNS, "*", "result.json"))):
        d = os.path.dirname(path)
        name = os.path.basename(d)
        if name.startswith("pilot") or name.startswith("smoke"):
            continue
        r = json.load(open(path))
        row = {"name": name, "config": re.sub(r"_s\d+$", "", name), "seed": r["args"].get("seed"),
               "train_min": r.get("train_time_s", 0) / 60, "steps": r.get("steps")}
        row.update(r["metrics"])
        pj = os.path.join(d, "probes.json")
        if os.path.exists(pj):
            row.update({"p_" + k: v for k, v in json.load(open(pj)).items()})
        rows.append(row)
    return pd.DataFrame(rows)


DF = load()
# every macro used in the paper exists even before its run has finished ("--")
DEFAULT_MACROS = """radiusTen spreadTen radiusOne spreadOne radiusFull spreadFull radiusKc spreadKc neffKc neffClFifty
oodSimclrCifar oodSimclrOod oodSimclrPCifar oodSimclrPOod oodSimclrTopWrong oodUnionCifar oodUnionOod oodUnionPCifar
oodUnionPOod oodUnionTopWrong oodSupconCifar oodSupconOod oodSupconPCifar oodSupconPOod oodSupconTopWrong oodCeCifar
oodCeOod oodCePCifar oodCePOod oodCeTopWrong histProbe clipCifar clipResSixteen clipResThirtyTwo clipResSixtyFour
clipResOneTwoEight clipResTwoTwoFour clipTest nOod""".split()
DF.to_csv(os.path.join(ROOT, "results", "summary.csv"), index=False)
MACROS = {}


def vals(config, key):
    if DF.empty or key not in DF.columns:
        return np.array([])
    return DF.loc[DF["config"] == config, key].dropna().to_numpy(dtype=float)


def ms(config, key, digits=1, pm=True):
    v = vals(config, key)
    if len(v) == 0:
        return "--"
    if len(v) == 1 or not pm:
        return f"{v.mean():.{digits}f}"
    return f"{v.mean():.{digits}f}{{\\scriptsize$\\pm${v.std(ddof=1):.{digits}f}}}"


def mean(config, key):
    v = vals(config, key)
    return float(v.mean()) if len(v) else float("nan")


def macro(name, value):
    MACROS[name] = value


def fmt(x, d=1, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


# ----------------------------------------------------------------------------
# headline numbers + significance tests
# ----------------------------------------------------------------------------
CORE = {"simclr": "simclr_all_e100", "supcon": "supcon_lab_f0.1_e100",
        "sub": "simclr_sub_f0.1_e100", "union": "union_f0.1_e100"}
Y = test_labels()


def stat_macros(tag, a, b, key):
    try:
        r = compare(a, b, Y, key, RUNS, n_boot=10000)
    except (FileNotFoundError, ValueError):
        return None
    macro(f"diff{tag}", fmt(r["diff"], 1, True))
    macro(f"ci{tag}", f"[{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]")
    p = r["p_boot"]
    macro(f"p{tag}", "<10^{-4}" if p <= 1e-4 else f"={p:.3f}" if p >= 0.001 else f"={p:.1e}")
    return r


STATS = []
for key, label in (("linear", "Lin"), ("knn", "Knn")):
    for tag, a, b in (("UnionSimclr", "union", "simclr"), ("UnionSupcon", "union", "supcon"),
                      ("SupconSimclr", "supcon", "simclr"), ("SimclrSub", "simclr", "sub"),
                      ("SupconSub", "supcon", "sub")):
        r = stat_macros(tag + label, CORE[a], CORE[b], key)
        if r:
            STATS.append((label, a, b, r))
for tag, a, b in (("UnionSimclr", "union", "simclr"), ("UnionSupcon", "union", "supcon"), ("SupconSimclr", "supcon", "simclr")):
    for key, label in (("finetune", "Ft"),):
        r = stat_macros(tag + label, "ft_" + CORE[a], "ft_" + CORE[b], key)
        if r:
            STATS.append((label, a, b, r))
r = stat_macros("LongLin", "simclr_all_e200", "supcon_lab_f0.1_e200", "linear")
if r:
    STATS.append(("Lin", "simclr@e200", "supcon@e200", r))
r = stat_macros("LongUnionLin", "union_f0.1_e200", "supcon_lab_f0.1_e200", "linear")
if r:
    STATS.append(("Lin", "union@e200", "supcon@e200", r))
r = stat_macros("LongSimclrUnionLin", "simclr_all_e200", "union_f0.1_e200", "linear")
if r:
    STATS.append(("Lin", "simclr@e200", "union@e200", r))
for tag, a_, b_ in (("FtLongSupcon", "ft_simclr_all_e200", "ft_supcon_lab_f0.1_e200"),
                    ("FtLongUnion", "ft_simclr_all_e200", "ft_union_f0.1_e200")):
    r = stat_macros(tag, a_, b_, "finetune")
    if r:
        STATS.append(("Ft", a_.replace("ft_", "") + "@ft", b_.replace("ft_", "") + "@ft", r))
r = stat_macros("ShortLin", "supcon_lab_f0.1_e10", "simclr_all_e10", "linear")
if r:
    STATS.append(("Lin", "supcon@e10", "simclr@e10", r))
# the original report's budget: SupCon trained 100 epochs over 5,000 images (= 10 full-data epochs)
r = stat_macros("BudgetLin", "supcon_lab_f0.1_e10", "simclr_all_e100", "linear")
if r:
    STATS.append(("Lin", "supcon@e10", "simclr@e100", r))

for m, c in CORE.items():
    for key, short in (("linear_acc", "Lin"), ("knn_acc", "Knn"), ("p_linear_0.1", "LinTen"),
                       ("seen_cdnv", "CdnvSeen"), ("heldout_cdnv", "CdnvHeld"), ("test_cdnv", "CdnvTest"),
                       ("test_ncc_acc", "Ncc"), ("test_etf_dev", "Etf")):
        macro(f"{m}{short}", ms(c, key, 2 if "cdnv" in key or "etf" in key else 1, pm=False))
    macro(f"{m}LinPm", ms(c, "linear_acc"))
    macro(f"{m}FtPm", ms("ft_" + c, "finetune_acc"))
    macro(f"{m}Ft", ms("ft_" + c, "finetune_acc", pm=False))
macro("supconTenLin", ms("supcon_lab_f0.1_e10", "linear_acc", pm=False))
macro("supconTenLinPm", ms("supcon_lab_f0.1_e10", "linear_acc"))
macro("simclrTenLin", ms("simclr_all_e10", "linear_acc", pm=False))
macro("gapBudget", fmt(mean("simclr_all_e100", "linear_acc") - mean("supcon_lab_f0.1_e10", "linear_acc")))
macro("gapMatched", fmt(mean("simclr_all_e100", "linear_acc") - mean("supcon_lab_f0.1_e100", "linear_acc"), 1, True))
macro("rFiftySupconTen", ms("r50_supcon_lab_f0.1_e10", "linear_acc", pm=False))
macro("rFiftySupconTenLow", ms("r50_supcon_lab_f0.1_e10_lr0.05", "linear_acc", pm=False))
macro("rFiftySimclrTen", ms("r50_simclr_all_e10", "linear_acc", pm=False))
macro("gapBudgetSimclr", fmt(mean("supcon_lab_f0.1_e10", "linear_acc") - mean("simclr_all_e10", "linear_acc"), 1, True))
for e, w in ((25, "TwentyFive"), (50, "Fifty"), (200, "TwoHundred")):
    macro(f"supcon{w}Lin", ms(f"supcon_lab_f0.1_e{e}", "linear_acc", pm=False))
    macro(f"simclr{w}Lin", ms(f"simclr_all_e{e}", "linear_acc", pm=False))
    macro(f"union{w}Lin", ms(f"union_f0.1_e{e}", "linear_acc", pm=False))
    macro(f"supcon{w}LinPm", ms(f"supcon_lab_f0.1_e{e}", "linear_acc"))
    macro(f"simclr{w}LinPm", ms(f"simclr_all_e{e}", "linear_acc"))
    macro(f"union{w}LinPm", ms(f"union_f0.1_e{e}", "linear_acc"))
    macro(f"supcon{w}Knn", ms(f"supcon_lab_f0.1_e{e}", "knn_acc", pm=False))
    macro(f"simclr{w}Knn", ms(f"simclr_all_e{e}", "knn_acc", pm=False))
    macro(f"union{w}Knn", ms(f"union_f0.1_e{e}", "knn_acc", pm=False))
macro("unionUnifTwoHundredLin", ms("unionunif_f0.1_e200", "linear_acc", pm=False))
macro("unionUnifLin", ms("unionunif_f0.1_e100", "linear_acc", pm=False))
macro("gapLong", fmt(mean("simclr_all_e200", "linear_acc") - mean("supcon_lab_f0.1_e200", "linear_acc"), 1))
macro("gapShort", fmt(mean("supcon_lab_f0.1_e10", "linear_acc") - mean("simclr_all_e10", "linear_acc"), 1))
macro("ceTen", ms("ce_f0.1_e100", "classifier_acc", pm=False))
macro("ceFull", ms("ce_f1.0_e100", "classifier_acc", pm=False))
for k in ("fixmatch", "mixmatch"):
    macro(f"{k}Acc", ms(f"{k}_f0.1", "classifier_acc", pm=False))
    macro(f"{k}Lin", ms(f"{k}_f0.1", "linear_acc", pm=False))
for k in ("byol", "simsiam"):
    macro(f"{k}Lin", ms(f"{k}_all_e100", "linear_acc", pm=False))
    macro(f"{k}Knn", ms(f"{k}_all_e100", "knn_acc", pm=False))
    macro(f"{k}Ft", ms(f"ft_{k}_all_e100", "finetune_acc", pm=False))
macro("nSeeds", str(len(vals(CORE["simclr"], "linear_acc"))))

# ----------------------------------------------------------------------------
# Table: main results at two budgets and three protocols
# ----------------------------------------------------------------------------
def write(name, text):
    with open(os.path.join(GEN, name), "w") as f:
        f.write(text)


MAIN_ROWS = [
    ("SimCLR", "all 50k", "--", "simclr_all_e{}"),
    ("SimCLR (subset)", "5k", "--", "simclr_sub_f0.1_e{}"),
    ("SupCon (label-only)", "5k", "5k", "supcon_lab_f0.1_e{}"),
    ("Union", "all 50k", "5k", "union_f0.1_e{}"),
    ("Union, uniform batches", "all 50k", "5k", "unionunif_f0.1_e{}"),
]
PROTS = (("linear_acc", None), ("p_linear_0.1", None), ("finetune_acc", "ft_"))


def cell(cfg, key, prefix, best):
    c = (prefix or "") + cfg
    out = ms(c, key)
    m = mean(c, key)
    if np.isfinite(m) and best is not None and abs(m - best) < 0.05:
        out = r"\textbf{" + out + "}"
    return out


lines = [r"\begin{tabular}{l cc ccc ccc}", r"\toprule",
         r" & \multicolumn{2}{c}{pretraining uses} & \multicolumn{3}{c}{9,700 steps (100 epochs)} & \multicolumn{3}{c}{19,400 steps (200 epochs)} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
         r"method & images & labels & probe, 50k & probe, 5k & fine-tune, 5k & probe, 50k & probe, 5k & fine-tune, 5k \\", r"\midrule"]
best = {}
for e in (100, 200):
    for key, prefix in PROTS:
        ms_ = [mean((prefix or "") + r[3].format(e), key) for r in MAIN_ROWS]
        ms_ = [m for m in ms_ if np.isfinite(m)]
        best[(e, key)] = max(ms_) if ms_ else None
for label, imgs, labs, pat in MAIN_ROWS:
    cells = [cell(pat.format(e), key, prefix, best[(e, key)]) for e in (100, 200) for key, prefix in PROTS]
    lines.append(f"{label} & {imgs} & {labs} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("table_main.tex", "\n".join(lines) + "\n")

# kNN and short-budget table (appendix)
lines = [r"\begin{tabular}{l ccccc}", r"\toprule", r"method & 970 & 2,425 & 4,850 & 9,700 & 19,400 steps \\", r"\midrule"]
for label, pat in (("SimCLR", "simclr_all_e{}"), ("SupCon (label-only)", "supcon_lab_f0.1_e{}"), ("Union", "union_f0.1_e{}")):
    for key, nm in (("linear_acc", "linear"), ("knn_acc", "$k$NN")):
        lines.append(f"{label}, {nm} & " + " & ".join(ms(pat.format(e), key) for e in (10, 25, 50, 100, 200)) + r" \\")
lines += [r"\midrule",
          r"ResNet-50 SupCon (label-only), lr 0.25 & " + ms("r50_supcon_lab_f0.1_e10", "linear_acc") + r" & & & " + ms("r50_supcon_lab_f0.1_e100", "linear_acc") + r" & \\",
          r"ResNet-50 SupCon (label-only), lr 0.05 & " + ms("r50_supcon_lab_f0.1_e10_lr0.05", "linear_acc") + r" & & & & \\",
          r"ResNet-50 SimCLR & " + ms("r50_simclr_all_e10", "linear_acc") + r" & & & " + ms("r50_simclr_all_e100", "linear_acc") + r" & \\",
          r"\bottomrule", r"\end{tabular}"]
write("table_budget.tex", "\n".join(lines) + "\n")

for e, w in ((100, ""), (200, "TwoHundred")):
    for k, pat in (("simclr", "simclr_all_e{}"), ("supcon", "supcon_lab_f0.1_e{}"), ("union", "union_f0.1_e{}"),
                   ("unionUnif", "unionunif_f0.1_e{}"), ("sub", "simclr_sub_f0.1_e{}")):
        macro(f"{k}{w}Ft", ms("ft_" + pat.format(e), "finetune_acc", pm=False))
        macro(f"{k}{w}FtPm", ms("ft_" + pat.format(e), "finetune_acc"))
        macro(f"{k}{w}ProbeFive", ms(pat.format(e), "p_linear_0.1", pm=False))
        macro(f"{k}{w}ProbeTrain", ms(pat.format(e), "p_linear_0.1_train", pm=False))
        macro(f"{k}{w}CdnvLab", ms(pat.format(e), "p_cdnv_labeled_0.1", 2, pm=False))
        macro(f"{k}{w}CdnvTestLab", ms(pat.format(e), "p_cdnv_test_0.1", 2, pm=False))

# significance table
lines = [r"\begin{tabular}{llcc}", r"\toprule", r"comparison & protocol & difference [95\% CI] & $p$ \\", r"\midrule"]
NICE = {"simclr": "SimCLR", "supcon": "SupCon (label-only)", "union": "Union", "sub": "SimCLR (subset)",
        "supcon@e10": "SupCon, 970 steps", "simclr@e100": "SimCLR, 9,700 steps", "simclr@e10": "SimCLR, 970 steps",
        "simclr@e200": "SimCLR, 19,400 steps", "supcon@e200": "SupCon, 19,400 steps", "union@e200": "Union, 19,400 steps",
        "simclr_all_e200@ft": "SimCLR, 19,400 steps", "supcon_lab_f0.1_e200@ft": "SupCon, 19,400 steps",
        "union_f0.1_e200@ft": "Union, 19,400 steps"}
PROT = {"Lin": "linear (all labels)", "Knn": "$k$NN (all labels)", "Ft": "fine-tune (5k labels)"}
for label, a, b, r in STATS:
    p = r["p_boot"]
    ps = r"$<10^{-4}$" if p <= 1e-4 else f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"
    lines.append(f"{NICE[a]} $-$ {NICE[b]} & {PROT[label]} & {r['diff']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] & {ps} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("table_stats.tex", "\n".join(lines) + "\n")

# ----------------------------------------------------------------------------
# Figure 1: the epoch trap (accuracy vs. gradient steps)
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
series = [("SimCLR, all images", "simclr", "simclr_all_e{}"), ("SupCon, 10% labeled only", "supcon", "supcon_lab_f0.1_e{}"),
          ("Union", "union", "union_f0.1_e{}")]
for label, k, pat in series:
    xs, ys, es = [], [], []
    for e in (10, 25, 50, 100, 200):
        v = vals(pat.format(e), "linear_acc")
        if len(v):
            steps = DF.loc[DF["config"] == pat.format(e), "steps"].iloc[0]
            xs.append(steps)
            ys.append(v.mean())
            es.append(v.std(ddof=1) if len(v) > 1 else 0)
    if xs:
        ax.errorbar(xs, ys, yerr=es, color=COL[k], marker="o", label=label, capsize=2, ms=4)
# the original report's comparison, in step units of this study
r_sup, r_sim = vals("supcon_lab_f0.1_e10", "linear_acc"), vals("simclr_all_e100", "linear_acc")
if len(r_sup) and len(r_sim):
    x0 = DF.loc[DF["config"] == "supcon_lab_f0.1_e10", "steps"].iloc[0]
    x1 = DF.loc[DF["config"] == "simclr_all_e100", "steps"].iloc[0]
    ax.annotate("", xy=(x0, r_sup.mean()), xytext=(x1, r_sim.mean()),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.8, ls="--", shrinkA=4, shrinkB=4))
    ax.text(np.sqrt(x0 * x1) * 0.95, (r_sup.mean() + r_sim.mean()) / 2 - 1.3,
            "\"100 epochs\" each:\nthe comparison\nin our report", fontsize=6.5, color=MUTED, ha="center", va="top")
ax.set_xscale("log")
ax.set_xlabel("gradient steps (batch 512)")
ax.set_ylabel("linear-probe accuracy (%)")
ax.set_title("(a) Labels buy speed, images buy accuracy", loc="left")
ax.legend(loc="lower right")

ax = axes[1]
groups = [("SimCLR", "simclr", "simclr_all_e{}"), ("SupCon\n(label-only)", "supcon", "supcon_lab_f0.1_e{}"),
          ("Union", "union", "union_f0.1_e{}")]
w = 0.36
for i, (label, k, pat) in enumerate(groups):
    for j, (key, prefix, hatch, lab) in enumerate((("linear_acc", "", None, "probe trained on all 50k labels"),
                                                     ("finetune_acc", "ft_", "////", "fine-tuned on the 5k labels"))):
        v = vals(prefix + pat.format(200), key)
        if not len(v):
            continue
        x = i + (j - 0.5) * (w + 0.03)
        ax.bar(x, v.mean(), w, color=COL[k] if hatch is None else "white", edgecolor=COL[k], hatch=hatch, linewidth=1.0,
               label=lab if i == 0 else None)
        if len(v) > 1:
            ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), color=INK, capsize=2, lw=0.8)
        ax.text(x, v.mean() + 0.35, f"{v.mean():.1f}", ha="center", va="bottom", fontsize=6.5, color=INK)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([g[0] for g in groups], fontsize=7.5)
ax.set_ylim(78, 93.5)
ax.set_ylabel("test accuracy (%)")
ax.set_title("(b) Labels spent in pretraining are gone", loc="left")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=MUTED, label="probe trained on all 50k labels"),
                   Patch(facecolor="white", edgecolor=MUTED, hatch="////", label="fine-tuned on the 5k labels only")],
          loc="upper right", fontsize=6.5, bbox_to_anchor=(1.0, 1.02))
ax.grid(axis="x", visible=False)
fig.savefig(os.path.join(FIG, "fig_main.pdf"))
plt.close(fig)

# fraction figure: (a) all-label probe, (b) fine-tuning with the labeled subset only
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.3))
fr = [0.01, 0.05, 0.1, 0.2]
FTF = {"simclr": lambda f: "ft_simclr_all_e100" if f == 0.1 else f"ft_simclr_all_e100_f{f}",
       "supcon": lambda f: f"ft_supcon_lab_f{f}_e100", "union": lambda f: f"ft_union_f{f}_e100"}
for ax, mode in ((axes[0], "probe"), (axes[1], "ft")):
    for label, k, pat in (("SimCLR, all images", "simclr", None), ("SimCLR, labeled images only", "sub", "simclr_sub_f{}_e100"),
                          ("SupCon, labeled images only", "supcon", "supcon_lab_f{}_e100"), ("Union", "union", "union_f{}_e100")):
        if mode == "ft" and k == "sub":
            continue
        xs, ys = [], []
        for f in fr:
            if mode == "probe":
                v = vals(CORE["simclr"] if pat is None else pat.format(f), "linear_acc")
            else:
                v = vals(FTF[k](f), "finetune_acc")
            if len(v):
                xs.append(100 * f)
                ys.append(v.mean())
        if xs:
            ax.plot(xs, ys, color=COL[k], marker="o", ls="--" if k == "sub" else "-", label=label, ms=4)
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 20])
    ax.set_xticklabels(["1%", "5%", "10%", "20%"])
    ax.set_xlabel("labeled fraction")
axes[0].set_ylabel("test accuracy (%)")
axes[0].set_title("(a) Probe trained on all labels", loc="left")
axes[1].set_title("(b) Fine-tuned on the labeled images only", loc="left")
axes[0].legend(loc="lower right", fontsize=6.3)
fig.savefig(os.path.join(FIG, "fig_fraction.pdf"))
plt.close(fig)

# fraction-sweep table (both protocols)
fr = [0.01, 0.05, 0.1, 0.2]
lines = [r"\begin{tabular}{l cccc cccc}", r"\toprule",
         r" & \multicolumn{4}{c}{linear probe, all labels} & \multicolumn{4}{c}{fine-tuned, labeled subset only} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"labeled fraction & 1\% & 5\% & 10\% & 20\% & 1\% & 5\% & 10\% & 20\% \\", r"\midrule"]
FT = {"simclr": lambda f: "ft_simclr_all_e100" if f == 0.1 else f"ft_simclr_all_e100_f{f}",
      "sub": lambda f: None,
      "supcon": lambda f: f"ft_supcon_lab_f{f}_e100",
      "union": lambda f: f"ft_union_f{f}_e100"}
for label, k, pat in (("SimCLR (all images)", "simclr", None), ("SimCLR (labeled images only)", "sub", "simclr_sub_f{}_e100"),
                      ("SupCon (label-only)", "supcon", "supcon_lab_f{}_e100"), ("Union", "union", "union_f{}_e100")):
    cells = [ms(CORE["simclr"] if pat is None else pat.format(f), "linear_acc", pm=False) for f in fr]
    cells += [ms(FT[k](f), "finetune_acc", pm=False) if FT[k](f) else "--" for f in fr]
    lines.append(label + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("table_fraction.tex", "\n".join(lines) + "\n")
for f, w in ((0.01, "One"), (0.05, "Five"), (0.2, "Twenty")):
    macro(f"fracSimclrFt{w}", ms(FT["simclr"](f), "finetune_acc", pm=False))
    macro(f"fracSupconFt{w}", ms(FT["supcon"](f), "finetune_acc", pm=False))
    macro(f"fracUnionFt{w}", ms(FT["union"](f), "finetune_acc", pm=False))
    macro(f"fracSupconLin{w}", ms(f"supcon_lab_f{f}_e100", "linear_acc", pm=False))
    macro(f"fracUnionLin{w}", ms(f"union_f{f}_e100", "linear_acc", pm=False))
    macro(f"fracSubLin{w}", ms(f"simclr_sub_f{f}_e100", "linear_acc", pm=False))

# ----------------------------------------------------------------------------
# Figure: coverage vs spread (diversity of the labeled pool)
# ----------------------------------------------------------------------------
div_path = os.path.join(ROOT, "results", "diversity.json")
if os.path.exists(div_path):
    div = json.load(open(div_path))["simclr"]["pools"]
    pts = []
    for name, v in div.items():
        f, kind, s = v["fraction"], v["kind"], v["seed"]
        if s != 0 or f >= 1:
            continue
        c = f"supcon_lab_f{f}_e100" if kind == "random" else f"supcon_lab_{kind}_f{f}_e100"
        acc = vals(c, "linear_acc")
        acc_sub = vals(c.replace("supcon_lab", "simclr_sub"), "linear_acc")
        if len(acc):
            pts.append((kind, f, v, acc.mean(), acc_sub.mean() if len(acc_sub) else None))
    if pts:
        fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.3), sharey=True)
        mk = {"random": "o", "cluster10": "s", "cluster50": "D", "kcenter": "^"}
        nm = {"random": "random pool", "cluster10": "redundant pool (k=10)", "cluster50": "redundant pool (k=50)",
              "kcenter": "k-center pool"}
        for ax, key, xl in ((axes[0], "radius", "coverage radius (lower = better covered)"),
                            (axes[1], "spread_aug", "within-class spread of augmented views")):
            seen = set()
            for kind, f, v, acc, acc_sub in pts:
                ax.scatter(v[key], acc, marker=mk[kind], s=30, color=COL["supcon"],
                           edgecolor="white", linewidth=0.6, zorder=3, label=None if kind in seen else nm[kind])
                if acc_sub is not None:
                    ax.scatter(v[key], acc_sub, marker=mk[kind], s=30, facecolor="white", edgecolor=COL["simclr"],
                               linewidth=1.0, zorder=3)
                seen.add(kind)
                if kind == "random":
                    ax.annotate(f"{100 * f:g}%", (v[key], acc), textcoords="offset points", xytext=(4, 2), fontsize=6.5, color=MUTED)
            ax.set_xlabel(xl)
        axes[0].invert_xaxis()
        axes[0].set_ylabel("linear-probe acc. (%)")
        axes[0].set_title("(a) Coverage of unseen images", loc="left")
        axes[1].set_title("(b) Spread within the pool", loc="left")
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], marker=mk[k], ls="", color=MUTED, label=nm[k]) for k in mk if k in {p_[0] for p_ in pts}]
        handles += [Line2D([], [], marker="o", ls="", color=COL["supcon"], label="SupCon (filled)"),
                    Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=COL["simclr"], label="SimCLR (hollow)")]
        axes[1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6.3)
        fig.savefig(os.path.join(FIG, "fig_coverage.pdf"))
        plt.close(fig)
    # diversity table
    rows = [r"\begin{tabular}{l r cccc cc c}", r"\toprule",
            r" & & \multicolumn{4}{c}{diversity of the pool} & \multicolumn{2}{c}{SupCon on the pool} & SimCLR on the pool \\",
            r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}\cmidrule(lr){9-9}",
            r"pool & images & spread (pairs) & spread (aug.\ views) & Vendi & coverage radius & probe, 50k & probe, own labels & probe, 50k \\", r"\midrule"]
    order = [f"random_f{f}_s0" for f in (0.01, 0.05, 0.1, 0.2, 1.0)] + ["cluster10_f0.1_s0", "cluster50_f0.1_s0", "kcenter_f0.1_s0"]
    nice = {"random": "random", "cluster10": "redundant ($k{=}10$)", "cluster50": "redundant ($k{=}50$)", "kcenter": "$k$-center"}
    for name in order:
        if name not in div:
            continue
        v = div[name]
        if v["fraction"] >= 1:
            c = CORE["supcon"].replace("f0.1", "f1.0")
        elif v["kind"] == "random":
            c = f"supcon_lab_f{v['fraction']}_e100"
        else:
            c = f"supcon_lab_{v['kind']}_f{v['fraction']}_e100"
        lab = nice[v["kind"]] + (f" {100 * v['fraction']:g}\\%" if v["kind"] == "random" else "")
        rows.append(f"{lab} & {v['n']:,} & {v['spread']:.3f} & {v['spread_aug']:.3f} & {v['vendi']:.1f} & "
                    f"{v['radius']:.4f} & {ms(c, 'linear_acc', pm=False)} & {ms(c, 'linear_lab_acc', pm=False)} & "
                    f"{ms(c.replace('supcon_lab', 'simclr_sub'), 'linear_acc', pm=False) if v['fraction'] < 1 else ms(CORE['simclr'], 'linear_acc', pm=False)} \\\\")
        if v["kind"] == "random" and v["fraction"] == 0.1:
            macro("radiusTen", f"{v['radius']:.3f}")
            macro("spreadTen", f"{v['spread_aug']:.3f}")
        if v["kind"] == "random" and v["fraction"] == 0.01:
            macro("radiusOne", f"{v['radius']:.3f}")
            macro("spreadOne", f"{v['spread_aug']:.3f}")
        if v["kind"] == "random" and v["fraction"] == 1.0:
            macro("radiusFull", f"{v['radius']:.3f}")
            macro("spreadFull", f"{v['spread_aug']:.3f}")
        if v["kind"] == "kcenter":
            macro("radiusKc", f"{v['radius']:.3f}")
            macro("spreadKc", f"{v['spread_aug']:.3f}")
            macro("neffKc", f"{v['n_eff']:,.0f}")
        if v["kind"] == "cluster50":
            macro("neffClFifty", f"{v['n_eff']:,.0f}")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("table_diversity.tex", "\n".join(rows) + "\n")
for k in ("cluster10", "cluster50", "kcenter"):
    nm_ = {"cluster10": "clTen", "cluster50": "clFifty", "kcenter": "kc"}[k]
    macro(nm_ + "Acc", ms(f"supcon_lab_{k}_f0.1_e100", "linear_acc", pm=False))
    macro(nm_ + "LabAcc", ms(f"supcon_lab_{k}_f0.1_e100", "linear_lab_acc", pm=False))
macro("supconLabAcc", ms("supcon_lab_f0.1_e100", "linear_lab_acc", pm=False))
macro("subKcAcc", ms("simclr_sub_kcenter_f0.1_e100", "linear_acc", pm=False))
macro("subClFiftyAcc", ms("simclr_sub_cluster50_f0.1_e100", "linear_acc", pm=False))

# ----------------------------------------------------------------------------
# Figure: geometry -- collapse on seen vs unseen images
# ----------------------------------------------------------------------------
geo = [("SimCLR", "simclr"), ("SimCLR (subset)", "sub"), ("SupCon (label-only)", "supcon"), ("Union", "union")]
have = [g for g in geo if len(vals(CORE[g[1]], "seen_cdnv"))]
if have:
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    w = 0.38
    for i, (label, k) in enumerate(have):
        c = CORE[k]
        seen = vals(c, "seen_cdnv").mean()
        # images that were not in the pretraining pool: held-out training images for label-only pools,
        # the test set otherwise (identical numbers within 0.02 for full-pool methods)
        held = vals(c, "heldout_cdnv")
        held = held.mean() if len(held) else vals(c, "test_cdnv").mean()
        ax.bar(i - w / 2 - 0.01, seen, w, color=COL[k], edgecolor="white", linewidth=0.8)
        ax.bar(i + w / 2 + 0.01, held, w, color="white", edgecolor=COL[k], hatch="////", linewidth=1.0)
        ax.text(i - w / 2, seen, f"{seen:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.text(i + w / 2, held, f"{held:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(have)))
    ax.set_xticklabels([h[0].replace(" (", "\n(") for h in have], fontsize=6.8)
    ax.set_ylabel("class-distance normalised\nvariance (CDNV, lower = tighter)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=MUTED, label="images seen in pretraining"),
                       Patch(facecolor="white", edgecolor=MUTED, hatch="////", label="unseen images")], loc="upper right", fontsize=6.5)
    ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(FIG, "fig_geometry.pdf"))
    plt.close(fig)

# geometry table
lines = [r"\begin{tabular}{l ccc ccc}", r"\toprule",
         r" & \multicolumn{3}{c}{CDNV $\downarrow$} & \multicolumn{3}{c}{test set} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"method & seen & unseen & ratio & NCC acc. & ETF dev.\ $\downarrow$ & equinorm CV $\downarrow$ \\", r"\midrule"]
for label, k in geo:
    c = CORE[k]
    s, h = mean(c, "seen_cdnv"), mean(c, "heldout_cdnv")
    if not np.isfinite(h):
        h = mean(c, "test_cdnv")
    lines.append(f"{label} & {fmt(s, 2)} & {fmt(h, 2)} & {fmt(h / s if np.isfinite(s) else float('nan'), 1)}$\\times$ & "
                 f"{fmt(mean(c, 'test_ncc_acc'))} & {fmt(mean(c, 'test_etf_dev'), 3)} & {fmt(mean(c, 'test_etf_norm_cv'), 3)} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("table_geometry.tex", "\n".join(lines) + "\n")
for k in ("simclr", "supcon", "union", "sub"):
    c = CORE[k]
    s, h = mean(c, "seen_cdnv"), mean(c, "heldout_cdnv")
    if not np.isfinite(h):
        h = mean(c, "test_cdnv")
    macro(f"{k}CdnvRatio", fmt(h / s, 1) if np.isfinite(s) and s > 0 else "--")

# ----------------------------------------------------------------------------
# Table: baselines
# ----------------------------------------------------------------------------
lines = [r"\begin{tabular}{l l c ccc}", r"\toprule",
         r"method & family & labels used & linear (all labels) & $k$NN & accuracy with 5k labels$^\ddagger$ \\", r"\midrule"]
BASE = [("Supervised CE", "supervised", "5k", "ce_f0.1_e100", "classifier_acc"),
        ("SimCLR", "contrastive SSL", "--", CORE["simclr"], "ft"),
        ("BYOL", "non-contrastive SSL", "--", "byol_all_e100", "ft"),
        ("SimSiam", "non-contrastive SSL", "--", "simsiam_all_e100", "ft"),
        ("SupCon (label-only)", "supervised contrastive", "5k", CORE["supcon"], "ft"),
        ("MixMatch$^\\S$", "semi-supervised", "5k", "mixmatch_f0.1", "classifier_acc"),
        ("FixMatch$^\\S$", "semi-supervised", "5k", "fixmatch_f0.1", "classifier_acc"),
        ("Union", "semi-supervised contrastive", "5k", CORE["union"], "ft"),
        ("Supervised CE, all labels (reference)", "supervised", "50k", "ce_f1.0_e100", "classifier_acc")]
for label, fam, labs, c, key in BASE:
    last = ms("ft_" + c, "finetune_acc") if key == "ft" else ms(c, key)
    lines.append(f"{label} & {fam} & {labs} & {ms(c, 'linear_acc')} & {ms(c, 'knn_acc')} & {last} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("table_baselines.tex", "\n".join(lines) + "\n")

# ----------------------------------------------------------------------------
# CIFAR-100 and ResNet-50
# ----------------------------------------------------------------------------
lines = [r"\begin{tabular}{l ccc ccc}", r"\toprule",
         r" & \multicolumn{3}{c}{CIFAR-100, ResNet-18} & \multicolumn{3}{c}{CIFAR-10, ResNet-50} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}", r"method & probe, 50k & $k$NN & fine-tune, 5k & probe, 50k & $k$NN & fine-tune, 5k \\", r"\midrule"]
for label, k in (("SimCLR", "simclr_all_e100"), ("SimCLR (subset)", "simclr_sub_f0.1_e100"),
                 ("SupCon (label-only)", "supcon_lab_f0.1_e100"), ("Union", "union_f0.1_e100")):
    lines.append(f"{label} & {ms('c100_' + k, 'linear_acc')} & {ms('c100_' + k, 'knn_acc')} & {ms('ft_c100_' + k, 'finetune_acc')} & "
                 f"{ms('r50_' + k, 'linear_acc')} & {ms('r50_' + k, 'knn_acc')} & {ms('ft_r50_' + k, 'finetune_acc')} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("table_generalization.tex", "\n".join(lines) + "\n")
for k, short in (("simclr_all_e100", "Simclr"), ("supcon_lab_f0.1_e100", "Supcon"), ("union_f0.1_e100", "Union"),
                 ("simclr_sub_f0.1_e100", "Sub")):
    macro(f"cHundred{short}", ms("c100_" + k, "linear_acc", pm=False))
    macro(f"rFifty{short}", ms("r50_" + k, "linear_acc", pm=False))
    macro(f"cHundred{short}Ft", ms("ft_c100_" + k, "finetune_acc", pm=False))
    macro(f"rFifty{short}Ft", ms("ft_r50_" + k, "finetune_acc", pm=False))

# ----------------------------------------------------------------------------
# colour shortcut + augmentation tiers
# ----------------------------------------------------------------------------
AUG = [("crop + flip", "Tier 1", "simclr_exp2_e100"), ("crop + flip + jitter", "Tier 2", "simclr_exp5_e100"),
       ("crop + flip + grayscale", "Tier 2", "simclr_exp6_e100"), ("crop + flip + jitter + grayscale", "Tier 3", CORE["simclr"])]
lines = [r"\begin{tabular}{l l ccc ccc}", r"\toprule",
         r" & & \multicolumn{3}{c}{accuracy (\%)} & \multicolumn{3}{c}{colour reliance} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"augmentation & tier & linear & gray test & $\Delta$ & CKA(hist.) & CKA(label) & hist.\ $R^2$ \\", r"\midrule"]
for label, tier, c in AUG:
    lin, gray = mean(c, "linear_acc"), mean(c, "linear_acc_gray")
    lines.append(f"{label} & {tier} & {fmt(lin)} & {fmt(gray)} & {fmt(gray - lin, 1, True)} & "
                 f"{fmt(mean(c, 'cka_hist'), 2)} & {fmt(mean(c, 'cka_label'), 2)} & {fmt(mean(c, 'hist_r2'), 2)} \\\\")
hp = os.path.join(ROOT, "results", "hist_probe.json")
if os.path.exists(hp):
    h = json.load(open(hp))["cifar10_bins8"]["acc"]
    lines.append(r"\midrule")
    lines.append(f"colour histogram only ($8^3$ bins, no encoder) & -- & {h:.1f} & -- & -- & 1.00 & -- & 1.00 \\\\")
    macro("histProbe", f"{h:.1f}")
lines += [r"\bottomrule", r"\end{tabular}"]
write("table_color.tex", "\n".join(lines) + "\n")
lines = [r"\begin{tabular}{l l ccccc}", r"\toprule",
         r"augmentation & tier & linear & CDNV$\downarrow$ & ETF dev.$\downarrow$ & $\cos$ mean (target $-0.111$) & equinorm CV$\downarrow$ \\", r"\midrule"]
for label, tier, c in AUG:
    lines.append(f"{label} & {tier} & {fmt(mean(c, 'linear_acc'))} & {fmt(mean(c, 'test_cdnv'), 2)} & {fmt(mean(c, 'test_etf_dev'), 3)} & "
                 f"{fmt(mean(c, 'test_etf_cos_mean'), 3)} & {fmt(mean(c, 'test_etf_norm_cv'), 3)} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("table_etf.tex", "\n".join(lines) + "\n")
for short, c in (("Two", "simclr_exp2_e100"), ("Five", "simclr_exp5_e100"), ("Six", "simclr_exp6_e100"), ("Eight", CORE["simclr"])):
    macro(f"aug{short}Lin", fmt(mean(c, "linear_acc")))
    macro(f"aug{short}Gray", fmt(mean(c, "linear_acc_gray") - mean(c, "linear_acc"), 1, True))
    macro(f"aug{short}Cka", fmt(mean(c, "cka_hist"), 2))
    macro(f"aug{short}CkaLabel", fmt(mean(c, "cka_label"), 2))
    macro(f"aug{short}GrayDrop", fmt(mean(c, "linear_acc") - mean(c, "linear_acc_gray"), 1))
    macro(f"aug{short}GrayAcc", fmt(mean(c, "linear_acc_gray"), 1))
    macro(f"aug{short}Etf", fmt(mean(c, "test_etf_dev"), 3))
    macro(f"aug{short}Cdnv", fmt(mean(c, "test_cdnv"), 2))
macro("supconNoJitterLin", ms("supcon_lab_f0.1_exp6_e100", "linear_acc", pm=False))
macro("augJitterCost", fmt(mean(CORE["simclr"], "linear_acc") - mean("simclr_exp6_e100", "linear_acc"), 1))
macro("supconJitterCost", fmt(mean(CORE["supcon"], "linear_acc") - mean("supcon_lab_f0.1_exp6_e100", "linear_acc"), 1))

# ----------------------------------------------------------------------------
# OOD resolution study
# ----------------------------------------------------------------------------
ood_path = os.path.join(ROOT, "results", "ood_resolution.json")
if os.path.exists(ood_path):
    ood = json.load(open(ood_path))
    enc = ood["encoders"]
    lines = [r"\begin{tabular}{l cc cc}", r"\toprule",
             r" & \multicolumn{2}{c}{dog recall (\%)} & \multicolumn{2}{c}{retrieval P@5 (\%)} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"encoder & CIFAR-10 dogs & ImageNet dogs$\to$32px & CIFAR-10 dogs & ImageNet dogs$\to$32px \\", r"\midrule"]
    names = {"simclr_all_e100_s0": "SimCLR", "simclr_all_e200_s0": "SimCLR, 19,400 steps",
             "supcon_lab_f0.1_e100_s0": "SupCon (label-only)",
             "union_f0.1_e100_s0": "Union", "ce_f1.0_e100_s0": "Supervised CE (all labels)",
             "simclr_sub_f0.1_e100_s0": "SimCLR (subset)"}
    for run, r in enc.items():
        lo, hi = r["dog_recall_ood_ci"]
        lines.append(f"{names.get(run, run)} & {r['dog_recall_cifar']:.1f} & {r['dog_recall_ood']:.1f} {{\\scriptsize[{lo:.1f}, {hi:.1f}]}} & "
                     f"{r['p_at_5_cifar']:.1f} & {r['p_at_5_ood']:.1f} \\\\")
        key = {"simclr_all_e100_s0": "Simclr", "union_f0.1_e100_s0": "Union", "supcon_lab_f0.1_e100_s0": "Supcon",
               "ce_f1.0_e100_s0": "Ce"}.get(run)
        if key:
            macro(f"ood{key}Cifar", f"{r['dog_recall_cifar']:.1f}")
            macro(f"ood{key}Ood", f"{r['dog_recall_ood']:.1f}")
            macro(f"ood{key}PCifar", f"{r['p_at_5_cifar']:.1f}")
            macro(f"ood{key}POod", f"{r['p_at_5_ood']:.1f}")
            top = max(r["ood_prediction_hist"].items(), key=lambda kv: kv[1] if kv[0] != "dog" else -1)
            macro(f"ood{key}TopWrong", f"{top[0]} ({top[1]:.0f}\\%)")
    if "clip" in ood:
        c = ood["clip"]
        res = c["dog_recall_ood_by_resolution"]
        lines.append(r"\midrule")
        lines.append(f"CLIP ViT-B/32 zero-shot (reference) & {c['dog_recall_cifar']:.1f} & {res['32']:.1f} & -- & -- \\\\")
        macro("clipCifar", f"{c['dog_recall_cifar']:.1f}")
        for rr, v in res.items():
            macro("clipRes" + {"16": "Sixteen", "32": "ThirtyTwo", "64": "SixtyFour", "128": "OneTwoEight", "224": "TwoTwoFour"}[rr], f"{v:.1f}")
        macro("clipTest", f"{c['test_acc']:.1f}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("table_ood.tex", "\n".join(lines) + "\n")
    macro("nOod", f"{ood['n_ood']:,}")

# ----------------------------------------------------------------------------
# validation-split design ablation for the union objective (pilot runs)
# ----------------------------------------------------------------------------
PIL = [("SimCLR, $\\tau{=}0.5$", "pilot_simclr_t0.5"), ("SimCLR, $\\tau{=}0.2$", "pilot_simclr_t0.2"),
       ("SimCLR, $\\tau{=}0.1$", "pilot_simclr_t0.1"),
       ("SupCon label-only, $\\tau{=}0.1$", "pilot_supconL_t0.1"), ("SupCon label-only, $\\tau{=}0.5$", "pilot_supconL_t0.5"),
       ("Union, $\\tau{=}0.1$ for all anchors", "pilot_union_t0.1"), ("Union, $\\tau{=}0.5$ for all anchors", "pilot_union_t0.5"),
       ("Union, $\\tau_\\ell{=}0.1$, $\\tau_u{=}0.5$, uniform batches", "pilot_union_mixed"),
       ("Union, $\\tau_\\ell{=}0.1$, $\\tau_u{=}0.5$, 50\\% labeled per batch", "pilot_union_mixed_sh0.5"),
       ("Union, $\\tau_\\ell{=}0.1$, $\\tau_u{=}0.5$, 25\\% labeled per batch (chosen)", "pilot_union_mixed_sh0.25")]
lines = [r"\begin{tabular}{l ccc}", r"\toprule", r"configuration (validation split, 50 epochs) & linear & $k$NN & linear, 10\% labels \\", r"\midrule"]
for label, n in PIL:
    p = os.path.join(RUNS, n, "result.json")
    if os.path.exists(p):
        m = json.load(open(p))["metrics"]
        lines.append(f"{label} & {m['linear_acc']:.2f} & {m['knn_acc']:.2f} & {m.get('linear_lab_acc', float('nan')):.2f} \\\\".replace("nan", "--"))
lines += [r"\bottomrule", r"\end{tabular}"]
write("table_pilot.tex", "\n".join(lines) + "\n")

# ----------------------------------------------------------------------------
# Appendix figure: colour-jitter pairs of the original report (ResNet-50, 200 epochs, single runs,
# NVIDIA RTX 5000 Ada). Numbers are copied from the report (Table V); no run of this study.
# ----------------------------------------------------------------------------
REPORT = [("pure rotation\n(36 / 38)", 34.40, 51.21), ("weak crop\n(35 / 39)", 59.22, 80.53),
          ("crop + blur\n(9 / 40)", 63.01, 80.65), ("crop + flip + blur\n(13 / 41)", 64.49, 84.30),
          ("crop + cutout\n(10 / 42)", 66.27, 81.21)]
fig, ax = plt.subplots(figsize=(6.0, 2.2))
x = np.arange(len(REPORT))
w = 0.38
ax.bar(x - w / 2 - 0.01, [r[1] for r in REPORT], w, color="#86b6ef", edgecolor="white", label="without colour jitter")
ax.bar(x + w / 2 + 0.01, [r[2] for r in REPORT], w, color=COL["simclr"], edgecolor="white", label="with colour jitter")
for i, (_, a, b) in enumerate(REPORT):
    ax.text(i - w / 2, a + 1, f"{a:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.text(i + w / 2, b + 1, f"{b:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK)
ax.set_xticks(x)
ax.set_xticklabels([r[0] for r in REPORT], fontsize=6.8)
ax.set_ylabel("linear-probe acc. (%)")
ax.set_ylim(0, 95)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", ncol=2)
fig.savefig(os.path.join(FIG, "fig_report_jitter.pdf"))
plt.close(fig)

# ----------------------------------------------------------------------------
# write macros
# ----------------------------------------------------------------------------
for k in DEFAULT_MACROS:
    MACROS.setdefault(k, "--")
for name in ("table_ood.tex", "table_diversity.tex"):
    if not os.path.exists(os.path.join(GEN, name)):
        write(name, "\\begin{tabular}{l}\\toprule (not yet computed) \\\\ \\bottomrule\\end{tabular}\n")
with open(os.path.join(GEN, "numbers.tex"), "w") as f:
    f.write("% generated by scripts/make_assets.py -- do not edit\n")
    for k, v in sorted(MACROS.items()):
        assert re.fullmatch(r"[A-Za-z]+", k), k
        f.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
print(f"{len(DF)} runs, {len(MACROS)} macros, tables in {GEN}, figures in {FIG}")
