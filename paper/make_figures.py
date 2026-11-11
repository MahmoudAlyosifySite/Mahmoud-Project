"""Make every figure, table and quoted number of the paper from the results files.

    python paper/make_figures.py

Reads only `results/**/summary.json` and `results/ledger.jsonl` (written by the
experiment scripts). Writes `paper/figures/*.pdf` and `paper/generated/*.tex`. The
paper's text quotes numbers only through the macros in `generated/numbers.tex`,
so no number is typed by hand. Appendix A (`generated/appendix_ledger.tex`) lists
the ledger entries behind each table.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"
GEN = ROOT / "paper" / "generated"
KEY = "impossible_vs_improbable"
COMPS = ("impossible_vs_possible", "impossible_vs_improbable", "improbable_vs_possible")
COMP_SHORT = {"impossible_vs_possible": "imp.\\ vs poss.", "impossible_vs_improbable": "imp.\\ vs improb.",
              "improbable_vs_possible": "improb.\\ vs poss."}
FAMILIES = ("energy", "permanence", "stochastic", "continuity")
SCORE_TEX = {"inv": "\\texttt{inv}", "inv_rel": "\\texttt{inv\\_rel}", "pe_max": "\\texttt{pe\\_max}",
             "pe_mean": "\\texttt{pe\\_mean}", "pe_rel": "\\texttt{pe\\_rel}", "oracle_pe": "\\texttt{oracle\\_pe}"}

macros: dict[str, str] = {}
appendix: list[tuple[str, str, str]] = []  # (what, ledger experiment tag, seeds)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def macro_name(*parts: str) -> str:
    out = ""
    for p in parts:
        p = "".join(ch for ch in str(p) if ch.isalnum())
        for digit, word in zip("0123456789", ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")):
            p = p.replace(digit, word)
        out += p[:1].upper() + p[1:]
    return out


def put(name: str, value, fmt: str = "{:.2f}") -> None:
    macros[name] = fmt.format(value) if not isinstance(value, str) else value


def ledger_index() -> dict[str, dict]:
    """experiment tag -> {seeds, commits, first and last time}."""
    idx: dict[str, dict] = defaultdict(lambda: {"seeds": set(), "commits": set(), "times": []})
    path = RES / "ledger.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        d = idx[e["experiment"]]
        d["seeds"].add(e["seed"])
        d["commits"].add(e["commit"])
        d["times"].append(e["time"])
    return idx


# ---------------------------------------------------------------- tables of AUROCs


def auroc_table(summary: dict, families, scores, caption: str, label: str, seeds) -> str:
    lines = [
        "\\begin{table}[t]\\centering\\small",
        f"\\caption{{{caption}}}\\label{{{label}}}",
        "\\begin{tabular}{ll" + "c" * len(COMPS) + "}\\toprule",
        "Family & Score & " + " & ".join(COMP_SHORT[c] for c in COMPS) + "\\\\\\midrule",
    ]
    for fam in families:
        if fam not in summary:
            continue
        for i, s in enumerate(scores):
            if s not in summary[fam]:
                continue
            cells = []
            for c in COMPS:
                v = summary[fam][s][c]
                cells.append(f"{v['mean']:.2f} $\\pm$ {v['std']:.2f}")
            lines.append(f"{fam if i == 0 else ''} & {SCORE_TEX[s]} & " + " & ".join(cells) + "\\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}",
              f"\\par\\smallskip\\footnotesize Mean $\\pm$ standard deviation over seeds {', '.join(map(str, seeds))}. "
              "0.5 is chance; below 0.5 the score ranks the lawful clips above the impossible ones.",
              "\\end{table}"]
    return "\n".join(lines)


def verdict_table(entries: list[tuple[str, dict]], caption: str, label: str) -> str:
    lines = [
        "\\begin{table}[t]\\centering\\small",
        f"\\caption{{{caption}}}\\label{{{label}}}",
        "\\begin{tabular}{llcccccc}\\toprule",
        "Run & Family & best PE score & best PE & \\texttt{inv} & gain & H1 & H2\\\\\\midrule",
    ]
    for name, verdict in entries:
        first = True
        for fam, v in verdict["families"].items():
            lines.append(
                f"{name if first else ''} & {fam} & {SCORE_TEX[v['best_pe']]} & {v['best_pe_auroc']:.2f} & "
                f"{v['inv_auroc']:.2f} & {v['gain']:+.2f} & {'yes' if v['h1'] else 'no'} & {'yes' if v['h2'] else 'no'}\\\\"
            )
            first = False
        lines.append(f"& \\multicolumn{{7}}{{l}}{{\\emph{{{'GO' if verdict['go'] else 'NO-GO'}}}: H2 in "
                     f"{verdict['h2_families']} of {len(verdict['families'])} conservation families}}\\\\\\midrule")
    lines[-1] = lines[-1].replace("\\midrule", "\\bottomrule")
    lines += ["\\end{tabular}",
              "\\par\\smallskip\\footnotesize AUROC impossible vs improbable, mean over seeds. "
              "The best PE score is chosen per family after the fact, in the baseline's favour. "
              "H1: best PE $\\le 0.70$. H2: \\texttt{inv} beats it by at least 0.15 on average and in every seed.",
              "\\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------- stages


def stage_a(idx) -> list[str]:
    base = RES / "pilot_synthetic"
    tex = []
    if not (base / "frames_span1" / "summary.json").exists():
        return tex
    main = load(base / "frames_span1" / "summary.json")
    seeds = sorted(int(s) for s in main["runs"])
    tex.append(auroc_table(main["summary"], FAMILIES, ["inv", "inv_rel", "pe_max", "pe_mean", "pe_rel", "oracle_pe"],
                           "Stage A: ideal frame latents. AUROC for each pair of conditions.", "tab:stageA", seeds))
    appendix.append(("Table~\\ref{tab:stageA}, Figure~\\ref{fig:stageA}", "pilot\\_synthetic/frames\\_span1", ", ".join(map(str, seeds))))
    for fam in FAMILIES:
        for s in ("inv", "pe_max", "pe_mean", "pe_rel", "oracle_pe"):
            put(macro_name("A", fam, s), main["summary"][fam][s][KEY]["mean"])
            put(macro_name("A", fam, s, "poss"), main["summary"][fam][s]["impossible_vs_possible"]["mean"])
    v = main["verdict"]
    for fam, d in v["families"].items():
        put(macro_name("A", fam, "bestpe"), d["best_pe_auroc"])
    put("AGo", "GO" if v["go"] else "NO-GO")
    put("ANfam", str(v["h2_families"]))
    r2e = [w["r2"]["energy"] for r in main["runs"].values() for w in r["worlds"].values()]
    r2c = [w["r2"]["count"] for r in main["runs"].values() for w in r["worlds"].values()]
    put("ARtwoEnergyMin", min(r2e), "{:.4f}")
    put("ARtwoCountMin", min(r2c), "{:.4f}")

    # sensitivity
    sens = []
    for tag, name in (("frames_span1", "default"), ("frames_span1_scale2", "scale 2"), ("frames_span1_scale4", "scale 4"),
                      ("frames_span1_gain1.2", "gain 1.2"), ("frames_span1_gain2", "gain 2.0")):
        p = base / tag / "summary.json"
        if p.exists():
            s = load(p)
            sens.append((name, s["verdict"]))
            if tag != "frames_span1":
                appendix.append(("Table~\\ref{tab:sensitivity}", f"pilot\\_synthetic/{tag.replace('_', chr(92) + '_')}",
                                 ", ".join(sorted(map(str, s["runs"])))))
                for fam, d in s["verdict"]["families"].items():
                    put(macro_name("Sens", tag.replace("frames_span1_", ""), fam, "bestpe"), d["best_pe_auroc"])
                    put(macro_name("Sens", tag.replace("frames_span1_", ""), fam, "inv"), d["inv_auroc"])
    tex.append(verdict_table(sens, "Stage A: the go rule, and its sensitivity to the scenario parameters "
                             "(speed of the improbable ball: scale; gain at the rough wall).", "tab:sensitivity"))

    # figure: impossible vs improbable, per family, per score, with seeds
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    scores = ["inv", "pe_max", "pe_mean", "pe_rel", "oracle_pe"]
    colors = ["#1f5fa8", "#c9772c", "#d9a35a", "#8c6a3f", "#7a7a7a"]
    width = 0.16
    for j, (s, col) in enumerate(zip(scores, colors)):
        for i, fam in enumerate(FAMILIES):
            d = main["summary"][fam][s][KEY]
            x = i + (j - 2) * width
            ax.bar(x, d["mean"], width, color=col, label=s if i == 0 else None)
            ax.scatter(np.full(len(d["per_seed"]), x), d["per_seed"], s=4, color="black", zorder=3)
    ax.axhline(0.5, color="black", lw=0.6, ls=":")
    ax.set_xticks(range(len(FAMILIES)), FAMILIES)
    ax.set_ylabel("AUROC impossible\nvs improbable")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "stageA.pdf")
    plt.close(fig)
    return tex


def stage_b(idx) -> list[str]:
    p = RES / "robustness" / "summary.json"
    if not p.exists():
        return []
    b = load(p)
    rows = b["rows"]
    alphas = sorted({r["alpha"] for r in rows})
    noises = sorted({r["noise"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
    for noise, ls in zip(noises, ("-", "--", ":")):
        for span, marker in ((1, "o"), (4, "s")):
            sel = [next(r for r in rows if r["alpha"] == a and r["noise"] == noise and r["span"] == span) for a in alphas]
            lab = f"noise {noise:g}, span {span}"
            axes[0].plot(alphas, [r["families"]["energy"]["r2_energy"] for r in sel], ls=ls, marker=marker, ms=3, label=lab)
            axes[1].plot(alphas, [r["families"]["energy"]["inv"] for r in sel], ls=ls, marker=marker, ms=3)
            axes[2].plot(alphas, [r["families"]["stochastic"]["inv"] for r in sel], ls=ls, marker=marker, ms=3)
    pe_e = [next(r for r in rows if r["alpha"] == a and r["noise"] == 0 and r["span"] == 1)["families"]["energy"]["best_pe"] for a in alphas]
    pe_s = [next(r for r in rows if r["alpha"] == a and r["noise"] == 0 and r["span"] == 1)["families"]["stochastic"]["best_pe"] for a in alphas]
    axes[1].plot(alphas, pe_e, color="gray", lw=2, alpha=0.5, label="best PE (noise 0)")
    axes[2].plot(alphas, pe_s, color="gray", lw=2, alpha=0.5)
    for ax, t in zip(axes, ("R$^2$ of invariants vs true energy", "energy: AUROC imp. vs improb.", "stochastic: AUROC imp. vs improb.")):
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlabel("nonlinearity $\\alpha$")
        ax.set_title(t, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="black", lw=0.5, ls=":")
    handles, labels = [], []
    for ax in axes[:2]:
        h, lab = ax.get_legend_handles_labels()
        handles += h
        labels += lab
    fig.legend(handles, labels, fontsize=6.5, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(FIG / "stageB.pdf")
    plt.close(fig)

    lines = ["\\begin{table}[t]\\centering\\small",
             "\\caption{Stage B: ideal latents bent by $\\alpha$ and given frame-to-frame noise. Mean over seeds "
             + ", ".join(map(str, b["seeds"])) + ".}\\label{tab:stageB}",
             "\\begin{tabular}{rrr|cccc|cccc}\\toprule",
             "& & & \\multicolumn{4}{c|}{energy family} & \\multicolumn{4}{c}{stochastic family}\\\\",
             "$\\alpha$ & noise & span & kept & R$^2_E$ & \\texttt{inv} & best PE & kept & R$^2_E$ & \\texttt{inv} & best PE\\\\\\midrule"]
    for r in rows:
        e, s = r["families"]["energy"], r["families"]["stochastic"]
        lines.append(f"{r['alpha']:g} & {r['noise']:g} & {r['span']} & {e['kept']:.1f} & {e['r2_energy']:.2f} & {e['inv']:.2f} & "
                     f"{e['best_pe']:.2f} & {s['kept']:.1f} & {s['r2_energy']:.2f} & {s['inv']:.2f} & {s['best_pe']:.2f}\\\\")
        appendix.append(("Table~\\ref{tab:stageB}, Figure~\\ref{fig:stageB}",
                         f"robustness/alpha{r['alpha']:g}\\_noise{r['noise']:g}\\_span{r['span']}", ", ".join(map(str, b["seeds"]))))
        put(macro_name("B", f"a{r['alpha']:g}", f"n{r['noise']:g}", f"s{r['span']}", "rtwo"), e["r2_energy"])
        put(macro_name("B", f"a{r['alpha']:g}", f"n{r['noise']:g}", f"s{r['span']}", "inv"), e["inv"])
        put(macro_name("B", f"a{r['alpha']:g}", f"n{r['noise']:g}", f"s{r['span']}", "invS"), s["inv"])
        put(macro_name("B", f"a{r['alpha']:g}", f"n{r['noise']:g}", f"s{r['span']}", "pe"), e["best_pe"])
        put(macro_name("B", f"a{r['alpha']:g}", f"n{r['noise']:g}", f"s{r['span']}", "peS"), s["best_pe"])
    lines += ["\\bottomrule\\end{tabular}",
              "\\par\\smallskip\\footnotesize kept: invariants kept by the selection rule. R$^2_E$: how well the kept "
              "invariants explain the true energy (diagnostic only). AUROCs are impossible vs improbable.",
              "\\end{table}"]
    return ["\n".join(lines)]


def stage_c(idx) -> list[str]:
    base = RES / "pilot_pixels"
    if not base.exists():
        return []
    tex = []
    entries = []
    for p in sorted(base.glob("*/summary.json")):
        s = load(p)
        tag = p.parent.name
        seeds = sorted(int(k) for k in s["runs"])
        if 0 in seeds:
            continue  # seed 0 is exploration only
        entries.append((tag, s))
    # the primary run (pca 16, span 1) first, then the declared secondary options
    entries.sort(key=lambda e: (not e[0].endswith("_pca16_span1"), e[0]))
    for k, (tag, s) in enumerate(entries):
        seeds = sorted(int(x) for x in s["runs"])
        label = f"tab:stageC{k}"
        nice = tag.replace("_", "\\_")
        role = "primary run" if tag.endswith("_pca16_span1") else "declared secondary option"
        if k == 0:
            tex.append(auroc_table(s["summary"], FAMILIES, ["inv", "inv_rel", "pe_max", "pe_mean", "pe_rel", "oracle_pe"],
                                   f"Stage C, {role}: latents of a pixel JEPA trained by us (\\texttt{{{nice}}}). "
                                   "The encoder gate is not met (Table~\\ref{tab:stageCenc}); these AUROCs are reported, "
                                   "not read as a test of H2.", label, seeds))
        appendix.append((f"Tables~\\ref{{{label if k == 0 else 'tab:stageCverdict'}}}, \\ref{{tab:stageCenc}}",
                         f"pilot\\_pixels/{nice}", ", ".join(map(str, seeds))))
        mtag = tag.replace("dim", "d").replace("_lam", "l").replace("_steps", "s").replace("_pca", "p").replace("_span", "sp")
        for fam, d in s["verdict"]["families"].items():
            put(macro_name("C", mtag, fam, "inv"), d["inv_auroc"])
            put(macro_name("C", mtag, fam, "bestpe"), d["best_pe_auroc"])
            put(macro_name("C", mtag, fam, "gain"), d["gain"], "{:+.2f}")
        put(macro_name("C", mtag, "go"), "GO" if s["verdict"]["go"] else "NO-GO")
        put(macro_name("C", mtag, "nfam"), str(s["verdict"]["h2_families"]))
        if "continuity" in s["summary"]:
            put(macro_name("C", mtag, "continuity", "inv"), s["summary"]["continuity"]["inv"][KEY]["mean"])
        probes = [w for r in s["runs"].values() for w in r["worlds"].values()]
        for key in ("probe_r2", "probe_r2_count", "probe_r2_occupancy", "probe_r2_slots", "latent_std_min"):
            vals = [w[key] for w in probes if key in w]
            if vals:
                put(macro_name("C", mtag, key, "min"), min(vals))
                put(macro_name("C", mtag, key, "max"), max(vals))
        slow = [min(w["slowness_val"]) for w in probes if w["slowness_val"]]
        if slow:
            put(macro_name("C", mtag, "slowbest", "min"), min(slow))
            put(macro_name("C", mtag, "slowbest", "max"), max(slow))
        for fam in s["summary"]:
            for sc_ in ("pe_max", "pe_mean", "pe_rel", "oracle_pe", "inv"):
                put(macro_name("C", mtag, fam, sc_, "all"), s["summary"][fam][sc_][KEY]["mean"])
        r2e = [w["r2"]["energy"] for w in probes]
        r2c = [w["r2"]["count"] for w in probes]
        kept = [w["invariants_kept"] for w in probes]
        put(macro_name("C", mtag, "rtwoE", "max"), max(r2e))
        put(macro_name("C", mtag, "rtwoC", "max"), max(r2c))
        put(macro_name("C", mtag, "kept", "min"), str(min(kept)))
        put(macro_name("C", mtag, "kept", "max"), str(max(kept)))
    if entries:
        tex.append(verdict_table([(t.replace("_", "\\_").replace("\\_pca", "\\\\ \\_pca") if False else t.replace("_", "\\_"), s["verdict"])
                                  for t, s in entries],
                                 "Stage C: the go rule on learned latents, for the primary run and each declared secondary option.",
                                 "tab:stageCverdict"))
        # encoder table
        lines = ["\\begin{table}[t]\\centering\\small",
                 "\\caption{Stage C: is the encoder any good? Per seed and world.}\\label{tab:stageCenc}",
                 "\\begin{tabular}{llcccccccc}\\toprule",
                 "Run & seed/world & probe & count & occup. & slots & std min & kept & R$^2_E$ & R$^2_N$\\\\\\midrule"]
        for t, s in entries[:1]:  # the secondary options reuse these encoders
            first = True
            for seed in sorted(s["runs"], key=int):
                for world, w in s["runs"][seed]["worlds"].items():
                    lines.append(f"{t.replace('_', chr(92) + '_') if first else ''} & {seed}/{world} & {w['probe_r2']:.2f} & "
                                 f"{w['probe_r2_count']:.2f} & {w['probe_r2_occupancy']:.2f} & {w['probe_r2_slots']:.2f} & "
                                 f"{w['latent_std_min']:.2f} & {w['invariants_kept']} & {w['r2']['energy']:.2f} & {w['r2']['count']:.2f}\\\\")
                    first = False
            lines.append("\\midrule")
        lines[-1] = "\\bottomrule"
        lines += ["\\end{tabular}",
                  "\\par\\smallskip\\footnotesize probe: the Stage C gate, mean of the count and occupancy read-outs. "
                  "slots: the old slot-indexed read-out, not gating. kept: invariants kept. R$^2_E$, R$^2_N$: kept "
                  "invariants against true energy and ball count (diagnostic only). Primary run; the secondary "
                  "options reuse the same encoders.",
                  "\\end{table}"]
        tex.append("\n".join(lines))
        # invariants kept, per option
        lines = ["\\begin{table}[t]\\centering\\small",
                 "\\caption{Stage C: invariants kept by the selection rule, per option, seed and world.}\\label{tab:stageCkept}",
                 "\\begin{tabular}{l" + "c" * 6 + "}\\toprule"]
        heads = [f"{seed}/{world[:4]}" for seed in sorted(entries[0][1]["runs"], key=int) for world in entries[0][1]["runs"][seed]["worlds"]]
        lines.append("Option & " + " & ".join(heads) + "\\\\\\midrule")
        all_kept = []
        for t, s in entries:
            ks = [str(w["invariants_kept"]) for seed in sorted(s["runs"], key=int) for w in s["runs"][seed]["worlds"].values()]
            all_kept += [int(k) for k in ks]
            lines.append(f"\\texttt{{{t.replace('_', chr(92) + '_')}}} & " + " & ".join(ks) + "\\\\")
        lines += ["\\bottomrule\\end{tabular}", "\\end{table}"]
        tex.append("\n".join(lines))
        put("CKeptAllMax", str(max(all_kept)))
        put("CNOptions", str(len(entries)))
    return tex


def diagnostics(idx) -> list[str]:
    p = RES / "encoder_diagnostics" / "summary.json"
    if not p.exists():
        return []
    d = load(p)
    lines = ["\\begin{table}[t]\\centering\\small",
             "\\caption{Why the learned latent hides energy. Read-outs of the frame from the frozen Stage C encoders, "
             "and two label-free properties of the latent next to the Stage B reference latents.}\\label{tab:diag}",
             "\\begin{tabular}{lcccccc}\\toprule",
             "Latent & lin.\\ count & MLP count & lin.\\ occup. & MLP occup. & roughness & clean steps\\\\\\midrule"]
    for r in d["encoders"]:
        lines.append(f"{r['latent']} & {r['linear_count']:.2f} & {r['mlp_count']:.2f} & {r['linear_occupancy']:.2f} & "
                     f"{r['mlp_occupancy']:.2f} & {r['roughness']:.3f} & {r['clean_share']:.3f}\\\\")
    lines.append("\\midrule")
    for r in d["references"]:
        lines.append(f"{r['latent']} & & & & & {r['roughness']:.3f} & {r['clean_share']:.3f}\\\\")
    lines += ["\\bottomrule\\end{tabular}",
              "\\par\\smallskip\\footnotesize Read-outs are $R^2$ on fresh lawful clips (diagnostic). Roughness: median "
              "norm of the standardised latent's second difference over that of its first difference. Clean steps: the "
              "share of steps the event detector trusts. Both are label-free.",
              "\\end{table}"]
    enc = d["encoders"]
    put("DRoughMin", min(r["roughness"] for r in enc), "{:.3f}")
    put("DRoughMax", max(r["roughness"] for r in enc), "{:.3f}")
    put("DCleanMin", min(r["clean_share"] for r in enc), "{:.3f}")
    put("DMlpOccMax", max(r["mlp_occupancy"] for r in enc))
    put("DLinOccMax", max(r["linear_occupancy"] for r in enc))
    put("DMlpCountMax", max(r["mlp_count"] for r in enc))
    put("DLinCountMax", max(r["linear_count"] for r in enc))
    for r in d["references"]:
        key = {"ideal frame latent": "Ideal"}.get(r["latent"], r["latent"].replace("Stage B mixer alpha ", "A").replace(", noise ", "N"))
        put(macro_name("DRef", key, "rough"), r["roughness"], "{:.3f}")
        put(macro_name("DRef", key, "clean"), r["clean_share"], "{:.3f}")
    seeds = ", ".join(str(r["seed"]) for r in enc)
    appendix.append(("Table~\\ref{tab:diag}", f"encoder\\_diagnostics/{d['config']['checkpoints'].replace('_', chr(92) + '_')}", seeds))
    appendix.append(("Table~\\ref{tab:diag} (references)", "encoder\\_diagnostics/reference\\_latents", "none"))
    return ["\n".join(lines)]


def agreement(idx) -> list[str]:
    p = RES / "agreement" / "summary.json"
    if not p.exists():
        return []
    rows = load(p)["rows"]
    lines = ["\\begin{table}[t]\\centering\\small",
             "\\caption{H4: canonical correlations between the kept invariants of two independently fitted encoders, "
             "on the same fresh lawful clips.}\\label{tab:agreement}",
             "\\begin{tabular}{llcccc}\\toprule",
             "Encoders & seeds & kept & canonical correlations & R$^2_E$ (a / b) & $\\ge 0.9$\\\\\\midrule"]
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        name = f"ideal, mixer $\\alpha={r['alpha']:g}$" if r["mode"] == "control" else f"pixel JEPA ({r['checkpoints'].replace('_', chr(92) + '_')})"
        tag = f"agreement/control\\_alpha{r['alpha']:g}" if r["mode"] == "control" else f"agreement/pixels\\_{r['checkpoints'].replace('_', chr(92) + '_')}"
        groups[(name, tag)].append(r)
    for (name, tag), rs in groups.items():
        for i, r in enumerate(rs):
            cc = ", ".join(f"{x:.3f}" for x in r["canonical_correlations"]) or "none"
            lines.append(f"{name if i == 0 else ''} & {r['seeds'][0]}, {r['seeds'][1]} & {r['k_a']} / {r['k_b']} & {cc} & "
                         f"{r['r2_energy_a']:.2f} / {r['r2_energy_b']:.2f} & {r['n_at_least_0.9']}\\\\")
        lines.append("\\midrule")
        seeds = sorted({s for r in rs for s in r["seeds"]})
        appendix.append(("Table~\\ref{tab:agreement}", tag, ", ".join(map(str, seeds))))
        key = name.split("(")[-1] if "pixel" in name else f"a{rs[0]['alpha']:g}"
        ccs = [x for r in rs for x in r["canonical_correlations"]]
        put(macro_name("H", "pix" if "pixel" in name else "ctl", key, "ccmin"), min(ccs) if ccs else 0.0, "{:.3f}")
        put(macro_name("H", "pix" if "pixel" in name else "ctl", key, "rtwomax"),
            max(max(r["r2_energy_a"], r["r2_energy_b"]) for r in rs))
        put(macro_name("H", "pix" if "pixel" in name else "ctl", key, "kmax"), str(max(max(r["k_a"], r["k_b"]) for r in rs)))
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}",
              "\\par\\smallskip\\footnotesize kept: invariants kept by each encoder. R$^2_E$: how well each encoder's kept "
              "invariants explain the true energy on the shared clips (diagnostic only). Last column: canonical "
              "correlations of at least 0.9 (the H4 threshold).",
              "\\end{table}"]
    return ["\n".join(lines)]


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    idx = ledger_index()
    blocks = {"stageA": stage_a(idx), "stageB": stage_b(idx), "stageC": stage_c(idx), "agreement": agreement(idx), "diagnostics": diagnostics(idx)}
    blocks["stageA_sens"] = blocks["stageA"][1:]
    blocks["stageA"] = blocks["stageA"][:1]
    for name, tex in blocks.items():
        (GEN / f"{name}.tex").write_text("\n\n".join(tex) + "\n", encoding="utf-8")
    (GEN / "numbers.tex").write_text(
        "% Generated by paper/make_figures.py from results/. Do not edit.\n"
        + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n", encoding="utf-8")
    rows = []
    for what, tag, seeds in appendix:
        raw = tag.replace("\\_", "_")
        info = idx.get(raw)
        commits = ", ".join(sorted(info["commits"])) if info else "MISSING FROM LEDGER"
        rows.append(f"{what} & \\texttt{{{tag}}} & {seeds} & {commits}\\\\")
    (GEN / "appendix_ledger.tex").write_text(
        "\\begin{longtable}{p{3.6cm}p{6.2cm}p{1.6cm}p{1.6cm}}\\toprule\n"
        "Claim or table & Ledger experiment & Seeds & Commit\\\\\\midrule\\endhead\n"
        + "\n".join(rows) + "\n\\bottomrule\\end{longtable}\n", encoding="utf-8")
    missing = [r for r in rows if "MISSING" in r]
    print(f"{len(macros)} macros, {len(rows)} ledger rows, {len(missing)} missing from the ledger")
    for r in missing:
        print("  missing:", r)


if __name__ == "__main__":
    main()
