"""Run the three-way comparison over seeds and families, summarise it, apply the go rule.

The go rule is fixed in docs/hypotheses.md. Change it there first, with a dated
note, never here alone.
"""
from __future__ import annotations

import numpy as np

from .pipeline import COMPARISONS, PE_SCORES, SCORES, PilotConfig, evaluate_family, fit_world, ground_truth_r2
from .worlds import CONDITIONS, ScenarioConfig, make_dataset

GO_FAMILIES = ("energy", "permanence", "stochastic")  # families whose violation breaks a conserved quantity
GO_MARGIN = 0.15  # H2: the invariant score must beat the best prediction-error score by this much
H1_CEILING = 0.70  # H1: the best prediction-error score stays at or below this
KEY = "impossible_vs_improbable"


def run_seed(seed: int, sc: ScenarioConfig, pc: PilotConfig, families) -> dict:
    worlds, out = {}, {"families": {}, "worlds": {}}
    for family in families:
        key = "rough" if family == "stochastic" else "specular"
        if key not in worlds:
            fw = fit_world(family, sc, pc, seed)
            clips = make_dataset(family, "possible", 40, sc, seed * 1000 + 3)
            out["worlds"][key] = {
                "invariants_kept": fw.head.k,
                "slowness_val": [round(float(x), 5) for x in fw.head_all.slowness_val],
                "r2": ground_truth_r2(fw, clips, pc, np.random.default_rng([seed, 3])),
            }
            worlds[key] = fw
        res = evaluate_family(worlds[key], family, sc, pc, seed)
        out["families"][family] = {
            "aurocs": res["aurocs"],
            "medians": {c: {s: float(np.median(res["scores"][c][s])) for s in SCORES} for c in CONDITIONS},
        }
    return out


def summarise(runs: dict[int, dict], families) -> dict:
    summary: dict = {}
    for family in families:
        summary[family] = {}
        for s in SCORES:
            summary[family][s] = {}
            for a, b in COMPARISONS:
                k = f"{a}_vs_{b}"
                vals = np.array([runs[seed]["families"][family]["aurocs"][s][k] for seed in runs])
                summary[family][s][k] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "per_seed": [float(v) for v in vals],
                }
    return summary


def decide(summary: dict) -> dict:
    """Apply H1 and H2 per family, then the go rule: H2 in at least two conservation families."""
    verdict: dict = {"families": {}}
    passed = 0
    for family in GO_FAMILIES:
        if family not in summary:
            continue
        best = max(PE_SCORES, key=lambda s: summary[family][s][KEY]["mean"])  # the baseline's best reading
        pe, inv = summary[family][best][KEY], summary[family]["inv"][KEY]
        gains = np.array(inv["per_seed"]) - np.array(pe["per_seed"])
        h1 = pe["mean"] <= H1_CEILING
        h2 = bool(gains.mean() >= GO_MARGIN and (gains > 0).all())
        passed += h2
        verdict["families"][family] = {
            "best_pe": best,
            "best_pe_auroc": pe["mean"],
            "inv_auroc": inv["mean"],
            "gain": float(gains.mean()),
            "h1": bool(h1),
            "h2": h2,
        }
    verdict["h2_families"] = int(passed)
    verdict["go"] = passed >= 2
    return verdict


def markdown_report(title: str, config: dict, runs: dict[int, dict], summary: dict, verdict: dict) -> str:
    lines = [f"# {title}", "", f"Seeds: {sorted(runs)}", "", "Config:", "", "```json"]
    import json

    lines += [json.dumps(config, indent=2, sort_keys=True), "```", ""]
    lines += ["## Verdict", ""]
    lines += [f"**{'GO' if verdict['go'] else 'NO-GO'}**: H2 holds in {verdict['h2_families']} of {len(verdict['families'])} conservation families (rule: at least 2).", ""]
    lines += ["| Family | Best prediction-error score | Its AUROC | Invariant AUROC | Gain | H1 | H2 |", "|---|---|---|---|---|---|---|"]
    for family, v in verdict["families"].items():
        lines.append(
            f"| {family} | {v['best_pe']} | {v['best_pe_auroc']:.3f} | {v['inv_auroc']:.3f} | {v['gain']:+.3f} | "
            f"{'yes' if v['h1'] else 'no'} | {'yes' if v['h2'] else 'no'} |"
        )
    lines += ["", "AUROC here is impossible vs improbable: 0.5 is chance, 1.0 is perfect, below 0.5 means the score ranks lawful rare clips above impossible ones.", ""]
    lines += ["## All AUROCs (mean +/- std over seeds)", ""]
    for family, by_score in summary.items():
        lines += [f"### {family}", "", "| Score | impossible vs possible | impossible vs improbable | improbable vs possible |", "|---|---|---|---|"]
        for s in SCORES:
            cells = [f"{by_score[s][f'{a}_vs_{b}']['mean']:.3f} +/- {by_score[s][f'{a}_vs_{b}']['std']:.3f}" for a, b in COMPARISONS]
            lines.append(f"| {s} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Invariants found", "", "| Seed | World | Kept | R2 vs true energy | R2 vs true count | Validation slowness of the first six candidates |", "|---|---|---|---|---|---|"]
    for seed in sorted(runs):
        for world, w in runs[seed]["worlds"].items():
            slow = ", ".join(f"{x:.3f}" for x in w["slowness_val"][:6])
            lines.append(f"| {seed} | {world} | {w['invariants_kept']} | {w['r2']['energy']:.4f} | {w['r2']['count']:.4f} | {slow} |")
    lines += ["", "Slowness is 0 for a perfectly conserved quantity and about 2 for one that is unrelated at long lag. R2 uses the simulator's ground truth and is a diagnostic only; nothing is fitted to it.", ""]
    return "\n".join(lines)
