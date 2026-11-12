"""Stage B: where does it break?

Sweeps how far the latent is from linear identifiability (alpha), how noisy it
is from frame to frame (noise), and the velocity baseline (span), and reports
whether low-degree invariants still exist and still separate the impossible from
the improbable. CPU only.

    python experiments/robustness_sweep.py
    python experiments/robustness_sweep.py --alphas 0 0.1 0.3 1 --noises 0 0.003 0.01 --spans 1 4 --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conserve import ledger  # noqa: E402
from conserve.pipeline import PE_SCORES, PilotConfig  # noqa: E402
from conserve.runner import KEY, run_seed, summarise  # noqa: E402
from conserve.worlds import ScenarioConfig  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.1, 0.3, 1.0])
    p.add_argument("--noises", type=float, nargs="+", default=[0.0, 0.003, 0.01])
    p.add_argument("--spans", type=int, nargs="+", default=[1, 4])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--families", nargs="+", default=["energy", "stochastic"])
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--n-val", type=int, default=60)
    p.add_argument("--n-test", type=int, default=100)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "robustness")
    p.add_argument("--ledger", type=Path, default=ROOT / "results" / "ledger.jsonl")
    args = p.parse_args()

    sc = ScenarioConfig()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for alpha, noise, span in itertools.product(args.alphas, args.noises, args.spans):
        pc = PilotConfig(
            n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
            alpha=alpha, noise=noise, span=span, predictor_iters=80,
        )
        t0 = time.time()
        runs = {}
        for seed in args.seeds:
            runs[seed] = run_seed(seed, sc, pc, args.families)
            ledger.append(
                args.ledger, f"robustness/alpha{alpha:g}_noise{noise:g}_span{span}",
                {"scenario": asdict(sc), "pilot": asdict(pc), "families": args.families}, runs[seed], seed,
            )
        summary = summarise(runs, args.families)
        row = {"alpha": alpha, "noise": noise, "span": span, "families": {}}
        for family in args.families:
            world = "rough" if family == "stochastic" else "specular"
            row["families"][family] = {
                "kept": float(np.mean([runs[s]["worlds"][world]["invariants_kept"] for s in runs])),
                "r2_energy": float(np.mean([runs[s]["worlds"][world]["r2"]["energy"] for s in runs])),
                "inv": summary[family]["inv"][KEY]["mean"],
                "inv_rel": summary[family]["inv_rel"][KEY]["mean"],
                "best_pe": max(summary[family][s][KEY]["mean"] for s in PE_SCORES),
            }
        rows.append(row)
        print(f"alpha={alpha:g} noise={noise:g} span={span}: {time.time() - t0:.0f}s  {json.dumps(row['families'])}", flush=True)

    (args.out / "summary.json").write_text(json.dumps({"rows": rows, "seeds": args.seeds}, indent=2), encoding="utf-8")
    lines = [
        "# Robustness sweep", "",
        f"Seeds: {args.seeds}. AUROCs are impossible vs improbable, mean over seeds.", "",
        "alpha: nonlinearity of the latent (0 = a rotation of the ideal latent). noise: per-frame latent noise in units of each dimension's spread. span: steps in the velocity baseline.", "",
    ]
    for family in args.families:
        lines += [f"## {family}", "", "| alpha | noise | span | invariants kept | R2 vs true energy | inv | inv_rel | best prediction error |", "|---|---|---|---|---|---|---|---|"]
        for row in rows:
            f = row["families"][family]
            lines.append(
                f"| {row['alpha']:g} | {row['noise']:g} | {row['span']} | {f['kept']:.1f} | {f['r2_energy']:.3f} | "
                f"{f['inv']:.3f} | {f['inv_rel']:.3f} | {f['best_pe']:.3f} |"
            )
        lines.append("")
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
