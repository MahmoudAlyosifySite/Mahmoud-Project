"""Stage A/B: the three-way pilot on synthetic billiards latents.

    python experiments/pilot_synthetic.py                      # Stage A: ideal frame latents, seeds 1-5
    python experiments/pilot_synthetic.py --alpha 0.3          # Stage B: a bent, non-identifiable latent
    python experiments/pilot_synthetic.py --alpha 0.1 --noise 0.003 --span 4

Seed 0 was used to explore while the code was written. Report seeds 1 and up.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conserve import ledger  # noqa: E402
from conserve.pipeline import PilotConfig  # noqa: E402
from conserve.runner import decide, markdown_report, run_seed, summarise  # noqa: E402
from conserve.worlds import FAMILIES, ScenarioConfig  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--families", nargs="+", default=list(FAMILIES), choices=FAMILIES)
    p.add_argument("--alpha", type=float, default=None, help="nonlinearity of the latent mixer; omit for raw frame latents")
    p.add_argument("--noise", type=float, default=0.0, help="per-frame latent noise (needs --alpha, use --alpha 0 for a pure rotation)")
    p.add_argument("--span", type=int, default=1, help="baseline in steps of the velocity estimate")
    p.add_argument("--n-train", type=int, default=300)
    p.add_argument("--n-val", type=int, default=100)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "pilot_synthetic")
    p.add_argument("--ledger", type=Path, default=ROOT / "results" / "ledger.jsonl")
    args = p.parse_args()
    if args.noise > 0 and args.alpha is None:
        p.error("--noise needs --alpha (use --alpha 0 for a rotation plus noise)")

    sc = ScenarioConfig()
    pc = PilotConfig(
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        alpha=args.alpha, noise=args.noise, span=args.span,
    )
    tag = "frames" if args.alpha is None else f"alpha{args.alpha:g}_noise{args.noise:g}"
    tag += f"_span{args.span}"
    out = args.out / tag
    out.mkdir(parents=True, exist_ok=True)
    config = {"scenario": asdict(sc), "pilot": asdict(pc), "families": args.families}

    runs = {}
    for seed in args.seeds:
        t0 = time.time()
        runs[seed] = run_seed(seed, sc, pc, args.families)
        ledger.append(args.ledger, f"pilot_synthetic/{tag}", config, runs[seed], seed)
        print(f"seed {seed}: {time.time() - t0:.0f}s", flush=True)

    summary = summarise(runs, args.families)
    verdict = decide(summary)
    (out / "summary.json").write_text(
        json.dumps({"config": config, "runs": runs, "summary": summary, "verdict": verdict}, indent=2), encoding="utf-8"
    )
    report = markdown_report(f"Synthetic pilot ({tag})", config, runs, summary, verdict)
    (out / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
