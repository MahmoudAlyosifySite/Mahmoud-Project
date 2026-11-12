"""H4: do two independently trained encoders of one world agree on what is conserved?

    python experiments/agreement.py control                # ideal latents through two random mixers (positive control)
    python experiments/agreement.py pixels --ckpt-dir /tmp/ckpt/<tag> --dim 64 --size 48 --seeds 1 2

For each encoder, invariants are fitted on that encoder's own lawful training clips,
exactly as in Stage A or C. Both are then evaluated on one shared set of fresh
lawful clips, on the steps that both call clean, and the canonical correlations
between the two sets of kept invariants are computed (`conserve.agreement`).
H4 (docs/hypotheses.md) asks for canonical correlation at least 0.9. The second
half of H4, filtering by agreement to lower false alarms, is not implemented.

Only lawful clips are used. Ground truth enters only the diagnostic R2 columns.
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
from conserve.agreement import canonical_correlations  # noqa: E402
from conserve.events import clean_mask  # noqa: E402
from conserve.latents import phase_space  # noqa: E402
from conserve.pipeline import PilotConfig, fit_latents, fit_world, invariant_values  # noqa: E402
from conserve.worlds import ScenarioConfig, make_dataset, world_for  # noqa: E402

SHARED_SEED = 777_000  # dataset seed of the shared evaluation clips; never used for fitting


def r2_against(values: np.ndarray, target: np.ndarray) -> float:
    x = np.concatenate([values, np.ones((len(values), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(x, target, rcond=None)
    return float(1.0 - (target - x @ coef).var() / max(target.var(), 1e-300))


def compare_pair(vals_a, vals_b, clips) -> dict:
    """vals_*: per clip (values (T-1, k), clean mask (T-1,)). Uses steps clean in both."""
    a_rows, b_rows, energy, count = [], [], [], []
    for (va, ca), (vb, cb), c in zip(vals_a, vals_b, clips):
        both = ca & cb
        a_rows.append(va[both])
        b_rows.append(vb[both])
        energy.append(c.kinetic_energy[1:][both])
        count.append(c.count[1:][both].astype(float))
    a, b = np.concatenate(a_rows), np.concatenate(b_rows)
    e, n = np.concatenate(energy), np.concatenate(count)
    cc = canonical_correlations(a, b) if a.shape[1] and b.shape[1] else np.zeros(0)
    return {
        "k_a": int(a.shape[1]), "k_b": int(b.shape[1]), "steps": int(len(a)),
        "canonical_correlations": [round(float(x), 5) for x in cc],
        "n_at_least_0.9": int((cc >= 0.9).sum()),
        "r2_energy_a": r2_against(a, e) if a.shape[1] else 0.0, "r2_energy_b": r2_against(b, e) if b.shape[1] else 0.0,
        "r2_count_a": r2_against(a, n) if a.shape[1] else 0.0, "r2_count_b": r2_against(b, n) if b.shape[1] else 0.0,
    }


def run_control(args, sc: ScenarioConfig) -> list[dict]:
    rows = []
    shared = make_dataset("energy", "possible", args.n_shared, sc, SHARED_SEED)
    for alpha in args.alphas:
        pc = PilotConfig(n_train=args.n_train, n_val=args.n_val, alpha=alpha, predictor_iters=1)
        worlds = {s: fit_world("energy", sc, pc, s) for s in args.seeds}
        vals = {s: invariant_values(worlds[s], shared, pc, np.random.default_rng([s, 9])) for s in args.seeds}
        for sa, sb in itertools.combinations(args.seeds, 2):
            row = {"mode": "control", "alpha": alpha, "seeds": [sa, sb], **compare_pair(vals[sa], vals[sb], shared)}
            rows.append(row)
            ledger.append(args.ledger, f"agreement/control_alpha{alpha:g}", {"scenario": asdict(sc), "pilot": asdict(pc),
                          "seeds": [sa, sb], "n_shared": args.n_shared}, row, None)
            print(json.dumps(row), flush=True)
    return rows


def run_pixels(args, sc: ScenarioConfig) -> list[dict]:
    import torch

    from conserve.jepa import TinyJEPA
    from conserve.worlds.render import render

    cfg = world_for("energy", sc)
    pc = PilotConfig(n_train=args.n_train, n_val=args.n_val, pca_dim=args.pca_dim, span=args.span)
    shared = make_dataset("energy", "possible", args.n_shared, sc, SHARED_SEED)
    vals = {}
    for seed in args.seeds:
        model = TinyJEPA(args.dim, args.size)
        model.load_state_dict(torch.load(args.ckpt_dir / f"seed{seed}_specular.pt", map_location="cpu"))
        model.to(args.device)

        def encode(c):
            return model.encode(render(c, cfg, args.size), args.device)

        # the same clips and fit as Stage C for this seed (experiments/pilot_pixels.py, run_world)
        train = make_dataset("energy", "possible", pc.n_train, sc, seed * 1000 + 1)
        val = make_dataset("energy", "possible", pc.n_val, sc, seed * 1000 + 2)
        fl = fit_latents([encode(c) for c in train], [encode(c) for c in val], pc, seed, with_predictor=False)
        out = []
        for c in shared:
            z = encode(c)
            out.append((fl.head(phase_space(z, pc.span)), clean_mask(z, fl.threshold, pc.span)))
        vals[seed] = out
    rows = []
    for sa, sb in itertools.combinations(args.seeds, 2):
        row = {"mode": "pixels", "checkpoints": str(args.ckpt_dir.name), "seeds": [sa, sb], **compare_pair(vals[sa], vals[sb], shared)}
        rows.append(row)
        ledger.append(args.ledger, f"agreement/pixels_{args.ckpt_dir.name}", {"scenario": asdict(sc), "pilot": asdict(pc),
                      "model": {"dim": args.dim, "size": args.size}, "seeds": [sa, sb], "n_shared": args.n_shared}, row, None)
        print(json.dumps(row), flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["control", "pixels"])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.3, 1.0])
    p.add_argument("--n-train", type=int, default=300)
    p.add_argument("--n-val", type=int, default=100)
    p.add_argument("--n-shared", type=int, default=100)
    p.add_argument("--ckpt-dir", type=Path)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--size", type=int, default=48)
    p.add_argument("--pca-dim", type=int, default=16)
    p.add_argument("--span", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "agreement")
    p.add_argument("--ledger", type=Path, default=ROOT / "results" / "ledger.jsonl")
    args = p.parse_args()
    if args.mode == "pixels" and args.ckpt_dir is None:
        p.error("pixels needs --ckpt-dir")
    if args.mode == "pixels" and args.n_train != 400:
        print("note: Stage C fits on --n-train 400 by default; pass the same value to reproduce its fit", flush=True)
    sc = ScenarioConfig()
    t0 = time.time()
    rows = run_control(args, sc) if args.mode == "control" else run_pixels(args, sc)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "summary.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"rows": []}
    keep = [r for r in old["rows"] if not (r["mode"] == args.mode and r.get("checkpoints") == (args.ckpt_dir.name if args.ckpt_dir else None))]
    path.write_text(json.dumps({"rows": keep + rows}, indent=2), encoding="utf-8")
    print(f"wall {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
