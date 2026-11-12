"""Stage C: the first real test. The same three-way comparison, on latents a pixel JEPA learned by itself.

On the GPU server:
    python experiments/pilot_pixels.py --device cuda --steps 20000

Smoke test on a CPU (checks the code path only; the numbers mean nothing):
    python experiments/pilot_pixels.py --device cpu --steps 30 --n-train 16 --n-val 8 --n-test 6 \
        --seeds 0 --families energy --out /tmp/pixels_smoke

Nothing here is tuned. The first job on the server is to get a model whose
latents encode position at all (see "probe_r2" in the report) before reading
anything into the AUROCs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conserve import ledger  # noqa: E402
from conserve.events import clean_mask  # noqa: E402
from conserve.jepa import TinyJEPA, train  # noqa: E402
from conserve.latents import phase_space  # noqa: E402
from conserve.probes import encoder_probes  # noqa: E402
from conserve.pipeline import SCORES, PilotConfig, compare, fit_latents, score_latent  # noqa: E402
from conserve.runner import decide, markdown_report, summarise  # noqa: E402
from conserve.scores import oracle_errors  # noqa: E402
from conserve.worlds import CONDITIONS, FAMILIES, ScenarioConfig, make_dataset, world_for  # noqa: E402
from conserve.worlds.render import render  # noqa: E402


def invariant_r2(fl, zs, clips, pc) -> dict:
    """How well the kept invariants explain the true energy and ball count (diagnostic only)."""
    if fl.head.k == 0:
        return {"energy": 0.0, "count": 0.0}
    rows, energy, count = [], [], []
    for z, c in zip(zs, clips):
        clean = clean_mask(z, fl.threshold, pc.span)
        rows.append(fl.head(phase_space(z, pc.span))[clean])
        energy.append(c.kinetic_energy[1:][clean])
        count.append(c.count[1:][clean].astype(float))
    x = np.concatenate(rows)
    x = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    out = {}
    for name, target in (("energy", np.concatenate(energy)), ("count", np.concatenate(count))):
        coef, *_ = np.linalg.lstsq(x, target, rcond=None)
        out[name] = float(1.0 - (target - x @ coef).var() / max(target.var(), 1e-300))
    return out


def run_world(families, sc, pc, args, seed, device, out_dir: Path, key: str):
    """Train one model on lawful clips of one world, then score every family that lives in it."""
    cfg = world_for(families[0], sc)
    train_clips = make_dataset(families[0], "possible", pc.n_train, sc, seed * 1000 + 1)
    val_clips = make_dataset(families[0], "possible", pc.n_val, sc, seed * 1000 + 2)
    frames = torch.from_numpy(np.stack([render(c, cfg, args.size) for c in train_clips])).to(device)

    torch.manual_seed(seed)
    model = TinyJEPA(args.dim, args.size).to(device)
    if args.load_ckpt_dir is not None:
        # a secondary option on the same encoders: only the invariant fit changes
        src = args.load_ckpt_dir / f"seed{seed}_{key}.pt"
        model.load_state_dict(torch.load(src, map_location=device))
        log = {"loaded_from": str(src)}
    else:
        log = train(model, frames, args.steps, args.batch, args.lam, args.lr)
        torch.save(model.state_dict(), args.ckpt_out / f"seed{seed}_{key}.pt")

    def encode(clip):
        return model.encode(render(clip, cfg, args.size), device)

    zs, zs_val = [encode(c) for c in train_clips], [encode(c) for c in val_clips]
    fl = fit_latents(zs, zs_val, pc, seed, with_predictor=False)
    fl.pe_floor = float(np.median(np.concatenate([model.prediction_errors(z, device) for z in zs[:50]])))
    stacked = np.concatenate(zs)
    info = {
        "invariants_kept": fl.head.k,
        "slowness_val": [round(float(x), 5) for x in fl.head_all.slowness_val],
        "r2": invariant_r2(fl, zs_val, val_clips, pc),
        **encoder_probes(zs, train_clips, zs_val, val_clips, cfg),
        "latent_std_min": float(stacked.std(0).min()),
        "latent_std_median": float(np.median(stacked.std(0))),
        "train_log": log,
    }
    print(f"  [{key}] probe R2 {info['probe_r2']:.3f} (count {info['probe_r2_count']:.3f}, "
          f"occupancy {info['probe_r2_occupancy']:.3f}, slots {info['probe_r2_slots']:.3f}), "
          f"latent std min {info['latent_std_min']:.3f}, invariants kept {fl.head.k}, "
          f"R2 energy {info['r2']['energy']:.3f}, R2 count {info['r2']['count']:.3f}", flush=True)

    results = {}
    for family in families:
        scores = {}
        for ci, condition in enumerate(CONDITIONS):
            clips = make_dataset(family, condition, pc.n_test, sc, seed * 1000 + 10 + 10 * FAMILIES.index(family) + ci)
            rows = []
            for c in clips:
                z = encode(c)
                row = score_latent(fl, z, pc, errors=model.prediction_errors(z, device))
                row["oracle_pe"] = float(oracle_errors(c, cfg).max())
                rows.append(row)
            scores[condition] = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        results[family] = {
            "aurocs": compare(scores, SCORES),
            "medians": {c: {s: float(np.median(scores[c][s])) for s in SCORES} for c in CONDITIONS},
        }
    return results, info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--families", nargs="+", default=list(FAMILIES), choices=FAMILIES)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lam", type=float, default=0.1, help="weight of the Gaussianity regulariser (untuned)")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--size", type=int, default=48)
    p.add_argument("--pca-dim", type=int, default=16)
    p.add_argument("--span", type=int, default=1)
    p.add_argument("--n-train", type=int, default=400)
    p.add_argument("--n-val", type=int, default=100)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "pilot_pixels")
    p.add_argument("--ckpt-dir", type=Path, default=None, help="where model checkpoints go (default: --out/<tag>)")
    p.add_argument("--load-ckpt-dir", type=Path, default=None,
                   help="load trained encoders from here instead of training (for secondary options of a frozen run)")
    p.add_argument("--ledger", type=Path, default=ROOT / "results" / "ledger.jsonl")
    args = p.parse_args()

    device = torch.device(args.device)
    sc = ScenarioConfig()
    pc = PilotConfig(n_train=args.n_train, n_val=args.n_val, n_test=args.n_test, pca_dim=args.pca_dim, span=args.span)
    tag = f"dim{args.dim}_lam{args.lam:g}_steps{args.steps}_pca{args.pca_dim}_span{args.span}"
    if args.size != 48:
        tag += f"_size{args.size}"
    out = args.out / tag
    out.mkdir(parents=True, exist_ok=True)
    args.ckpt_out = args.ckpt_dir / tag if args.ckpt_dir else out
    args.ckpt_out.mkdir(parents=True, exist_ok=True)
    model_args = {k: getattr(args, k) for k in ("steps", "batch", "lr", "lam", "dim", "size")}
    config = {"scenario": asdict(sc), "pilot": asdict(pc), "model": model_args, "families": args.families}

    runs = {}
    for seed in args.seeds:
        t0 = time.time()
        runs[seed] = {"families": {}, "worlds": {}}
        groups = {
            "specular": [f for f in args.families if f != "stochastic"],
            "rough": [f for f in args.families if f == "stochastic"],
        }
        for key, families in groups.items():
            if not families:
                continue
            print(f"seed {seed}, {key} world: training on {device}", flush=True)
            results, info = run_world(families, sc, pc, args, seed, device, out, key)
            runs[seed]["families"].update(results)
            runs[seed]["worlds"][key] = info
        ledger.append(args.ledger, f"pilot_pixels/{tag}", config, runs[seed], seed)
        print(f"seed {seed}: {time.time() - t0:.0f}s", flush=True)

    summary = summarise(runs, args.families)
    verdict = decide(summary)
    (out / "summary.json").write_text(
        json.dumps({"config": config, "runs": runs, "summary": summary, "verdict": verdict}, indent=2), encoding="utf-8"
    )
    report = markdown_report(f"Pixel pilot ({tag})", config, runs, summary, verdict)
    probes = "\n".join(
        f"- seed {seed}, {key}: probe R2 {w['probe_r2']:.3f} (count {w['probe_r2_count']:.3f}, "
        f"occupancy {w['probe_r2_occupancy']:.3f}; slot-indexed, not gating, {w['probe_r2_slots']:.3f}), latent std min/median "
        f"{w['latent_std_min']:.3f}/{w['latent_std_median']:.3f}"
        for seed in sorted(runs) for key, w in runs[seed]["worlds"].items()
    )
    report += "\n## Is the encoder any good?\n\nProbe R2 (the Stage C gate, docs/hypotheses.md) is the mean of two linear read-outs from the latent on held-out lawful clips: the number of balls, and a 12 by 12 occupancy map of the frame (1.0 is perfect). If it is below 0.9, fix the model before reading the AUROCs.\n\n" + probes + "\n"
    (out / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
