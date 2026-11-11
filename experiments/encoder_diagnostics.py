"""Why does a learned latent hide what is conserved? Diagnostics for Stage C encoders.

    python experiments/encoder_diagnostics.py --ckpt-dir /tmp/ckpt/<tag> --dim 256 --size 48 --seeds 1 2 3

For each trained encoder (specular world), on fresh lawful clips:

- linear and nonlinear (MLP) read-outs of the gate targets (ball count, occupancy map).
  If the MLP reads the frame well and the linear probe does not, the latent carries
  the state but bent, which Stage B shows hides low-degree invariants.
- roughness: the median norm of the latent's second difference over the median norm of
  its first difference, on standardised latents. A latent that moves smoothly between
  bounces has low roughness; frame-to-frame noise raises it. Label-free.
- the share of steps the event detector calls clean.

The same two label-free numbers are computed for reference latents: the ideal frame
latent, and the Stage B mixer at alpha 0.3 (no noise) and at alpha 0 with noise 0.003
and 0.01, where Stage B located the failure. Diagnostic only; ground truth enters
only the probes. Results go to the ledger and to results/encoder_diagnostics/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conserve import ledger  # noqa: E402
from conserve.events import clean_mask, fit_event_threshold  # noqa: E402
from conserve.latents import Mixer, frame_latent  # noqa: E402
from conserve.probes import ball_count, linear_probe_r2, occupancy  # noqa: E402
from conserve.worlds import ScenarioConfig, make_dataset, world_for  # noqa: E402

DIAG_SEED = 555_000  # dataset seeds of the diagnostic clips; never used for fitting anything else


def roughness(zs: list[np.ndarray], ref: list[np.ndarray]) -> float:
    """Median |second difference| / median |first difference|, after standardising with `ref`."""
    stacked = np.concatenate(ref)
    mean, std = stacked.mean(0), stacked.std(0)
    std = np.where(std < 1e-9, 1.0, std)
    d1, d2 = [], []
    for z in zs:
        s = (z - mean) / std
        d1.append(np.linalg.norm(s[1:] - s[:-1], axis=1))
        d2.append(np.linalg.norm(s[2:] - 2 * s[1:-1] + s[:-2], axis=1))
    return float(np.median(np.concatenate(d2)) / max(np.median(np.concatenate(d1)), 1e-300))


def clean_share(zs: list[np.ndarray], zs_fit: list[np.ndarray]) -> float:
    thr = fit_event_threshold(zs_fit)
    return float(np.mean(np.concatenate([clean_mask(z, thr, 1) for z in zs])))


def _r2(y, pred, weighted: bool) -> float:
    resid = y - pred
    var = y.var(0)
    keep = var > 1e-12
    if weighted:
        return float(1.0 - resid.var(0)[keep].sum() / var[keep].sum())
    return float((1.0 - resid.var(0)[keep] / var[keep]).mean())


def mlp_r2(x_train, y_train, x_test, y_test, clip_len: int, seed: int, weighted: bool) -> float:
    """Nonlinear read-out. The L2 weight is chosen on whole held-out training clips (the last
    fifth), never on frames of clips it was fitted on, so that it cannot memorise clips."""
    from sklearn.neural_network import MLPRegressor

    y_train = y_train.reshape(len(y_train), -1)
    y_test = y_test.reshape(len(y_test), -1)
    mean, std = x_train.mean(0), x_train.std(0)
    std = np.where(std < 1e-9, 1.0, std)
    ym, ys = y_train.mean(0), y_train.std(0)
    ys = np.where(ys < 1e-9, 1.0, ys)
    xs, yn = (x_train - mean) / std, (y_train - ym) / ys
    n_clips = len(x_train) // clip_len
    cut = clip_len * (n_clips - max(1, n_clips // 5))
    best, best_alpha = -np.inf, None
    for alpha in (1e-2, 1.0, 10.0):
        m = MLPRegressor(hidden_layer_sizes=(256,), alpha=alpha, max_iter=300, random_state=seed)
        m.fit(xs[:cut], yn[:cut])
        score = _r2(y_train[cut:], m.predict(xs[cut:]).reshape(len(xs) - cut, -1) * ys + ym, weighted)
        if score > best:
            best, best_alpha = score, alpha
    m = MLPRegressor(hidden_layer_sizes=(256,), alpha=best_alpha, max_iter=300, random_state=seed)
    m.fit(xs, yn)
    return _r2(y_test, m.predict((x_test - mean) / std).reshape(len(x_test), -1) * ys + ym, weighted)


def reference_rows(sc: ScenarioConfig, n: int) -> list[dict]:
    train = make_dataset("energy", "possible", n, sc, DIAG_SEED + 1)
    test = make_dataset("energy", "possible", n, sc, DIAG_SEED + 2)
    raw_train, raw_test = [frame_latent(c) for c in train], [frame_latent(c) for c in test]
    rows = [{"latent": "ideal frame latent", "roughness": roughness(raw_test, raw_train),
             "clean_share": clean_share(raw_test, raw_train)}]
    for alpha, noise in ((0.3, 0.0), (0.0, 0.003), (0.0, 0.01)):
        mixer = Mixer(raw_train[0].shape[1], alpha, noise, 1).fit(raw_train)
        rng = np.random.default_rng(1)
        a = [mixer(z, rng) for z in raw_train]
        b = [mixer(z, rng) for z in raw_test]
        rows.append({"latent": f"Stage B mixer alpha {alpha:g}, noise {noise:g}", "roughness": roughness(b, a),
                     "clean_share": clean_share(b, a)})
    return rows


def encoder_row(args, sc, seed: int) -> dict:
    import torch

    from conserve.jepa import TinyJEPA
    from conserve.worlds.render import render

    cfg = world_for("energy", sc)
    model = TinyJEPA(args.dim, args.size)
    model.load_state_dict(torch.load(args.ckpt_dir / f"seed{seed}_specular.pt", map_location="cpu"))
    model.to(args.device)
    train = make_dataset("energy", "possible", args.n, sc, DIAG_SEED + 10 * seed + 1)
    test = make_dataset("energy", "possible", args.n, sc, DIAG_SEED + 10 * seed + 2)
    zt = [model.encode(render(c, cfg, args.size), args.device) for c in train]
    zv = [model.encode(render(c, cfg, args.size), args.device) for c in test]
    xt, xv = np.concatenate(zt), np.concatenate(zv)
    ct, cv = np.concatenate([ball_count(c) for c in train]), np.concatenate([ball_count(c) for c in test])
    ot, ov = np.concatenate([occupancy(c, cfg) for c in train]), np.concatenate([occupancy(c, cfg) for c in test])
    return {
        "latent": f"pixel JEPA seed {seed}", "seed": seed,
        "linear_count": linear_probe_r2(xt, ct, xv, cv), "linear_occupancy": linear_probe_r2(xt, ot, xv, ov, weighted=True),
        "mlp_count": mlp_r2(xt, ct, xv, cv, sc.T, seed, False), "mlp_occupancy": mlp_r2(xt, ot, xv, ov, sc.T, seed, True),
        "roughness": roughness(zv, zt), "clean_share": clean_share(zv, zt),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--size", type=int, default=48)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--n", type=int, default=100, help="clips for fitting and for testing the probes")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "encoder_diagnostics")
    p.add_argument("--ledger", type=Path, default=ROOT / "results" / "ledger.jsonl")
    args = p.parse_args()
    sc = ScenarioConfig()
    t0 = time.time()
    config = {"scenario": asdict(sc), "checkpoints": args.ckpt_dir.name, "dim": args.dim, "size": args.size, "n": args.n}
    refs = reference_rows(sc, args.n)
    ledger.append(args.ledger, "encoder_diagnostics/reference_latents", config, {"rows": refs}, None)
    rows = []
    for seed in args.seeds:
        row = encoder_row(args, sc, seed)
        rows.append(row)
        ledger.append(args.ledger, f"encoder_diagnostics/{args.ckpt_dir.name}", config, row, seed)
        print(json.dumps(row), flush=True)
    for r in refs:
        print(json.dumps(r), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps({"config": config, "encoders": rows, "references": refs}, indent=2),
                                           encoding="utf-8")
    print(f"wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
