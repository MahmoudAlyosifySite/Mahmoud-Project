"""Diversity statistics of pretraining pools, measured in a fixed reference feature space.

For every pool (a set of training images) and every class we compute

  spread    mean pairwise cosine distance between images of the class;
  spread_aug  the same between one augmented view of every image
              ("average pairwise augmentation distance within a class");
  vendi     Vendi score = exp(entropy of the eigenvalues of the cosine kernel / n),
            the effective number of distinct directions (Friedman & Dieng 2023);
  radius    coverage radius: mean cosine distance from a *test* image of the class to its
            nearest pool image of the class (how far unseen images are from what was seen);
  coverage  share of test images whose nearest same-class pool image is closer than r0, the
            median such distance for the full training set (so the full set has coverage 0.5).

All numbers are averaged over classes. ``n_eff`` converts ``radius`` into the size of a
random pool with the same coverage radius (log-linear interpolation of the random pools).

    python scripts/diversity.py --ref_run simclr_all_e100_s0 --ckpt_dir <dir>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dlt.data import load_dataset, stratified_subset   # noqa: E402
from dlt.gpu_aug import GPUAugment                      # noqa: E402
from dlt.models import ContrastiveModel, extract_features, load_encoder   # noqa: E402


@torch.no_grad()
def pool_stats(f_pool, y_pool, f_aug, f_test, y_test, C, r0=None):
    out = {"spread": [], "spread_aug": [], "vendi": [], "radius": [], "nn": []}
    for c in range(C):
        a = f_pool[y_pool == c].double()
        n = a.shape[0]
        s = a.sum(0)
        out["spread"].append((1 - ((s @ s) - n) / (n * (n - 1))).item())
        b = f_aug[y_pool == c].double()
        sb = b.sum(0)
        out["spread_aug"].append((1 - ((sb @ sb) - n) / (n * (n - 1))).item())
        ev = torch.linalg.eigvalsh(a.t() @ a / n).clamp(min=1e-12)
        ev = ev / ev.sum()
        out["vendi"].append(torch.exp(-(ev * ev.log()).sum()).item())
        t = f_test[y_test == c].double()
        d = (1 - t @ a.t()).min(1).values
        out["radius"].append(d.mean().item())
        out["nn"].append(d)
    res = {k: float(np.mean(v)) for k, v in out.items() if k != "nn"}
    nn = torch.cat(out["nn"])
    res["radius_median"] = nn.median().item()
    if r0 is not None:
        res["coverage"] = (nn <= r0).float().mean().item()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref_run", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--data_dir", default=os.environ.get("DLT_DATA", "./data"))
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--fractions", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--pools", default="results/pools")
    p.add_argument("--extra_refs", nargs="*", default=[], help="tag=run_name of further reference encoders")
    p.add_argument("--out", default="results/diversity.json")
    args = p.parse_args()
    dev = "cuda"
    data = load_dataset(args.dataset, args.data_dir)
    C = data["num_classes"]
    ckpt = torch.load(os.path.join(args.ckpt_dir, args.ref_run, "model.pt"), map_location=dev)
    model = ContrastiveModel(ckpt["args"]["backbone"]).to(dev)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ckpt["model"].items()})
    model = model.to(memory_format=torch.channels_last).eval()
    X, Y = data["train_x"].to(dev), data["train_y"].to(dev)
    Xt, Yt = data["test_x"].to(dev), data["test_y"].to(dev)
    g = torch.Generator(device=dev).manual_seed(0)
    aug = GPUAugment("simclr", data["mean"], data["std"], dev, g)
    spaces = {}
    ftr = F.normalize(extract_features(model.encoder, X, aug.normalize), dim=1)
    fte = F.normalize(extract_features(model.encoder, Xt, aug.normalize), dim=1)
    faug = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 2000):
            v = aug(X[i:i + 2000]).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16):
                faug.append(model.encoder(v).float())
    faug = torch.cat(faug)
    spaces["simclr"] = (ftr, F.normalize(faug, dim=1), fte)
    # further reference encoders (e.g. a supervised network), to check that the result does not
    # depend on measuring distances in a SimCLR feature space
    for tag_ref in args.extra_refs:
        tag, ref = tag_ref.split("=")
        enc, _ = load_encoder(os.path.join(args.ckpt_dir, ref, "model.pt"))
        fa = []
        with torch.no_grad():
            for i in range(0, X.shape[0], 2000):
                v = aug(X[i:i + 2000]).contiguous(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.float16):
                    fa.append(enc(v).float())
        spaces[tag] = (F.normalize(extract_features(enc, X, aug.normalize), dim=1), F.normalize(torch.cat(fa), dim=1),
                       F.normalize(extract_features(enc, Xt, aug.normalize), dim=1))
    # a model-free space: mean-centred pixels
    px = aug.normalize(X).flatten(1)
    pxa = aug(X).flatten(1)
    pxt = aug.normalize(Xt).flatten(1)
    spaces["pixel"] = (F.normalize(px, dim=1), F.normalize(pxa, dim=1), F.normalize(pxt, dim=1))

    pools = {}
    labels = Y.cpu().numpy()
    for f in args.fractions:
        for s in (args.seeds if f < 1 else [0]):
            pools[f"random_f{f}_s{s}"] = (np.arange(len(labels)) if f >= 1 else stratified_subset(labels, f, s),
                                           {"kind": "random", "fraction": f, "seed": s})
    for path in sorted(glob.glob(os.path.join(args.pools, "*.npy"))):
        name = os.path.basename(path)[:-4]
        kind, frac, seed = name.split("_")
        pools[name] = (np.load(path), {"kind": kind, "fraction": float(frac[1:]), "seed": int(seed[1:])})

    results = {}
    for space, (f_tr, f_aug, f_te) in spaces.items():
        full = pool_stats(f_tr, Y, f_aug, f_te, Yt, C)
        r0 = full["radius_median"]
        rows = {}
        for name, (idx, meta) in pools.items():
            ii = torch.as_tensor(idx, device=dev)
            st = pool_stats(f_tr[ii], Y[ii], f_aug[ii], f_te, Yt, C, r0)
            st.update(meta)
            st["n"] = int(len(idx))
            rows[name] = st
        # calibration: coverage radius of random pools as a function of pool size
        sizes = sorted({v["n"] for v in rows.values() if v["kind"] == "random"})
        rad = [float(np.mean([v["radius"] for v in rows.values() if v["kind"] == "random" and v["n"] == n])) for n in sizes]
        for v in rows.values():
            # radius decreases with n: interpolate log(n) as a function of radius
            v["n_eff"] = float(np.exp(np.interp(-v["radius"], [-r for r in rad], np.log(sizes))))
        results[space] = {"pools": rows, "calibration": {"n": sizes, "radius": rad}}
        print(f"== space: {space}")
        for name, v in rows.items():
            if v["seed"] == 0:
                print(f"  {name:26s} n={v['n']:6d} spread {v['spread']:.4f} spread_aug {v['spread_aug']:.4f} "
                      f"vendi {v['vendi']:.1f} radius {v['radius']:.4f} cov {v['coverage']:.3f} n_eff {v['n_eff']:.0f}")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
