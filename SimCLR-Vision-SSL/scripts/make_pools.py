"""Build labeled pools of a fixed size whose *diversity* is controlled.

All pools hold exactly the same number of images per class (and therefore the
same number of labels) as the random stratified subset; only how the images are
spread over each class changes:

  clusterK   n/K randomly chosen seed images per class, each with its K-1 nearest
             same-class neighbours (K = 1 is the random subset). Larger K = more
             redundancy, lower diversity.
  kcenter    greedy farthest-point selection inside each class (maximum coverage).

Neighbours are measured by cosine similarity in the feature space of a
label-free reference encoder (SimCLR trained on all images).

    python scripts/make_pools.py --ref_run simclr_all_e100_s0 --ckpt_dir <dir> --seeds 0 1 2
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dlt.data import load_dataset                       # noqa: E402
from dlt.gpu_aug import GPUAugment                      # noqa: E402
from dlt.models import ContrastiveModel, extract_features   # noqa: E402


def reference_features(ref_run, ckpt_dir, data, dev="cuda"):
    ckpt = torch.load(os.path.join(ckpt_dir, ref_run, "model.pt"), map_location=dev)
    model = ContrastiveModel(ckpt["args"]["backbone"]).to(dev)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ckpt["model"].items()})
    model = model.to(memory_format=torch.channels_last)
    aug = GPUAugment("none", data["mean"], data["std"], dev)
    ftr = F.normalize(extract_features(model.encoder, data["train_x"].to(dev), aug.normalize), dim=1)
    fte = F.normalize(extract_features(model.encoder, data["test_x"].to(dev), aug.normalize), dim=1)
    return ftr, fte


def cluster_pool(feat, labels, per_class, k, seed):
    rng = np.random.RandomState(seed)
    out = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        perm = rng.permutation(idx)             # same permutation as the random subset
        f = feat[torch.as_tensor(idx, device=feat.device)]
        sim = f @ f.t()
        pos = {g: i for i, g in enumerate(idx)}
        taken = torch.zeros(len(idx), dtype=torch.bool, device=feat.device)
        chosen = []
        for g in perm:
            if len(chosen) >= per_class:
                break
            i = pos[g]
            if taken[i]:
                continue
            s = sim[i].clone()
            s[taken] = -2.0
            take = s.topk(min(k, per_class - len(chosen))).indices   # includes the seed itself
            taken[take] = True
            chosen.extend(idx[take.cpu().numpy()].tolist())
        out.append(np.array(chosen[:per_class]))
    return np.sort(np.concatenate(out))


def kcenter_pool(feat, labels, per_class, seed):
    rng = np.random.RandomState(seed)
    out = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        f = feat[torch.as_tensor(idx, device=feat.device)]
        first = int(np.where(idx == rng.permutation(idx)[0])[0][0])
        chosen = [first]
        mind = 1 - f @ f[first]
        for _ in range(per_class - 1):
            nxt = int(mind.argmax())
            chosen.append(nxt)
            mind = torch.minimum(mind, 1 - f @ f[nxt])
        out.append(idx[np.array(chosen)])
    return np.sort(np.concatenate(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref_run", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--data_dir", default=os.environ.get("DLT_DATA", "./data"))
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--fraction", type=float, default=0.1)
    p.add_argument("--ks", type=int, nargs="+", default=[10, 50])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--mix", type=float, nargs="*", default=[],
                   help="also build pools in which this share of every class comes from the k-center / "
                        "redundant (k=50) construction and the rest is random")
    p.add_argument("--out", default="results/pools")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data = load_dataset(args.dataset, args.data_dir)
    labels = data["train_y"].numpy()
    ftr, _ = reference_features(args.ref_run, args.ckpt_dir, data)
    per_class = int(round(args.fraction * (labels == 0).sum()))
    for s in args.seeds:
        for k in args.ks:
            pool = cluster_pool(ftr, labels, per_class, k, s)
            np.save(os.path.join(args.out, f"cluster{k}_f{args.fraction}_s{s}.npy"), pool)
            print(f"cluster{k} seed {s}: {len(pool)} images, {np.bincount(labels[pool]).tolist()}")
        pool = kcenter_pool(ftr, labels, per_class, s)
        np.save(os.path.join(args.out, f"kcenter_f{args.fraction}_s{s}.npy"), pool)
        print(f"kcenter seed {s}: {len(pool)} images")
        for share in args.mix:
            m = int(round(share * per_class))
            for kind, part in (("kcenter", kcenter_pool(ftr, labels, m, s)), ("cluster50", cluster_pool(ftr, labels, m, 50, s))):
                rng = np.random.RandomState(1000 + s)
                fill = []
                for c in np.unique(labels):
                    rest = np.setdiff1d(np.where(labels == c)[0], part)
                    fill.append(rng.permutation(rest)[:per_class - m])
                pool = np.sort(np.concatenate([part] + fill))
                assert len(pool) == per_class * len(np.unique(labels)) and len(np.unique(pool)) == len(pool)
                np.save(os.path.join(args.out, f"mix{kind}{int(round(100 * share))}_f{args.fraction}_s{s}.npy"), pool)
                print(f"mix {kind} {share:.2f} seed {s}: {len(pool)} images")


if __name__ == "__main__":
    main()
