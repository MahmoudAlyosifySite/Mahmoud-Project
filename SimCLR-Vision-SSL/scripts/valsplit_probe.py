"""LP with the 10% labels on the validation split for an encoder trained with --holdout (third revision).

dlt.train reports this probe (linear_lab_acc) only for runs that use labels; for the label-free base encoder of
the SimCLR->SupCon stage selection it is computed here with the same split and the same probe:
5,000 training images held out (split seed 12345) serve as the evaluation set, the probe is fitted on the 10%
labeled subset of the other 45,000 images. Writes results/runs/<name>/valprobe.json.

    python scripts/valsplit_probe.py --ckpt_dir <dir> --run val_simclr_all_s0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dlt.data import labeled_subset, load_dataset, stratified_subset   # noqa: E402
from dlt.evaluate import linear_probe                                    # noqa: E402
from dlt.gpu_aug import GPUAugment                                       # noqa: E402
from dlt.models import extract_features, load_encoder                    # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--ckpt_dir", required=True)
p.add_argument("--run", required=True)
p.add_argument("--root", default="results/runs")
p.add_argument("--data_dir", default=os.environ.get("DLT_DATA", "./data"))
args = p.parse_args()
enc, a = load_encoder(os.path.join(args.ckpt_dir, args.run, "model.pt"))
assert a.get("holdout", 0) > 0, "not a validation-split run"
data = load_dataset(a.get("dataset", "cifar10"), args.data_dir)
y = data["train_y"].cpu().numpy()
val_idx = stratified_subset(y, a["holdout"] / len(y), 12345)          # as in dlt.train
keep = np.setdiff1d(np.arange(len(y)), val_idx)
data = dict(data, train_x=data["train_x"][keep], train_y=data["train_y"][keep])
lab = labeled_subset(data, 0.1, a.get("split_seed", a.get("seed", 0)) or 0)
norm = GPUAugment("none", data["mean"], data["std"], "cuda").normalize
xs = load_dataset(a.get("dataset", "cifar10"), args.data_dir)["train_x"]
ftr = extract_features(enc, data["train_x"].cuda(), norm)
fva = extract_features(enc, xs[val_idx].cuda(), norm)
ytr, yva = data["train_y"].cuda(), torch.as_tensor(y[val_idx]).cuda()
stats = (ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True) + 1e-6)
li = torch.as_tensor(lab, device="cuda")
pred, info, _ = linear_probe(ftr[li], ytr[li], fva, data["num_classes"], stats, a.get("seed", 0))
res = {"linear_lab_acc": 100 * (pred == yva).float().mean().item(), "linear_lab_l2": info["l2"], "n_labeled": int(len(lab))}
json.dump(res, open(os.path.join(args.root, args.run, "valprobe.json"), "w"), indent=1)
print(args.run, res)
