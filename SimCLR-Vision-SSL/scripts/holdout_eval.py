"""Linear probe whose regularisation is chosen on labels held out from pretraining (third revision).

For an encoder of seed s, the 5,000 labeled CIFAR-10 images of that seed are split as in ft_search.py and
make_holdout_pools.py (stratified 20%, split seed 777) into 4,000 "fit" and 1,000 "held-out" images.

    linear_heldout       l2 strength chosen by training on the 4,000 and scoring the 1,000, then refitted on
                         all 5,000 and evaluated on the test set (the rule a practitioner who held out 1,000
                         labels before pretraining can use)
    linear_std           the paper's LP-5k rule (l2 chosen on a random stratified 20% of the 5,000)
    heldout_val_acc      best accuracy on the 1,000 held-out images (training on the 4,000)
    fit_train_acc        accuracy of that probe on its own 4,000 training images
    cdnv_fit/_heldout/_rest/_test   class tightness on the 4,000, the 1,000, the other 45,000 training images
                         and the test images (each set with its own class means)

Results go to results/runs/<name>/holdout.json (test predictions: holdout_preds.npz).

    python scripts/holdout_eval.py --ckpt_dir <dir> --runs hold_supcon_f0.1_e200_s0 supcon_lab_f0.1_e200_s0 ...
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dlt.data import labeled_subset, load_dataset, stratified_subset   # noqa: E402
from dlt.evaluate import L2_GRID, fit_logreg, geometry, linear_probe     # noqa: E402
from dlt.gpu_aug import GPUAugment                                      # noqa: E402
from dlt.models import extract_features, load_encoder                   # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--ckpt_dir", required=True)
p.add_argument("--root", default="results/runs")
p.add_argument("--data_dir", default=os.environ.get("DLT_DATA", "./data"))
p.add_argument("--runs", nargs="+", required=True)
p.add_argument("--force", action="store_true")
args = p.parse_args()

data = load_dataset("cifar10", args.data_dir)
for k in ("train_x", "train_y", "test_x", "test_y"):
    data[k] = data[k].cuda()
C = data["num_classes"]
ytr, yte = data["train_y"], data["test_y"]
norm = GPUAugment("none", data["mean"], data["std"], "cuda").normalize

for name in args.runs:
    out = os.path.join(args.root, name, "holdout.json")
    ck = os.path.join(args.ckpt_dir, name, "model.pt")
    if os.path.exists(out) and not args.force:
        continue
    if not os.path.exists(ck):
        print("[missing checkpoint]", name, flush=True)
        continue
    enc, a = load_encoder(ck)
    assert a.get("dataset", "cifar10") == "cifar10"
    seed = a.get("seed", 0)
    labeled = labeled_subset(data, 0.1, seed)
    held = labeled[stratified_subset(ytr.cpu().numpy()[labeled], 0.2, 777)]
    fit = np.setdiff1d(labeled, held)
    if a.get("pool_file"):   # the held-out arm: check that pretraining used exactly the 4,000 fit images
        assert np.array_equal(np.sort(np.load(a["pool_file"])), fit), name
    ftr = extract_features(enc, data["train_x"], norm)
    fte = extract_features(enc, data["test_x"], norm)
    mu, sd = ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True) + 1e-6
    x, xt = (ftr - mu) / sd, (fte - mu) / sd
    li, fi, hi = (torch.as_tensor(v, device="cuda") for v in (labeled, fit, held))
    best = (-1.0, None, None)
    for l2 in L2_GRID:
        w, b = fit_logreg(x[fi], ytr[fi], C, l2)
        acc = ((x[hi] @ w + b).argmax(1) == ytr[hi]).float().mean().item()
        if acc > best[0]:
            best = (acc, l2, ((x[fi] @ w + b).argmax(1) == ytr[fi]).float().mean().item())
    w, b = fit_logreg(x[li], ytr[li], C, best[1])
    pred_ho = (xt @ w + b).argmax(1)
    res = {"linear_heldout": 100 * (pred_ho == yte).float().mean().item(),
           "linear_heldout_l2": best[1], "heldout_val_acc": 100 * best[0], "fit_train_acc": 100 * best[2]}
    pr, info, _ = linear_probe(ftr[li], ytr[li], fte, C, (mu, sd), seed)
    res["linear_std"] = 100 * (pr == yte).float().mean().item()
    res["linear_std_l2"] = info["l2"]
    rest = torch.ones(ftr.shape[0], dtype=torch.bool, device="cuda")
    rest[li] = False
    for key, f, y in (("fit", ftr[fi], ytr[fi]), ("heldout", ftr[hi], ytr[hi]), ("rest", ftr[rest], ytr[rest]),
                      ("test", fte, yte)):
        res[f"cdnv_{key}"] = geometry(f, y, C)[0]["cdnv"]
    np.savez_compressed(os.path.join(args.root, name, "holdout_preds.npz"),
                        linear_heldout=pred_ho.cpu().numpy().astype(np.int16), linear_std=pr.cpu().numpy().astype(np.int16))
    json.dump(res, open(out, "w"), indent=1)
    print(f"{name:36s} " + " ".join(f"{k} {v:.3g}" for k, v in res.items()), flush=True)
