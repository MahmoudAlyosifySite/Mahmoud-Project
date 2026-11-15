"""Label-limited evaluation of every saved encoder.

For each run, linear probes and kNN classifiers are fitted with only 1%, 5%, 10% and 20%
of the training labels (the nested stratified subsets of the run's seed, i.e. the same
images whose labels the semi-supervised methods were given). Results are written to
results/runs/<name>/probes.json and the 10%-probe predictions to probes_preds.npz.

    python scripts/reeval.py --ckpt_dir <dir>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dlt.data import load_dataset, stratified_subset   # noqa: E402
from dlt.evaluate import geometry, knn_predict, linear_probe   # noqa: E402
from dlt.gpu_aug import GPUAugment                       # noqa: E402
from dlt.models import extract_features, load_encoder    # noqa: E402

FRACTIONS = (0.01, 0.05, 0.1, 0.2, 0.5)

p = argparse.ArgumentParser()
p.add_argument("--ckpt_dir", required=True)
p.add_argument("--root", default="results/runs")
p.add_argument("--data_dir", default=os.environ.get("DLT_DATA", "./data"))
p.add_argument("--force", action="store_true")
p.add_argument("--filter", default=None)
args = p.parse_args()

cache = {}
for res_path in sorted(glob.glob(os.path.join(args.root, "*", "result.json"))):
    d = os.path.dirname(res_path)
    name = os.path.basename(d)
    if name.startswith("pilot") or (args.filter and args.filter not in name):
        continue
    ck = os.path.join(args.ckpt_dir, name, "model.pt")
    out = os.path.join(d, "probes.json")
    if not os.path.exists(ck):
        continue
    if os.path.exists(out) and not args.force and all(k in json.load(open(out)) for k in ("cdnv_rest_0.1", "linear_0.5")):
        continue
    enc, a = load_encoder(ck)
    ds = a.get("dataset", "cifar10")
    if ds == "stl10":      # only the labeled split has labels; dlt.train already reports everything
        continue
    if ds not in cache:
        data = load_dataset(ds, args.data_dir)
        for k in ("train_x", "train_y", "test_x", "test_y"):
            data[k] = data[k].cuda()
        cache[ds] = data
    data = cache[ds]
    C = data["num_classes"]
    norm = GPUAugment("none", data["mean"], data["std"], "cuda").normalize
    ftr = extract_features(enc, data["train_x"], norm)
    fte = extract_features(enc, data["test_x"], norm)
    stats = (ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True) + 1e-6)
    seed = a.get("split_seed", a.get("seed", 0))
    seed = a.get("seed", 0) if seed is None else seed
    res, preds = {}, {}
    for f in FRACTIONS:
        li = torch.as_tensor(stratified_subset(data["train_y"].cpu().numpy(), f, seed), device="cuda")
        pr, info, _ = linear_probe(ftr[li], data["train_y"][li], fte, C, stats, seed)
        res[f"linear_{f}"] = 100 * (pr == data["test_y"]).float().mean().item()
        pk = knn_predict(ftr[li], data["train_y"][li], fte, C, k=min(200, len(li) // C))
        res[f"knn_{f}"] = 100 * (pk == data["test_y"]).float().mean().item()
        if f == 0.1:
            preds["linear_0.1"] = pr.cpu().numpy().astype(np.int16)
            # how "used up" are these labels? fit of the probe on its own training images,
            # and class tightness of the labeled images vs. unseen test images
            w, b, mu, sd = _
            tr = ((((ftr[li] - mu) / sd) @ w + b).argmax(1) == data["train_y"][li]).float().mean().item()
            res["linear_0.1_train"] = 100 * tr
            g_lab, means_lab = geometry(ftr[li], data["train_y"][li], C)
            g_te, _m = geometry(fte, data["test_y"], C, ref_means=means_lab)
            res["cdnv_labeled_0.1"] = g_lab["cdnv"]
            res["cdnv_test_0.1"] = g_te["cdnv"]
            # the training images outside the labeled subset (for methods trained on all images these
            # were seen without a label; for methods trained on the labeled subset they were never seen)
            rest = torch.ones(ftr.shape[0], dtype=torch.bool, device="cuda")
            rest[li] = False
            res["cdnv_rest_0.1"] = geometry(ftr[rest], data["train_y"][rest], C, ref_means=means_lab)[0]["cdnv"]
            res["etf_dev_labeled_0.1"], res["etf_dev_test_0.1"] = g_lab["etf_dev"], g_te["etf_dev"]
            res["ncc_0.1"] = g_te["ncc_acc"]
    json.dump(res, open(out, "w"), indent=1)
    np.savez_compressed(os.path.join(d, "probes_preds.npz"), **preds)
    print(f"{name:36s} " + " ".join(f"{k} {v:.2f}" for k, v in res.items()), flush=True)
