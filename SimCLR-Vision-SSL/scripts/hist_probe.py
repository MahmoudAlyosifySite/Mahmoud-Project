"""The trivial colour baseline: a linear probe on the joint RGB histogram of the raw images.

    python scripts/hist_probe.py      -> results/hist_probe.json
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dlt.data import load_dataset                 # noqa: E402
from dlt.evaluate import color_histogram, linear_probe   # noqa: E402

out = {}
for name in ("cifar10", "cifar100"):
    d = load_dataset(name, os.environ.get("DLT_DATA", "./data"))
    xtr, ytr = d["train_x"].cuda(), d["train_y"].cuda()
    xte, yte = d["test_x"].cuda(), d["test_y"].cuda()
    for bins in (4, 8):
        htr, hte = color_histogram(xtr, bins), color_histogram(xte, bins)
        pred, info, _ = linear_probe(htr, ytr, hte, d["num_classes"])
        acc = 100 * (pred == yte).float().mean().item()
        out[f"{name}_bins{bins}"] = {"acc": acc, "l2": info["l2"], "dim": bins ** 3}
        print(name, bins, f"{acc:.2f}")
json.dump(out, open("results/hist_probe.json", "w"), indent=1)
