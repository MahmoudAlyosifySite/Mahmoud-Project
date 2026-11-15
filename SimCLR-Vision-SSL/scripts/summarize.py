"""Collect every results/runs/*/result.json into one table and print group means.

    python scripts/summarize.py                 # all runs, grouped by configuration
    python scripts/summarize.py --filter pilot  # only runs whose name contains "pilot"
    python scripts/summarize.py --csv results/all_runs.csv
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

KEYS = ["linear_acc", "knn_acc", "linear_lab_acc", "classifier_acc", "finetune_acc"]


def load_runs(root="results/runs"):
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "*", "result.json"))):
        r = json.load(open(path))
        a = r["args"]
        row = {
            "name": r["name"],
            "config": re.sub(r"_s\d+(?=(__|$))", "", r["name"]),
            "seed": a.get("seed"),
            "method": a.get("method", "finetune"),
            "dataset": a.get("dataset", "cifar10"),
            "backbone": a.get("backbone", "resnet18"),
            "aug": a.get("aug"),
            "pool": a.get("pool"),
            "fraction": a.get("label_fraction"),
            "epochs": a.get("epochs"),
            "n_pool": r.get("n_pool"),
            "n_labeled": r.get("n_labeled"),
            "steps": r.get("steps"),
            "train_min": r.get("train_time_s", 0) / 60,
        }
        row.update(r["metrics"])
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/runs")
    p.add_argument("--filter", default=None)
    p.add_argument("--csv", default=None)
    p.add_argument("--keys", default=",".join(KEYS))
    args = p.parse_args()
    df = load_runs(args.root)
    if df.empty:
        print("no runs found")
        return
    if args.filter:
        df = df[df["name"].str.contains(args.filter)]
    if args.csv:
        df.to_csv(args.csv, index=False)
    keys = [k for k in args.keys.split(",") if k in df.columns]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    out = []
    for cfg, g in df.groupby("config", sort=False):
        row = {"config": cfg, "n": len(g), "min": g["train_min"].mean()}
        for k in keys:
            v = g[k].dropna()
            if len(v):
                row[k] = f"{v.mean():.2f}" + (f" ± {v.std(ddof=1):.2f}" if len(v) > 1 else "")
        out.append(row)
    print(pd.DataFrame(out).fillna("").to_string(index=False))


if __name__ == "__main__":
    main()
