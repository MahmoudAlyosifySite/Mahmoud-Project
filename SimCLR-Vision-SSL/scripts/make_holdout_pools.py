"""Labeled pools for the "held-out labels" arm (third revision).

A practitioner who wants to tune what comes after a label-using pretraining can hold out part of the labels
before pretraining. For each seed, the 5,000 labeled CIFAR-10 images of that seed are split exactly as
scripts/ft_search.py splits them for recipe selection (stratified 20%, split seed 777): the 4,000 training
images go into the pretraining loss, the 1,000 validation images are never used with their label during
pretraining. The pool file holds the 4,000 indices, so that

    dlt.train ... --pool_file results/pools/holdout_fit_f0.1_s<seed>.npy

pretrains on them and ft_search.py --seed <seed> later selects its recipe on exactly the held-out 1,000.

    python scripts/make_holdout_pools.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dlt.data import labeled_subset, load_dataset, stratified_subset   # noqa: E402

data = load_dataset("cifar10", os.environ.get("DLT_DATA", "./data"))
y = data["train_y"].numpy() if hasattr(data["train_y"], "numpy") else np.asarray(data["train_y"])
os.makedirs("results/pools", exist_ok=True)
for seed in range(5):
    labeled = labeled_subset(data, 0.1, seed)
    val_idx = labeled[stratified_subset(y[labeled], 0.2, 777)]
    fit_idx = np.setdiff1d(labeled, val_idx)
    assert len(fit_idx) == 4000 and len(val_idx) == 1000 and len(np.intersect1d(fit_idx, val_idx)) == 0
    assert np.all(np.bincount(y[fit_idx], minlength=10) == 400)
    np.save(f"results/pools/holdout_fit_f0.1_s{seed}.npy", fit_idx.astype(np.int64))
    print(seed, len(fit_idx), len(val_idx))
