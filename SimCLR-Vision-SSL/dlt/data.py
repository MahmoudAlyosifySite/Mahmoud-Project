"""Dataset loading and label/pool subsampling.

CIFAR-10/100 are read from the original python pickles and returned as uint8
tensors of shape [N, 3, 32, 32]. Subsets are stratified (same number of images
per class) and *nested*: for a fixed seed, the 1% subset is contained in the 5%
subset, which is contained in the 10% subset, and so on.
"""

import os
import pickle
import tarfile
import urllib.request

import numpy as np
import torch

URLS = {
    "cifar10": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "cifar100": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
}
FOLDERS = {"cifar10": "cifar-10-batches-py", "cifar100": "cifar-100-python"}

# Channel statistics used for normalisation. The CIFAR-10 values are the ones
# used by the original code base of this project.
STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def _unpickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def _maybe_download(name, root):
    folder = os.path.join(root, FOLDERS[name])
    if os.path.isdir(folder):
        return folder
    os.makedirs(root, exist_ok=True)
    archive = os.path.join(root, os.path.basename(URLS[name]))
    if not os.path.exists(archive):
        print(f"[data] downloading {URLS[name]}")
        urllib.request.urlretrieve(URLS[name], archive)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root)
    return folder


def _load_stl10(root):
    """STL-10 (Coates et al., 2011): 5,000 labeled + 100,000 unlabeled training images and
    8,000 test images at 96 px. The training tensor holds the labeled images first; the
    unlabeled images follow with label -1. Built by scripts/prepare_stl10.py."""
    path = os.path.join(root, "stl10_uint8.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: run scripts/prepare_stl10.py first")
    d = np.load(path)
    n_lab = d["train_x"].shape[0]
    train_x = np.concatenate([d["train_x"], d["unlabeled_x"]])
    train_y = np.concatenate([d["train_y"], -np.ones(d["unlabeled_x"].shape[0], dtype=np.int64)])
    return {
        "name": "stl10",
        "train_x": torch.from_numpy(train_x),
        "train_y": torch.from_numpy(train_y).long(),
        "test_x": torch.from_numpy(d["test_x"]),
        "test_y": torch.from_numpy(d["test_y"]).long(),
        "num_classes": 10,
        "mean": tuple(float(v) for v in d["mean"]),
        "std": tuple(float(v) for v in d["std"]),
        "labeled_idx": np.arange(n_lab),
    }


def labeled_subset(data, fraction, seed):
    """Indices of the training images whose labels may be used.

    CIFAR: a nested class-balanced random subset. STL-10: the fixed labeled split
    (any positive fraction selects all of it)."""
    if "labeled_idx" in data:
        return data["labeled_idx"] if fraction > 0 else np.arange(0)
    n = data["train_y"].shape[0]
    if fraction >= 1.0:
        return np.arange(n)
    if fraction <= 0.0:
        return np.arange(0)
    return stratified_subset(data["train_y"].cpu().numpy(), fraction, seed)


def load_dataset(name, root):
    """Return a dict with uint8 image tensors [N,3,H,W] and int64 labels."""
    if name == "stl10":
        return _load_stl10(root)
    folder = _maybe_download(name, root)
    if name == "cifar10":
        xs, ys = [], []
        for i in range(1, 6):
            d = _unpickle(os.path.join(folder, f"data_batch_{i}"))
            xs.append(d[b"data"])
            ys.extend(d[b"labels"])
        train_x, train_y = np.concatenate(xs), np.array(ys)
        d = _unpickle(os.path.join(folder, "test_batch"))
        test_x, test_y = d[b"data"], np.array(d[b"labels"])
        num_classes = 10
    elif name == "cifar100":
        d = _unpickle(os.path.join(folder, "train"))
        train_x, train_y = d[b"data"], np.array(d[b"fine_labels"])
        d = _unpickle(os.path.join(folder, "test"))
        test_x, test_y = d[b"data"], np.array(d[b"fine_labels"])
        num_classes = 100
    else:
        raise ValueError(f"unknown dataset {name}")

    def to_tensor(x):
        return torch.from_numpy(np.ascontiguousarray(x.reshape(-1, 3, 32, 32)))

    mean, std = STATS[name]
    return {
        "name": name,
        "train_x": to_tensor(train_x),
        "train_y": torch.from_numpy(train_y).long(),
        "test_x": to_tensor(test_x),
        "test_y": torch.from_numpy(test_y).long(),
        "num_classes": num_classes,
        "mean": mean,
        "std": std,
    }


def stratified_subset(labels, fraction, seed):
    """Indices of a class-balanced random subset.

    For a fixed seed the subsets are nested across fractions, because every
    class keeps the first ``round(fraction * n_c)`` entries of one fixed random
    permutation of its images.
    """
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    out = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        perm = rng.permutation(idx)
        k = int(round(fraction * len(idx)))
        out.append(perm[:k])
    return np.sort(np.concatenate(out))


class IndexSampler:
    """Yields batches of indices by cycling through shuffled copies of a pool.

    If the pool is smaller than the batch, an image can appear more than once
    in a batch. The loss groups views by image index, so such repeats are
    treated as extra positives and never as negatives.
    """

    def __init__(self, pool, batch_size, generator, device):
        self.pool = torch.as_tensor(pool, dtype=torch.long, device=device)
        self.batch_size = batch_size
        self.g = generator
        self.device = device
        self.buf = torch.empty(0, dtype=torch.long, device=device)

    def next(self, n=None):
        n = self.batch_size if n is None else n
        while self.buf.numel() < n:
            perm = torch.randperm(self.pool.numel(), generator=self.g, device=self.device)
            self.buf = torch.cat([self.buf, self.pool[perm]])
        out, self.buf = self.buf[:n], self.buf[n:]
        return out
