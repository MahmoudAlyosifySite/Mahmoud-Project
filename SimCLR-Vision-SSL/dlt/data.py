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


def load_dataset(name, root):
    """Return a dict with uint8 image tensors [N,3,32,32] and int64 labels."""
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
