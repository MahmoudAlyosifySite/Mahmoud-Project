"""Convert STL-10 (Hugging Face parquet export, repo jxie/stl10) into one uint8 archive.

    python scripts/prepare_stl10.py --src <dir with data/*.parquet> --out <data_dir>

Writes <data_dir>/stl10_uint8.npz with train_x [5000,3,96,96], train_y, test_x [8000,...], test_y,
unlabeled_x [100000,...] and the channel mean / std of the training images (labeled + unlabeled).
"""
import argparse
import glob
import io
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def decode(b):
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8).transpose(2, 0, 1)


def read(files):
    xs, ys = [], []
    with ProcessPoolExecutor(32) as ex:
        for f in sorted(files):
            t = pq.read_table(f).to_pydict()
            xs.extend(ex.map(decode, [d["bytes"] for d in t["image"]], chunksize=256))
            ys.extend(t["label"])
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    g = lambda pat: glob.glob(os.path.join(a.src, "data", pat))   # noqa: E731
    train_x, train_y = read(g("train-*.parquet"))
    test_x, test_y = read(g("test-*.parquet"))
    unl_x, unl_y = read(g("unlabeled-*.parquet"))
    print(train_x.shape, test_x.shape, unl_x.shape, np.bincount(train_y), np.unique(unl_y))
    allx = np.concatenate([train_x, unl_x])
    mean = [float(allx[:, c].mean() / 255) for c in range(3)]
    std = [float((allx[:, c] / 255.0).std()) for c in range(3)]
    print("mean", mean, "std", std)
    np.savez(os.path.join(a.out, "stl10_uint8.npz"), train_x=train_x, train_y=train_y, test_x=test_x,
             test_y=test_y, unlabeled_x=unl_x, mean=np.array(mean), std=np.array(std))


if __name__ == "__main__":
    main()
