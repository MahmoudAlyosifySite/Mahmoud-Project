"""Convert Tiny-ImageNet (Hugging Face parquet export, repo zh-plus/tiny-imagenet) into one uint8 archive.

    python scripts/prepare_tinyimagenet.py --src <dir with data/*.parquet> --out <data_dir>

Writes <data_dir>/tinyimagenet_uint8.npz with train_x [100000,3,64,64], train_y, test_x [10000,...] (the
official validation split, used as the test set), test_y, and the channel mean / std of the training images.
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
    train_x, train_y = read(glob.glob(os.path.join(a.src, "data", "train-*.parquet")))
    test_x, test_y = read(glob.glob(os.path.join(a.src, "data", "valid-*.parquet")))
    print(train_x.shape, test_x.shape, len(np.unique(train_y)), np.bincount(train_y).min(), np.bincount(train_y).max())
    mean = [float(train_x[:, c].mean() / 255) for c in range(3)]
    std = [float((train_x[:, c] / 255.0).std()) for c in range(3)]
    print("mean", mean, "std", std)
    np.savez(os.path.join(a.out, "tinyimagenet_uint8.npz"), train_x=train_x, train_y=train_y, test_x=test_x,
             test_y=test_y, mean=np.array(mean), std=np.array(std))


if __name__ == "__main__":
    main()
