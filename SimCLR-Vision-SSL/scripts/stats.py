"""Significance tests between two configurations.

For configurations A and B (each a set of runs that differ only in the seed) we report

* the difference of mean test accuracy with a 95% bootstrap confidence interval. The
  bootstrap is *paired* over test images (both configurations are scored on the same
  resampled images) and also resamples the seeds, so the interval covers both test-set
  noise and run-to-run noise;
* the two-sided bootstrap p-value for "no difference";
* Welch's t-test across seeds, as a second opinion that uses only run-to-run variance.

    python scripts/stats.py simclr_all_e100 supcon_lab_f0.1_e100 --key linear
"""
import argparse
import glob
import os
import re

import numpy as np
from scipy import stats as sps


def config_runs(config, root="results/runs"):
    runs = []
    for d in sorted(glob.glob(os.path.join(root, config + "_s*"))):
        if re.fullmatch(re.escape(config) + r"_s\d+", os.path.basename(d)) and os.path.exists(os.path.join(d, "preds.npz")):
            runs.append(d)
    return runs


def load_correct(config, key, y, root="results/runs"):
    """Boolean matrix [seeds, test images]: was the image classified correctly?"""
    out = []
    for d in config_runs(config, root):
        z = np.load(os.path.join(d, "preds.npz"))
        if key in z:
            out.append(z[key] == y)
    if not out:
        raise FileNotFoundError(f"no predictions '{key}' for {config}")
    return np.stack(out).astype(np.float32)


def paired_bootstrap(ca, cb, n_boot=10000, seed=0, chunk=500):
    """ca, cb: correctness matrices [seeds_a, n], [seeds_b, n] on the same n test images."""
    rng = np.random.default_rng(seed)
    n = ca.shape[1]
    diffs = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, chunk):
        m = min(chunk, n_boot - start)
        w = rng.multinomial(n, np.full(n, 1.0 / n), size=m).astype(np.float32) / n   # [m, n] image weights
        acc_a = w @ ca.T                                                             # [m, seeds_a]
        acc_b = w @ cb.T
        ia = rng.integers(0, ca.shape[0], size=(m, ca.shape[0]))
        ib = rng.integers(0, cb.shape[0], size=(m, cb.shape[0]))
        diffs[start:start + m] = (np.take_along_axis(acc_a, ia, 1).mean(1)
                                  - np.take_along_axis(acc_b, ib, 1).mean(1))
    obs = ca.mean() - cb.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": 100 * obs, "ci_lo": 100 * lo, "ci_hi": 100 * hi, "p_boot": max(p, 1.0 / n_boot)}


def compare(a, b, y, key="linear", root="results/runs", n_boot=10000):
    ca, cb = load_correct(a, key, y, root), load_correct(b, key, y, root)
    res = paired_bootstrap(ca, cb, n_boot)
    acc_a, acc_b = 100 * ca.mean(1), 100 * cb.mean(1)
    res.update({"a": a, "b": b, "key": key, "n_a": len(acc_a), "n_b": len(acc_b),
                "mean_a": acc_a.mean(), "std_a": acc_a.std(ddof=1) if len(acc_a) > 1 else float("nan"),
                "mean_b": acc_b.mean(), "std_b": acc_b.std(ddof=1) if len(acc_b) > 1 else float("nan")})
    if len(acc_a) > 1 and len(acc_b) > 1:
        t = sps.ttest_ind(acc_a, acc_b, equal_var=False)
        res["p_welch"] = float(t.pvalue)
    return res


def test_labels(dataset="cifar10", data_dir=None):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dlt.data import load_dataset
    return load_dataset(dataset, data_dir or os.environ.get("DLT_DATA", "./data"))["test_y"].numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--key", default="linear")
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--root", default="results/runs")
    args = p.parse_args()
    r = compare(args.a, args.b, test_labels(args.dataset), args.key, args.root)
    print(f"{r['a']}: {r['mean_a']:.2f} ± {r['std_a']:.2f} (n={r['n_a']})")
    print(f"{r['b']}: {r['mean_b']:.2f} ± {r['std_b']:.2f} (n={r['n_b']})")
    print(f"difference {r['diff']:+.2f} pp, 95% CI [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}], "
          f"bootstrap p = {r['p_boot']:.4g}" + (f", Welch p = {r['p_welch']:.3g}" if "p_welch" in r else ""))


if __name__ == "__main__":
    main()
