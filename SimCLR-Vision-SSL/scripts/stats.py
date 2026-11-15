"""Statistical comparison of two configurations (sets of runs that differ only in the seed).

Hierarchical (two-level) bootstrap
    One bootstrap replicate (i) resamples the seeds of each configuration with replacement and
    (ii) resamples the test images with replacement, using the SAME resampled images for both
    configurations (paired at the image level). When the two configurations were run with the same
    seeds, the seeds are resampled jointly, because runs with the same seed share the labeled
    subset. The statistic is the difference of mean test accuracy. We report the percentile 95%
    interval and a two-sided p-value 2*min(P(d<=0), P(d>=0)), which cannot be smaller than
    1/n_boot and is reported as "< 0.001" below that.

Equivalence (TOST)
    For "no difference" claims we test H0: |d| >= margin with two one-sided tests on the same
    bootstrap distribution: p_tost = max(P(d <= -margin), P(d >= margin)). Equivalence within the
    margin is declared when p_tost < 0.05 (the 90% interval lies inside +-margin).

Multiple comparisons
    holm() applies the Holm-Bonferroni step-down correction to a family of p-values.

    python scripts/stats.py simclr_all_e100 supcon_lab_f0.1_e100 --key linear
"""
import argparse
import glob
import os
import re

import numpy as np
from scipy import stats as sps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def config_runs(config, root="results/runs"):
    runs = {}
    for d in sorted(glob.glob(os.path.join(root, glob.escape(config) + "_s*"))):
        m = re.fullmatch(re.escape(config) + r"_s(\d+)", os.path.basename(d))
        if m:
            runs[int(m.group(1))] = d
    return runs


def load_correct(config, key, y, root="results/runs", files=("preds.npz", "probes_preds.npz")):
    """Dict seed -> boolean vector over test images (was the image classified correctly?)."""
    out = {}
    for seed, d in config_runs(config, root).items():
        for fn in files:
            p = os.path.join(d, fn)
            if os.path.exists(p):
                z = np.load(p)
                if key in z:
                    out[seed] = (z[key] == y).astype(np.float32)
                    break
    return out


def hierarchical_bootstrap(ca, cb, n_boot=10000, seed=0, chunk=500, margin=1.0):
    """ca, cb: dicts seed -> correctness vector on the same n test images."""
    rng = np.random.default_rng(seed)
    sa, sb = sorted(ca), sorted(cb)
    A, B = np.stack([ca[s] for s in sa]), np.stack([cb[s] for s in sb])
    paired = sa == sb
    n = A.shape[1]
    diffs = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, chunk):
        m = min(chunk, n_boot - start)
        w = rng.multinomial(n, np.full(n, 1.0 / n), size=m).astype(np.float32) / n   # [m, n] image weights
        acc_a, acc_b = w @ A.T, w @ B.T                                              # [m, seeds]
        ia = rng.integers(0, A.shape[0], size=(m, A.shape[0]))
        ib = ia if paired else rng.integers(0, B.shape[0], size=(m, B.shape[0]))
        diffs[start:start + m] = (np.take_along_axis(acc_a, ia, 1).mean(1)
                                  - np.take_along_axis(acc_b, ib, 1).mean(1))
    diffs *= 100
    obs = 100 * (A.mean() - B.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    p_tost = max((diffs <= -margin).mean(), (diffs >= margin).mean())
    acc_a, acc_b = 100 * A.mean(1), 100 * B.mean(1)
    res = {"diff": obs, "ci_lo": lo, "ci_hi": hi, "p": float(min(1.0, max(p, 1.0 / n_boot))),
           "p_tost": float(max(p_tost, 1.0 / n_boot)), "margin": margin, "paired_seeds": paired,
           "n_a": len(sa), "n_b": len(sb), "mean_a": acc_a.mean(), "mean_b": acc_b.mean(),
           "std_a": acc_a.std(ddof=1) if len(sa) > 1 else float("nan"),
           "std_b": acc_b.std(ddof=1) if len(sb) > 1 else float("nan")}
    if len(sa) > 1 and len(sb) > 1:
        res["p_welch"] = float(sps.ttest_ind(acc_a, acc_b, equal_var=False).pvalue)
    return res


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def mean_ci(values, n_boot=10000, seed=0):
    """Mean and percentile 95% interval of a handful of per-seed numbers."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float(v.mean()), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def test_labels(dataset="cifar10", data_dir=None):
    """Test labels, cached in results/test_labels_<dataset>.npy."""
    cache = os.path.join(ROOT, "results", f"test_labels_{dataset}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    import sys
    sys.path.insert(0, ROOT)
    from dlt.data import load_dataset
    y = load_dataset(dataset, data_dir or os.environ.get("DLT_DATA", "./data"))["test_y"].numpy()
    np.save(cache, y)
    return y


def compare(a, b, y, key="linear", key_b=None, root="results/runs", n_boot=10000, margin=1.0):
    ca, cb = load_correct(a, key, y, root), load_correct(b, key_b or key, y, root)
    if not ca or not cb:
        raise FileNotFoundError(f"missing predictions for {a} [{key}] or {b} [{key_b or key}]")
    res = hierarchical_bootstrap(ca, cb, n_boot, margin=margin)
    res.update({"a": a, "b": b, "key": key})
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--key", default="linear")
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--root", default="results/runs")
    p.add_argument("--margin", type=float, default=1.0)
    args = p.parse_args()
    r = compare(args.a, args.b, test_labels(args.dataset), args.key, root=args.root, margin=args.margin)
    print(f"{r['a']}: {r['mean_a']:.2f} ± {r['std_a']:.2f} (n={r['n_a']})")
    print(f"{r['b']}: {r['mean_b']:.2f} ± {r['std_b']:.2f} (n={r['n_b']})")
    print(f"difference {r['diff']:+.2f} points, 95% CI [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}], p = {r['p']:.4g}, "
          f"TOST (±{r['margin']}) p = {r['p_tost']:.4g}" + (f", Welch p = {r['p_welch']:.3g}" if "p_welch" in r else ""))


if __name__ == "__main__":
    main()
