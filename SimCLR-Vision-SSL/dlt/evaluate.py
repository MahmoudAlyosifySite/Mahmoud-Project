"""Evaluation of frozen encoders: probes, kNN, representation geometry, colour reliance.

All functions work on cached features that live on the GPU.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .gpu_aug import rgb_to_gray
from .models import extract_features

L2_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


# ----------------------------------------------------------------------------
# classifiers on frozen features
# ----------------------------------------------------------------------------

def fit_logreg(x, y, num_classes, l2, max_iter=300, init=None):
    """Multinomial logistic regression (mean cross-entropy + l2/2 ||W||^2), full-batch L-BFGS."""
    d = x.shape[1]
    with torch.enable_grad():
        w = torch.zeros(d, num_classes, device=x.device, requires_grad=True)
        b = torch.zeros(num_classes, device=x.device, requires_grad=True)
        if init is not None:
            w.data.copy_(init[0])
            b.data.copy_(init[1])
        opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter, history_size=20,
                                tolerance_grad=1e-7, tolerance_change=1e-10,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(x @ w + b, y) + 0.5 * l2 * (w ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
    return w.detach(), b.detach()


def _val_split(y, num_classes, val_frac, seed):
    """Stratified split of indices into (fit, val)."""
    g = torch.Generator().manual_seed(seed)
    y_cpu = y.cpu()
    fit, val = [], []
    for c in range(num_classes):
        idx = (y_cpu == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(idx.numel(), generator=g)]
        k = max(1, int(round(val_frac * idx.numel()))) if idx.numel() > 1 else 0
        val.append(idx[:k])
        fit.append(idx[k:])
    return torch.cat(fit).to(y.device), torch.cat(val).to(y.device)


def linear_probe(train_f, train_y, test_f, num_classes, stats=None, seed=0, l2_grid=L2_GRID):
    """Linear evaluation on frozen features.

    Features are standardised, the L2 strength is chosen on a held-out 20% of
    the *training* labels, and the classifier is refit on all training labels.
    The test set is only used for the final prediction.

    Returns (test predictions, dict with chosen l2 and validation accuracy, (W, b, mean, std)).
    """
    if stats is None:
        mean, std = train_f.mean(0, keepdim=True), train_f.std(0, keepdim=True) + 1e-6
    else:
        mean, std = stats
    xtr = (train_f - mean) / std
    xte = (test_f - mean) / std
    fit_idx, val_idx = _val_split(train_y, num_classes, 0.2, seed)
    best = (-1.0, None)
    for l2 in l2_grid:
        w, b = fit_logreg(xtr[fit_idx], train_y[fit_idx], num_classes, l2)
        acc = ((xtr[val_idx] @ w + b).argmax(1) == train_y[val_idx]).float().mean().item()
        if acc > best[0]:
            best = (acc, l2)
    w, b = fit_logreg(xtr, train_y, num_classes, best[1])
    pred = (xte @ w + b).argmax(1)
    return pred, {"l2": best[1], "val_acc": 100 * best[0]}, (w, b, mean, std)


@torch.no_grad()
def knn_predict(train_f, train_y, test_f, num_classes, k=200, temperature=0.1):
    """Weighted k-nearest-neighbour classifier on cosine similarity (Wu et al. 2018)."""
    train_f = F.normalize(train_f, dim=1)
    test_f = F.normalize(test_f, dim=1)
    k = min(k, train_f.shape[0])
    preds = []
    for chunk in test_f.split(2000):
        sim, idx = (chunk @ train_f.t()).topk(k, dim=1)
        votes = torch.zeros(chunk.shape[0], num_classes, device=chunk.device)
        votes.scatter_add_(1, train_y[idx], (sim / temperature).exp())
        preds.append(votes.argmax(1))
    return torch.cat(preds)


# ----------------------------------------------------------------------------
# geometry: neural-collapse / ETF statistics
# ----------------------------------------------------------------------------

@torch.no_grad()
def class_means_vars(f, y, num_classes):
    f = f.double()
    means = torch.zeros(num_classes, f.shape[1], device=f.device, dtype=f.dtype)
    means.index_add_(0, y, f)
    counts = torch.bincount(y, minlength=num_classes).clamp(min=1).to(f.dtype)
    means = means / counts.view(-1, 1)
    sq = ((f - means[y]) ** 2).sum(1)
    var = torch.zeros(num_classes, device=f.device, dtype=f.dtype).index_add_(0, y, sq) / counts
    return means, var


@torch.no_grad()
def geometry(f, y, num_classes, ref_means=None):
    """Neural-collapse statistics of features ``f`` with labels ``y``.

    cdnv            class-distance normalised variance (Galanti et al. 2022); lower = tighter classes
    etf_cos_mean    mean cosine between centred class means (-1/(C-1) for a simplex ETF)
    etf_cos_std     std of those cosines (0 for an ETF)                       -> equiangularity
    etf_norm_cv     coefficient of variation of the class-mean norms (0 for an ETF) -> equinorm
    etf_dev         mean |cos + 1/(C-1)|; distance of the angles from the simplex ETF
    ncc_acc         accuracy of the nearest-class-centre rule, centres taken from ``ref_means``
                    (or from ``f`` itself if not given)
    """
    means, var = class_means_vars(f, y, num_classes)
    dist2 = torch.cdist(means, means) ** 2
    off = ~torch.eye(num_classes, dtype=torch.bool, device=f.device)
    cdnv = ((var.view(-1, 1) + var.view(1, -1)) / (2 * dist2.clamp(min=1e-12)))[off].mean().item()
    centred = means - means.mean(0, keepdim=True)
    norms = centred.norm(dim=1)
    cos = F.normalize(centred, dim=1) @ F.normalize(centred, dim=1).t()
    cos_off = cos[off]
    target = -1.0 / (num_classes - 1)
    centres = means if ref_means is None else ref_means
    pred = torch.cdist(f.double(), centres).argmin(1)
    return {
        "cdnv": cdnv,
        "etf_cos_mean": cos_off.mean().item(),
        "etf_cos_std": cos_off.std().item(),
        "etf_norm_cv": (norms.std() / norms.mean()).item(),
        "etf_dev": (cos_off - target).abs().mean().item(),
        "ncc_acc": 100 * (pred == y).float().mean().item(),
    }, means


@torch.no_grad()
def align_uniform(z1, z2):
    """Alignment and uniformity of L2-normalised embeddings (Wang & Isola 2020)."""
    z1, z2 = F.normalize(z1.float(), dim=1), F.normalize(z2.float(), dim=1)
    align = (z1 - z2).pow(2).sum(1).mean().item()
    sub = z1[:5000]
    d2 = torch.pdist(sub).pow(2)
    uniform = d2.mul(-2).exp().mean().log().item()
    return align, uniform


# ----------------------------------------------------------------------------
# colour reliance
# ----------------------------------------------------------------------------

@torch.no_grad()
def color_histogram(x_uint8, bins=8):
    """Joint RGB histogram (bins^3 values, square-rooted). Contains no spatial information."""
    q = (x_uint8.long() * bins) // 256
    code = (q[:, 0] * bins + q[:, 1]) * bins + q[:, 2]
    code = code.flatten(1)
    hist = torch.zeros(code.shape[0], bins ** 3, device=x_uint8.device)
    hist.scatter_add_(1, code, torch.ones_like(code, dtype=torch.float))
    return (hist / code.shape[1]).sqrt()


@torch.no_grad()
def linear_cka(a, b):
    """Linear centred kernel alignment between two feature matrices with the same rows."""
    a = a.double() - a.double().mean(0, keepdim=True)
    b = b.double() - b.double().mean(0, keepdim=True)
    hsic = (a.t() @ b).pow(2).sum()
    return (hsic / ((a.t() @ a).norm() * (b.t() @ b).norm())).item()


@torch.no_grad()
def ridge_r2(xtr, ytr, xte, yte, lam=1e-2):
    """R^2 of a ridge regression from features to targets (how decodable the targets are)."""
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True) + 1e-6
    xtr, xte = ((xtr - mu) / sd).double(), ((xte - mu) / sd).double()
    ymu = ytr.double().mean(0, keepdim=True)
    a = xtr.t() @ xtr + lam * xtr.shape[0] * torch.eye(xtr.shape[1], device=xtr.device, dtype=xtr.dtype)
    w = torch.linalg.solve(a, xtr.t() @ (ytr.double() - ymu))
    pred = xte @ w + ymu
    ss_res = (yte.double() - pred).pow(2).sum()
    ss_tot = (yte.double() - yte.double().mean(0, keepdim=True)).pow(2).sum()
    return (1 - ss_res / ss_tot).item()


def to_gray_uint8(x_uint8):
    g = rgb_to_gray(x_uint8.float() / 255.0).expand(-1, 3, -1, -1)
    return (g * 255.0).round().clamp(0, 255).to(torch.uint8)


# ----------------------------------------------------------------------------
# full evaluation of one encoder
# ----------------------------------------------------------------------------

@torch.no_grad()
def evaluate_encoder(encoder, data, normalize_fn, labeled_idx, pool_idx, seed=0,
                     head=None, aug=None, full=True):
    """Run every evaluation used in the paper on a frozen encoder.

    data         dict with train_x/train_y/test_x/test_y on the GPU
    labeled_idx  indices of training images whose labels the method was allowed to use
    pool_idx     indices of training images the encoder was pretrained on
    Returns (metrics dict, dict of prediction arrays).
    """
    C = data["num_classes"]
    xtr, ytr, xte, yte = data["train_x"], data["train_y"], data["test_x"], data["test_y"]
    ftr = extract_features(encoder, xtr, normalize_fn)
    fte = extract_features(encoder, xte, normalize_fn)
    stats = (ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True) + 1e-6)
    m, preds = {}, {}

    # 1. standard linear evaluation: the probe sees all training labels
    pred, info, (w, b, mu, sd) = linear_probe(ftr, ytr, fte, C, stats, seed)
    m["linear_acc"] = 100 * (pred == yte).float().mean().item()
    m["linear_l2"], m["linear_val_acc"] = info["l2"], info["val_acc"]
    preds["linear"] = pred

    # 2. weighted kNN with the full labeled training set as memory bank
    pred = knn_predict(ftr, ytr, fte, C)
    m["knn_acc"] = 100 * (pred == yte).float().mean().item()
    preds["knn"] = pred
    if not full:
        return m, preds

    # 3. label-limited linear probe: the probe sees only the labels the method was given
    if labeled_idx is not None and 0 < len(labeled_idx) < xtr.shape[0]:
        li = torch.as_tensor(labeled_idx, device=xtr.device)
        pred, info, _ = linear_probe(ftr[li], ytr[li], fte, C, stats, seed)
        m["linear_lab_acc"] = 100 * (pred == yte).float().mean().item()
        m["linear_lab_l2"] = info["l2"]
        preds["linear_lab"] = pred

    # 4. geometry on images seen during pretraining vs. the test set
    pi = torch.as_tensor(pool_idx, device=xtr.device)
    g_seen, means_seen = geometry(ftr[pi], ytr[pi], C)
    g_test, _ = geometry(fte, yte, C, ref_means=means_seen)
    m.update({f"seen_{k}": v for k, v in g_seen.items()})
    m.update({f"test_{k}": v for k, v in g_test.items()})
    if labeled_idx is not None and 0 < len(labeled_idx) < len(pool_idx):
        li = torch.as_tensor(labeled_idx, device=xtr.device)
        g_lab, _ = geometry(ftr[li], ytr[li], C)
        m.update({f"labeled_{k}": v for k, v in g_lab.items()})
    held = torch.ones(xtr.shape[0], dtype=torch.bool, device=xtr.device)
    held[pi] = False
    if held.sum() >= 10 * C:
        g_held, _ = geometry(ftr[held], ytr[held], C, ref_means=means_seen)
        m.update({f"heldout_{k}": v for k, v in g_held.items()})

    # 5. colour reliance
    f_gray = extract_features(encoder, to_gray_uint8(xte), normalize_fn)
    m["linear_acc_gray"] = 100 * ((((f_gray - mu) / sd) @ w + b).argmax(1) == yte).float().mean().item()
    f_perm = extract_features(encoder, xte[:, [2, 0, 1]].contiguous(), normalize_fn)
    m["linear_acc_chperm"] = 100 * ((((f_perm - mu) / sd) @ w + b).argmax(1) == yte).float().mean().item()
    hist_te = color_histogram(xte)
    hist_tr = color_histogram(xtr)
    onehot = F.one_hot(yte, C).float()
    m["cka_hist"] = linear_cka(fte, hist_te)
    m["cka_label"] = linear_cka(fte, onehot)
    m["hist_r2"] = ridge_r2(ftr, hist_tr, fte, hist_te)

    # 6. projection-space statistics
    if head is not None and aug is not None:
        sub = xte[:5000]
        v1 = aug(sub).contiguous(memory_format=torch.channels_last)
        v2 = aug(sub).contiguous(memory_format=torch.channels_last)
        was = (encoder.training, head.training)
        encoder.eval(), head.eval()
        with torch.autocast("cuda", dtype=torch.float16):
            z1, z2 = head(encoder(v1)), head(encoder(v2))
        encoder.train(was[0]), head.train(was[1])
        m["align"], m["uniform"] = align_uniform(z1, z2)
    return m, preds


def to_numpy_preds(preds):
    return {k: v.cpu().numpy().astype(np.int16) for k, v in preds.items()}
