"""Shared training utilities: run directories, data loading, schedules,
embedding extraction and quick validation metrics."""
from __future__ import annotations

import glob
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from . import data as D
from . import models as M

# checkpoints and logs: persistent disk; simulated shards: scratch (regenerable from seeds)
WORK = os.environ.get("V1_WORK", os.path.expanduser("~/7oda-work/v1"))
DATA = os.environ.get("V1_DATA", "/tmp/claude-1000/7oda-v1/data")
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def run_dir(run_id: str) -> str:
    d = os.path.join(WORK, "runs", run_id)
    os.makedirs(d, exist_ok=True)
    return d


class Logger:
    def __init__(self, directory: str):
        self.dir = directory
        self.t0 = time.time()
        self.fh = open(os.path.join(directory, "log.txt"), "a", buffering=1)
        self.mf = open(os.path.join(directory, "metrics.jsonl"), "a", buffering=1)

    def log(self, msg: str):
        line = f"[{time.time() - self.t0:8.1f}s] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")

    def metrics(self, **kw):
        kw["time_s"] = round(time.time() - self.t0, 1)
        self.mf.write(json.dumps(kw) + "\n")


def load_sets(patterns, max_samples: int | None = None, device=None) -> D.TensorSet:
    """Concatenate TensorSet shards matching glob patterns (comma-separated).

    With ``device``, each shard's ``x`` goes to that device (float16) as soon
    as it is read, so host memory holds one shard at a time; the returned set
    then has a torch ``x``.
    """
    if isinstance(patterns, str):
        patterns = [p for p in patterns.split(",") if p]
    files = sorted(f for p in patterns for f in glob.glob(p))
    if not files:
        raise FileNotFoundError(f"no shards match {patterns}")
    sets, xs, n = [], [], 0
    for f in files:
        s = D.TensorSet.load(f)
        if device is not None:
            xs.append(torch.from_numpy(np.ascontiguousarray(s.x, dtype=np.float16)).to(device))
            s.x = np.empty((len(xs[-1]), 0, 0), np.float16)
        sets.append(s)
        n += len(s)
        if max_samples and n >= max_samples:
            break
    ts = D.TensorSet.concat(sets)
    if device is not None:
        ts.x = torch.cat(xs) if len(xs) > 1 else xs[0]
        del xs
    if max_samples and len(ts) > max_samples:
        ts = ts.subset(np.arange(max_samples))
    return ts


def cosine_lr(step: int, total: int, warmup: int, base: float, final_frac: float = 0.02) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))


def set_lr(opt, lr: float):
    for g in opt.param_groups:
        g["lr"] = lr * g.get("lr_mult", 1.0)


@torch.no_grad()
def embed(model, x, device="cuda", batch: int = 4096, bf16: bool = True):
    """Embeddings and hand logits for an array/tensor ``[M, 10, 256]``."""
    model.eval()
    embs, hands = [], []
    for i in range(0, x.shape[0], batch):
        xb = x[i:i + batch]
        xb = torch.as_tensor(np.asarray(xb) if not isinstance(xb, torch.Tensor) else xb)
        xb = xb.to(device).float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and str(device).startswith("cuda")):
            out = model(xb)
        embs.append(out["embedding"].float().cpu())
        hands.append(out["hand"].float().cpu())
    return torch.cat(embs).numpy(), torch.cat(hands).numpy()


def eer(pos: np.ndarray, neg: np.ndarray) -> float:
    """Equal error rate for similarity scores (higher = same)."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    frr = 1 - tp / pos.size
    far = fp / neg.size
    i = int(np.argmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2)


def quick_validate(model, vs: D.TensorSet, device="cuda", max_ids: int = 400) -> dict:
    """Cross-session validation on held-out people.

    Prototypes per (person, hand) from each person's first session, tested on
    the person's later sessions: closed-set top-1 over all validation
    identities, hand-head accuracy, and verification EER of prototype scores.
    """
    emb, hand = embed(model, vs.x, device)
    ident = vs.person.astype(np.int64) * 2 + vs.hand.astype(np.int64)
    one = vs.hand < 2
    first = np.zeros(len(vs), dtype=bool)
    # first session of each person (lowest session index seen for that person)
    first_sess = {}
    for p, s in zip(vs.person, vs.session):
        first_sess[p] = min(first_sess.get(p, s), s)
    first = np.array([s == first_sess[p] for p, s in zip(vs.person, vs.session)]) & one
    test = (~first) & one
    ids = np.unique(ident[first])[:max_ids]
    keep_e = first & np.isin(ident, ids)
    keep_t = test & np.isin(ident, ids)
    out = {}
    pred_hand = hand.argmax(1)
    out["hand_acc"] = float(np.mean(pred_hand[vs.hand < 3] == vs.hand[vs.hand < 3]))
    if keep_t.sum() == 0 or len(ids) < 2:
        return out
    protos = np.stack([emb[keep_e & (ident == i)].mean(0) for i in ids])
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    s = emb[keep_t] @ protos.T
    true = np.searchsorted(ids, ident[keep_t])
    pred = s.argmax(1)
    out["top1_id"] = float(np.mean(pred == true))
    out["top1_person"] = float(np.mean(ids[pred] // 2 == ident[keep_t] // 2))
    pos = s[np.arange(len(true)), true]
    mask = np.ones_like(s, dtype=bool)
    mask[np.arange(len(true)), true] = False
    neg = s[mask]
    rng = np.random.default_rng(0)
    if neg.size > 200000:
        neg = rng.choice(neg, 200000, replace=False)
    out["eer"] = eer(pos, neg)
    out["n_ids"] = int(len(ids))
    return out


def save_ckpt(path: str, model, kind: str, model_kw: dict, extra: dict | None = None):
    torch.save({"kind": kind, "model_kw": model_kw, "state": model.state_dict(), "extra": extra or {}}, path)


def load_model(path: str, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck["kind"] == "student_folded":          # BatchNorm-folded, palettized export copy
        from .export_coreml import fold_batchnorm
        model = fold_batchnorm(M.Student(**ck.get("model_kw", {})))
    else:
        model = M.build_model(ck["kind"], **ck.get("model_kw", {}))
    model.load_state_dict(ck["state"])
    return model.to(device), ck


def make_param_groups(model, weight_decay: float, extra_modules=()):
    decay, no_decay = [], []
    mods = [model, *extra_modules]
    for m in mods:
        for n, p in m.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def write_results(run_id: str, directory: str, summary: dict | None = None):
    """Copy a run's validation curve and summary into research/results (small files),
    so results survive a lost scratch disk."""
    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    mpath = os.path.join(directory, "metrics.jsonl")
    if os.path.exists(mpath):
        with open(mpath) as fh:
            for line in fh:
                r = json.loads(line)
                flat = {k: v for k, v in r.items() if not isinstance(v, dict)}
                for key in ("val", "held"):
                    if isinstance(r.get(key), dict):
                        flat.update({f"{key}_{k}": v for k, v in r[key].items()})
                rows.append(flat)
    if rows:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "step", k))
        with open(os.path.join(RESULTS, f"v1_{run_id}_curve.csv"), "w") as fh:
            fh.write(",".join(keys) + "\n")
            for r in rows:
                fh.write(",".join("" if r.get(k) is None else (f"{r[k]:.5g}" if isinstance(r[k], float) else str(r[k]))
                                  for k in keys) + "\n")
    if summary is not None:
        with open(os.path.join(RESULTS, f"v1_{run_id}_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=1, default=str)


def save_state(path: str, **objs):
    """Resumable training state (modules / optimizers via state_dict, plus plain values)."""
    state = {k: (v.state_dict() if hasattr(v, "state_dict") else v) for k, v in objs.items()}
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_state(path: str, **objs):
    """Load what ``save_state`` wrote into the given modules / optimizers; returns the dict."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    for k, v in objs.items():
        if k in state and hasattr(v, "load_state_dict"):
            v.load_state_dict(state[k])
    return state


def gpu_note() -> str:
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"nvidia-smi failed: {e}"
