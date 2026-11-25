"""Learnability check: can identity be read from the raw v1 tensor at all?

Closed-set (person, hand) classification on simulated people, v1 network vs
v0 (36 hand-made features + the v0 Gaussian model of SPEC 9.2) on the same
split. Per person, session and hand the plan records ``per_hand`` patterns
``k = 0..per_hand-1`` with grip instance ``k // 5``:

* train: sessions ``0..S-2``, ``k`` not in the test sets below;
* ``same_grip``: ``k in {4, 9}`` of the training sessions (grip seen in training);
* ``new_grip``: ``k in {10, 11}`` of the training sessions (same day, new grip);
* ``next_session``: every pattern of session ``S-1`` (a new day).

A large ``same_grip`` / ``next_session`` gap means the simulated day-to-day
variation hides the person; accuracy far above chance on ``next_session``
means identity is learnable across sessions.

Usage (from ``research/``)::

    python -m v1.learnability --data ~/7oda-work/v1/data/learn --people 200 --tag base
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import data as D
from . import models as M
from . import simdata

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SAME_GRIP = (4, 9)
NEW_GRIP = (10, 11)


def session_number(ts: D.TensorSet) -> np.ndarray:
    num = np.array([int(re.search(r"(\d+)$", s).group(1)) for s in ts.session_ids])
    return num[ts.session]


def pattern_index(ts: D.TensorSet, sess: np.ndarray) -> np.ndarray:
    """Running index k of each record inside its (person, session, hand) block (plan order)."""
    k = np.zeros(len(ts), np.int32)
    seen: dict = {}
    for i, key in enumerate(zip(ts.person.tolist(), sess.tolist(), ts.hand.tolist())):
        k[i] = seen.get(key, 0)
        seen[key] = k[i] + 1
    return k


def splits(ts: D.TensorSet):
    sess = session_number(ts)
    k = pattern_index(ts, sess)
    last = sess.max()
    test_same = (sess < last) & np.isin(k, SAME_GRIP)
    test_new = (sess < last) & np.isin(k, NEW_GRIP)
    train = (sess < last) & ~test_same & ~test_new
    return {"train": train, "same_grip": test_same, "new_grip": test_new, "next_session": sess == last}


def v0_gaussian(xtr, ytr, n_classes):
    """Vectorised SPEC 9.2 model: standardise, class means, Ledoit-Wolf pooled covariance."""
    from auditor_ref.model import ledoit_wolf

    mu = xtr.mean(0)
    sd = np.maximum(xtr.std(0), 1e-6)
    z = (xtr - mu) / sd
    means = np.stack([z[ytr == c].mean(0) for c in range(n_classes)])
    r = z - means[ytr]
    s_emp, mu_lw, lam = ledoit_wolf(r)
    lam = min(max(lam, 0.1), 1.0)                       # SPEC 9.2: shrinkageMin = 0.1
    s = (1 - lam) * s_emp + lam * mu_lw * np.eye(s_emp.shape[0])
    w = np.linalg.inv(np.linalg.cholesky(s))

    def dist(x):
        zz = ((x - mu) / sd) @ w.T
        mm = means @ w.T
        return (zz ** 2).sum(1)[:, None] - 2 * zz @ mm.T + (mm ** 2).sum(1)[None, :]
    return dist


def accuracy(pred_class, true_class):
    pc, tc = np.asarray(pred_class), np.asarray(true_class)
    return {"top1": float(np.mean(pc == tc)), "person": float(np.mean(pc // 2 == tc // 2)),
            "hand_via_class": float(np.mean(pc % 2 == tc % 2))}


def train_net(ts, masks, ident, device, epochs, width, log, seed=0, aug="full", margin=0.2, lr_max=2e-3):
    torch.manual_seed(seed)
    n_cls = int(ident.max()) + 1
    widths = tuple(int(width * f) for f in (1, 2, 3, 4))
    net = M.Teacher(widths=widths, dropout=0.1).to(device)
    head = M.ArcFace(M.EMB_DIM, n_cls, margin=margin, scale=30.0).to(device)
    params = list(net.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=2e-3, weight_decay=5e-4)
    tr = np.flatnonzero(masks["train"])
    x = torch.from_numpy(ts.x.astype(np.float16)).to(device)
    y = torch.from_numpy(ident.astype(np.int64)).to(device)
    hand = torch.from_numpy(ts.hand.astype(np.int64)).to(device)
    augmenter = D.Augmenter(D.AugmentConfig.named(aug), seed=seed, device=device)
    bs = 256
    steps_per_epoch = max(1, len(tr) // bs)
    total = epochs * steps_per_epoch
    tr_t = torch.from_numpy(tr).to(device)
    step = 0
    net.train()
    for ep in range(epochs):
        perm = tr_t[torch.randperm(len(tr_t), device=device)]
        for i in range(steps_per_epoch):
            idx = perm[i * bs:(i + 1) * bs]
            lr = lr_max * min(1.0, (step + 1) / 200) * 0.5 * (1 + math.cos(math.pi * step / total))
            for g in opt.param_groups:
                g["lr"] = lr
            xb = augmenter(x[idx].float())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(xb)
            loss_id, _ = head(out["embedding"], y[idx])
            loss = loss_id + 0.3 * F.cross_entropy(out["hand"].float(), hand[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            step += 1
        if ep % 10 == 9 or ep == epochs - 1:
            log(f"  epoch {ep + 1}/{epochs} loss {loss.item():.3f}")
    net.eval()
    embs, hands = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(ts), 2048):
            o = net(x[i:i + 2048].float())
            embs.append(o["embedding"].float())
            hands.append(o["hand"].float())
    emb = torch.cat(embs).cpu().numpy()
    hand_logits = torch.cat(hands).cpu().numpy()
    w = F.normalize(head.weight.detach().float(), dim=1).cpu().numpy()
    return emb, hand_logits, w, M.count_params(net)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--people", type=int, default=200)
    ap.add_argument("--sessions", type=int, default=5)
    ap.add_argument("--per-hand", type=int, default=12)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--phone-seed", type=int, default=3)
    ap.add_argument("--imu-mode", default="none")
    ap.add_argument("--person-seed-base", type=int, default=700000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--aug", default="full", choices=["full", "light", "none"])
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)
    if not glob.glob(os.path.join(args.data, "shard_*.npz")):
        opts = {"phone_seed": args.phone_seed, "imu_mode": args.imu_mode,
                "person_seed_base": args.person_seed_base}
        simdata.generate(args.data, "sim", "train", args.people, 11, args.workers, 10, "L", opts,
                         per_hand=args.per_hand, sessions=args.sessions, accessory_prob=0.0,
                         keep_raw=True, log=log)
    ts = D.TensorSet.concat([D.TensorSet.load(f) for f in sorted(glob.glob(os.path.join(args.data, "shard_*.npz")))])
    ok = ts.extra["quality_ok"] & np.isfinite(ts.extra["v0"]).all(1)
    ts = ts.subset(np.flatnonzero(ok))
    ident = ts.identity()
    masks = splits(ts)
    n_cls = int(ident.max()) + 1
    log(f"{len(ts)} records passing quality, {len(ts.person_ids)} people, {n_cls} classes, "
        + ", ".join(f"{k} {int(v.sum())}" for k, v in masks.items()))

    res = {"tag": args.tag, "args": vars(args), "people": len(ts.person_ids), "classes": n_cls, "chance_top1": 1.0 / n_cls,
           "chance_person": 2.0 / n_cls, "n": {k: int(v.sum()) for k, v in masks.items()}}
    # v0
    v0 = ts.extra["v0"]
    dist = v0_gaussian(v0[masks["train"]], ident[masks["train"]], n_cls)
    res["v0"] = {k: accuracy(dist(v0[m]).argmin(1), ident[m]) for k, m in masks.items()}
    log("v0 " + json.dumps({k: round(v["top1"], 3) for k, v in res["v0"].items()}))
    # v1 network
    emb, hand_logits, w, n_params = train_net(ts, masks, ident, args.device, args.epochs, args.width, log,
                                              aug=args.aug, margin=args.margin, lr_max=args.lr)
    tr = masks["train"]
    protos = np.stack([emb[tr & (ident == c)].mean(0) for c in range(n_cls)])
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    res["v1_params"] = n_params
    res["v1_classifier"] = {k: accuracy((emb[m] @ w.T).argmax(1), ident[m]) for k, m in masks.items()}
    res["v1_prototype"] = {k: accuracy((emb[m] @ protos.T).argmax(1), ident[m]) for k, m in masks.items()}
    res["v1_hand_head"] = {k: float(np.mean(hand_logits[m].argmax(1) == ts.hand[m])) for k, m in masks.items()}
    for name in ("v1_classifier", "v1_prototype"):
        log(name + " " + json.dumps({k: round(v["top1"], 3) for k, v in res[name].items()}))
    log("v1_hand_head " + json.dumps({k: round(v, 3) for k, v in res["v1_hand_head"].items()}))
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"v1_learnability_{args.tag}.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
