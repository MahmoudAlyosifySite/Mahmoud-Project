"""Is the fine-tuning result a property of one recipe?  Hyper-parameter search for label-budget fine-tuning.

For one pretrained encoder, every configuration of the grid
    head init   {linear-probe, random}
    lr          {0.003, 0.01, 0.03, 0.1}
    epochs      {30, 60, 120}         (passes over the labeled images)
    regulariser {none, label smoothing 0.1, mixup alpha 0.2}
is trained on 80% of the labeled images and scored on the other 20% -- the test set and the
unlabeled labels are never used for selection. ``--grid ext`` widens the grid to lr up to 0.3 and up to
240 passes (120 recipes). The best configuration is then retrained on all
labeled images and evaluated on the test set (the test accuracy of every configuration is
also stored, for transparency, but is not used to choose).

    python scripts/ft_search.py --from_run simclr_all_e200_s0 --ckpt_dir <dir> --label_fraction 0.1 --seed 0

Every recipe result is cached in <out>/<name>/recipes/, so a search can be interrupted, can reuse the recipes
of an earlier search of the same encoder (``--reuse_from``), and can be shared by several processes started
with the same arguments (each claims recipes that no other live process is working on; the first to see the
grid complete runs the final step, the others wait for it).
"""
import argparse
import itertools
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dlt.data import IndexSampler, labeled_subset, load_dataset, stratified_subset   # noqa: E402
from dlt.evaluate import linear_probe                                  # noqa: E402
from dlt.gpu_aug import GPUAugment, sup_pipeline                       # noqa: E402
from dlt.models import Classifier, extract_features                    # noqa: E402

GRID = {"head": ["lp", "random"], "lr": [0.003, 0.01, 0.03, 0.1], "epochs": [30, 60, 120],
        "reg": ["none", "ls", "mixup"]}
GRID_EXT = {"head": ["lp", "random"], "lr": [0.003, 0.01, 0.03, 0.1, 0.3], "epochs": [30, 60, 120, 240],
            "reg": ["none", "ls", "mixup"]}
GRID_SMALL = {"head": ["lp", "random"], "lr": [0.01, 0.03, 0.1], "epochs": [30, 60], "reg": ["none", "mixup"]}
# the region of the widened grid where the label-using encoders had their best recipes (36 recipes)
GRID_FOCUS = {"head": ["lp", "random"], "lr": [0.03, 0.1, 0.3], "epochs": [60, 120, 240], "reg": ["ls", "mixup"]}
KEYS = ("head", "lr", "epochs", "reg")


def recipe_key(cfg):
    return f"{cfg['head']}_lr{cfg['lr']}_ep{cfg['epochs']}_{cfg['reg']}"


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def claim(path):
    """Create ``path`` exclusively (holding our pid); a lock left by a dead process is taken over."""
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                pid = int(open(path).read().strip() or 0)
            except (OSError, ValueError):
                pid = 0
            if pid and _alive(pid):
                return False
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
    return False


def finetune(enc_sd, backbone, data, fit_idx, eval_x, eval_y, cfg, seed, dev="cuda", batch=128):
    """Fine-tune on the images ``fit_idx`` and return accuracy on (eval_x, eval_y) and predictions."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    X, Y, C = data["train_x"], data["train_y"], data["num_classes"]
    model = Classifier(backbone, C).to(dev)
    model.encoder.load_state_dict(enc_sd)
    model = model.to(memory_format=torch.channels_last)
    aug = GPUAugment(sup_pipeline(X.shape[-1]), data["mean"], data["std"], dev, gen)
    fi = torch.as_tensor(fit_idx, device=dev)
    if cfg["head"] == "lp":
        ftr = extract_features(model.encoder, X[fi], aug.normalize)
        stats = (ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True) + 1e-6)
        _, _, (w, b, mu, sd) = linear_probe(ftr, Y[fi], ftr[:10], C, stats, seed)
        with torch.no_grad():
            model.fc.weight.copy_((w / sd.t()).t())
            model.fc.bias.copy_(b - ((mu / sd) @ w).squeeze(0))
    steps = int(round(cfg["epochs"] * len(fit_idx) / batch))
    warm = max(1, int(0.05 * steps))
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, nesterov=True, weight_decay=5e-4)
    scaler = torch.amp.GradScaler("cuda")
    sampler = IndexSampler(fit_idx, batch, gen, dev)
    smooth = 0.1 if cfg["reg"] == "ls" else 0.0
    model.train()
    for step in range(steps):
        lr = cfg["lr"] * (step + 1) / warm if step < warm else \
            0.5 * cfg["lr"] * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
        for gr in opt.param_groups:
            gr["lr"] = lr
        idx = sampler.next()
        x, y = aug(X[idx]), Y[idx]
        if cfg["reg"] == "mixup":
            lam = float(np.random.beta(0.2, 0.2))
            perm = torch.randperm(x.shape[0], device=dev, generator=gen)
            x = lam * x + (1 - lam) * x[perm]
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x.contiguous(memory_format=torch.channels_last)).float()
        if cfg["reg"] == "mixup":
            loss = lam * F.cross_entropy(logits, y) + (1 - lam) * F.cross_entropy(logits, y[perm])
        else:
            loss = F.cross_entropy(logits, y, label_smoothing=smooth)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        pred = torch.cat([model(aug.normalize(eval_x[i:i + 2000]).contiguous(memory_format=torch.channels_last)).argmax(1)
                          for i in range(0, eval_x.shape[0], 2000)])
    return 100 * (pred == eval_y).float().mean().item(), pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--from_run", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--out", default="results/runs")
    p.add_argument("--config_from", default=None,
                   help="name of a finished search: reuse its selected configuration instead of searching")
    p.add_argument("--data_dir", default=os.environ.get("DLT_DATA", "./data"))
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--label_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0, help="seed of the pretraining run (defines the labeled subset)")
    p.add_argument("--final_only", default=None, help="JSON of a configuration: skip the search and only run it")
    p.add_argument("--no_test_per_config", action="store_true")
    p.add_argument("--oracle", action="store_true",
                   help="score every recipe on the TEST set after training on all labeled images. This is not a "
                        "legitimate selection rule; it gives an upper bound on what recipe selection can achieve")
    p.add_argument("--grid", default="full", choices=["full", "small", "ext", "focus"],
                   help="small: 2 heads x lr {0.01, 0.03, 0.1} x {30, 60} epochs x {none, mixup}; "
                        "ext: the full grid widened to lr 0.3 and 240 passes (120 recipes)")
    p.add_argument("--reuse_from", default=None,
                   help="finished search of the same encoder, seed and mode whose recipe scores are reused")
    p.add_argument("--worker", type=int, default=0, help="only labels the process; workers share the cache")
    p.add_argument("--skip_default", action="store_true",
                   help="do not also fine-tune with the default recipe (when another run already measured it)")
    args = p.parse_args()
    out_dir = os.path.join(args.out, args.name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "result.json")
    if args.config_from:
        prev = json.load(open(os.path.join(args.out, args.config_from, "result.json")))
        args.final_only = json.dumps(prev["best"])
    if os.path.exists(out_path):
        print("[skip]", out_path)
        return
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    np.random.seed(args.seed)
    data = load_dataset(args.dataset, args.data_dir)
    for k in ("train_x", "train_y", "test_x", "test_y"):
        data[k] = data[k].to(dev)
    labeled = labeled_subset(data, args.label_fraction, args.seed)
    # split the LABELED images 80/20 (stratified); selection uses nothing else
    val = stratified_subset(data["train_y"].cpu().numpy()[labeled], 0.2, 777)
    val_idx = labeled[val]
    fit_idx = np.setdiff1d(labeled, val_idx)
    vi = torch.as_tensor(val_idx, device=dev)
    ck = torch.load(os.path.join(args.ckpt_dir, args.from_run, "model.pt"), map_location=dev)["model"]
    enc_sd = {k[len("encoder."):]: (v.float() if v.is_floating_point() else v) for k, v in ck.items() if k.startswith("encoder.")}

    rows = []
    t0 = time.time()
    if args.final_only:
        best = json.loads(args.final_only)
    else:
        grid = {"full": GRID, "small": GRID_SMALL, "ext": GRID_EXT, "focus": GRID_FOCUS}[args.grid]
        cache = os.path.join(out_dir, "recipes")
        os.makedirs(cache, exist_ok=True)
        if args.reuse_from:   # scores of an earlier search of the same encoder (same code, seed and split)
            prev = json.load(open(os.path.join(args.out, args.reuse_from, "result.json")))
            assert prev["from_run"] == args.from_run and bool(prev.get("oracle")) == bool(args.oracle), args.reuse_from
            for r in prev["grid"]:
                f = os.path.join(cache, recipe_key(r) + ".json")
                if not os.path.exists(f):
                    json.dump({**{k: r[k] for k in KEYS}, **{k: r[k] for k in ("val_acc", "test_acc") if k in r},
                               "reused_from": args.reuse_from}, open(f, "w"))
        cfgs = [dict(zip(KEYS, v)) for v in itertools.product(*(grid[k] for k in KEYS))]
        while True:
            for cfg in cfgs:
                f = os.path.join(cache, recipe_key(cfg) + ".json")
                if os.path.exists(f) or not claim(f[:-5] + ".lock"):
                    continue
                if os.path.exists(f):   # finished by another worker between the two checks
                    os.remove(f[:-5] + ".lock")
                    continue
                if args.oracle:
                    acc, _ = finetune(enc_sd, args.backbone, data, labeled, data["test_x"], data["test_y"], cfg, args.seed)
                    row = {**cfg, "test_acc": acc, "val_acc": acc}
                else:
                    acc, _ = finetune(enc_sd, args.backbone, data, fit_idx, data["train_x"][vi], data["train_y"][vi], cfg,
                                      args.seed)
                    row = {**cfg, "val_acc": acc}
                json.dump(row, open(f + ".tmp", "w"))
                os.replace(f + ".tmp", f)
                os.remove(f[:-5] + ".lock")
                print(f"{args.from_run} [w{args.worker}] {cfg} {'test' if args.oracle else 'val'} {acc:.2f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            missing = [c for c in cfgs if not os.path.exists(os.path.join(cache, recipe_key(c) + ".json"))]
            if not missing:
                break
            if not any(os.path.exists(os.path.join(cache, recipe_key(c) + ".lock")) for c in missing):
                continue                      # a worker died and left recipes undone: take them over
            time.sleep(20)                    # other workers are still training their recipes
        rows = [json.load(open(os.path.join(cache, recipe_key(c) + ".json"))) for c in cfgs]
        best = max(rows, key=lambda r: r["val_acc"])
        best = {k: best[k] for k in KEYS}
        # one process runs the final step; the others wait for its result (or take over if it died)
        while not claim(os.path.join(out_dir, "final.lock")):
            if os.path.exists(out_path):
                print("[done by another worker]", out_path)
                return
            time.sleep(20)
        if os.path.exists(out_path):
            os.remove(os.path.join(out_dir, "final.lock"))
            return
    test_acc, pred = finetune(enc_sd, args.backbone, data, labeled, data["test_x"], data["test_y"], best, args.seed)
    default = {"head": "lp", "lr": 0.01, "epochs": 60, "reg": "none"}
    default_acc = None if args.skip_default else \
        finetune(enc_sd, args.backbone, data, labeled, data["test_x"], data["test_y"], default, args.seed)[0]
    res = {"name": args.name, "args": vars(args), "from_run": args.from_run, "n_labeled": int(len(labeled)),
           "grid": rows, "best": best, "default": default, "train_time_s": time.time() - t0,
           "oracle": bool(args.oracle), "grid_name": args.grid, "n_recipes": len(rows),
           "metrics": {("finetune_oracle_acc" if args.oracle else "finetune_tuned_acc"): test_acc,
                       **({} if default_acc is None else {"finetune_default_acc": default_acc})}}
    np.savez_compressed(os.path.join(out_dir, "preds.npz"), finetune_tuned=pred.cpu().numpy().astype(np.int16))
    json.dump(res, open(out_path + ".tmp", "w"), indent=1)
    os.replace(out_path + ".tmp", out_path)
    if os.path.exists(os.path.join(out_dir, "final.lock")):
        os.remove(os.path.join(out_dir, "final.lock"))
    print(f"[done] {args.from_run}: best {best} -> test {test_acc:.2f} (default recipe {default_acc})", flush=True)


if __name__ == "__main__":
    main()
