"""Is the fine-tuning result a property of one recipe?  Hyper-parameter search for label-budget fine-tuning.

For one pretrained encoder, every configuration of the grid
    head init   {linear-probe, random}
    lr          {0.003, 0.01, 0.03, 0.1}
    epochs      {30, 60, 120}         (passes over the labeled images)
    regulariser {none, label smoothing 0.1, mixup alpha 0.2}
is trained on 80% of the labeled images and scored on the other 20% -- the test set and the
unlabeled labels are never used for selection. The best configuration is then retrained on all
labeled images and evaluated on the test set (the test accuracy of every configuration is
also stored, for transparency, but is not used to choose).

    python scripts/ft_search.py --from_run simclr_all_e200_s0 --ckpt_dir <dir> --label_fraction 0.1 --seed 0
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
from dlt.data import IndexSampler, load_dataset, stratified_subset   # noqa: E402
from dlt.evaluate import linear_probe                                  # noqa: E402
from dlt.gpu_aug import GPUAugment                                     # noqa: E402
from dlt.models import Classifier, extract_features                    # noqa: E402

GRID = {"head": ["lp", "random"], "lr": [0.003, 0.01, 0.03, 0.1], "epochs": [30, 60, 120],
        "reg": ["none", "ls", "mixup"]}


def finetune(enc_sd, backbone, data, fit_idx, eval_x, eval_y, cfg, seed, dev="cuda", batch=128):
    """Fine-tune on the images ``fit_idx`` and return accuracy on (eval_x, eval_y) and predictions."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    X, Y, C = data["train_x"], data["train_y"], data["num_classes"]
    model = Classifier(backbone, C).to(dev)
    model.encoder.load_state_dict(enc_sd)
    model = model.to(memory_format=torch.channels_last)
    aug = GPUAugment("sup", data["mean"], data["std"], dev, gen)
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
    labeled = stratified_subset(data["train_y"].cpu().numpy(), args.label_fraction, args.seed)
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
        for head, lr, ep, reg in itertools.product(GRID["head"], GRID["lr"], GRID["epochs"], GRID["reg"]):
            cfg = {"head": head, "lr": lr, "epochs": ep, "reg": reg}
            val_acc, _ = finetune(enc_sd, args.backbone, data, fit_idx, data["train_x"][vi], data["train_y"][vi], cfg, args.seed)
            rows.append({**cfg, "val_acc": val_acc})
            print(f"{args.from_run} {cfg} val {val_acc:.2f} ({time.time() - t0:.0f}s)", flush=True)
        best = max(rows, key=lambda r: r["val_acc"])
        best = {k: best[k] for k in ("head", "lr", "epochs", "reg")}
    test_acc, pred = finetune(enc_sd, args.backbone, data, labeled, data["test_x"], data["test_y"], best, args.seed)
    default = {"head": "lp", "lr": 0.01, "epochs": 60, "reg": "none"}
    default_acc, _ = finetune(enc_sd, args.backbone, data, labeled, data["test_x"], data["test_y"], default, args.seed)
    res = {"name": args.name, "args": vars(args), "from_run": args.from_run, "n_labeled": int(len(labeled)),
           "grid": rows, "best": best, "default": default, "train_time_s": time.time() - t0,
           "metrics": {"finetune_tuned_acc": test_acc, "finetune_default_acc": default_acc}}
    np.savez_compressed(os.path.join(out_dir, "preds.npz"), finetune_tuned=pred.cpu().numpy().astype(np.int16))
    json.dump(res, open(out_path, "w"), indent=1)
    print(f"[done] {args.from_run}: best {best} -> test {test_acc:.2f} (default recipe {default_acc:.2f})", flush=True)


if __name__ == "__main__":
    main()
