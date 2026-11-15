"""Pretraining + evaluation of one run.

Methods
-------
simclr   NT-Xent on the pool, labels never used.
supcon   Grouped contrastive loss. Labeled images use their class as positives,
         unlabeled images (if the pool contains any) use only their own second view.
           --pool labeled  -> SupCon on the labeled subset only        ("label-only")
           --pool all      -> SupCon over all images, partially labeled ("union")
byol, simsiam   non-contrastive self-supervised baselines.
ce       supervised cross-entropy on the labeled images.

Budget
------
``--epochs`` is counted in passes over the *full* training set, i.e. the number
of gradient steps is epochs * (N_train // batch_size) no matter how small the
pool is. All methods therefore see the same number of augmented views.

Example
-------
python -m dlt.train --name supcon_L_f0.1_s0 --method supcon --pool labeled \
       --label_fraction 0.1 --seed 0
"""

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import IndexSampler, labeled_subset, load_dataset, stratified_subset
from .evaluate import evaluate_encoder, geometry, knn_predict, to_numpy_preds
from .gpu_aug import GPUAugment, sup_pipeline
from .losses import grouped_contrastive_loss, make_groups, negative_cosine
from .models import BYOLModel, Classifier, ContrastiveModel, SimSiamModel, extract_features

DEFAULTS = {
    # method: (lr at batch 512, weight decay, temperature)
    "simclr": (0.25, 1e-4, 0.5),     # official SupContrast recipe: lr 0.5 @ 1024, tau 0.5
    "supcon": (0.25, 1e-4, 0.1),     # official SupContrast recipe: lr 0.5 @ 1024, tau 0.1
    "simsiam": (0.06, 5e-4, None),   # Chen & He 2021, CIFAR: lr 0.03 * B/256
    "byol": (0.24, 5e-4, None),      # lightly CIFAR-10 benchmark: lr 0.06 * B/128
    "ce": (0.2, 5e-4, None),         # lr 0.1 * B/256
}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--out", default="results/runs")
    p.add_argument("--ckpt_dir", default=None, help="where to store the final encoder (default: <out>/<name>)")
    p.add_argument("--data_dir", default=os.environ.get("DLT_DATA", "./data"))
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "stl10", "tinyimagenet"])
    p.add_argument("--method", default="simclr", choices=list(DEFAULTS))
    p.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50", "resnet18_mp", "resnet50_mp"])
    p.add_argument("--aug", default="simclr")
    p.add_argument("--pool", default="all", choices=["all", "labeled"])
    p.add_argument("--label_fraction", type=float, default=0.0)
    p.add_argument("--pool_file", default=None, help=".npy with the indices of the labeled images")
    p.add_argument("--labeled_share", type=float, default=None,
                   help="supcon with --pool all: fraction of every batch drawn from the labeled images")
    p.add_argument("--epochs", type=float, default=100.0)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--wd", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--temperature_unlabeled", type=float, default=None,
                   help="supcon with --pool all: temperature of anchors without a label "
                        "(default: same as --temperature)")
    p.add_argument("--holdout", type=int, default=0,
                   help="development mode: hold out this many training images as a validation "
                        "set and report on them instead of the test set")
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split_seed", type=int, default=None)
    p.add_argument("--curve_points", type=int, default=8)
    p.add_argument("--compile", type=int, default=1)
    p.add_argument("--init_run", default=None,
                   help="initialise encoder and head from the checkpoint of this run (two-stage pipelines)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no_save_ckpt", action="store_true")
    args = p.parse_args(argv)
    lr, wd, temp = DEFAULTS[args.method]
    if args.lr is None:
        args.lr = lr * args.batch_size / 512
    if args.wd is None:
        args.wd = wd
    if args.temperature is None:
        args.temperature = temp
    if args.split_seed is None:
        args.split_seed = args.seed
    return args


def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    return 0.5 * base * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def main(argv=None):
    args = parse_args(argv)
    out_dir = os.path.join(args.out, args.name)
    res_path = os.path.join(out_dir, "result.json")
    if os.path.exists(res_path) and not args.force:
        print(f"[skip] {res_path} exists")
        return
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.ckpt_dir, args.name) if args.ckpt_dir else out_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    data = load_dataset(args.dataset, args.data_dir)
    for k in ("train_x", "train_y", "test_x", "test_y"):
        data[k] = data[k].to(dev)
    if args.holdout > 0:
        # validation protocol used for all design decisions: the test set is not touched
        val_idx = stratified_subset(data["train_y"].cpu().numpy(), args.holdout / data["train_x"].shape[0], 12345)
        keep = np.setdiff1d(np.arange(data["train_x"].shape[0]), val_idx)
        vi, ki = torch.as_tensor(val_idx, device=dev), torch.as_tensor(keep, device=dev)
        data["test_x"], data["test_y"] = data["train_x"][vi], data["train_y"][vi]
        data["train_x"], data["train_y"] = data["train_x"][ki], data["train_y"][ki]
    X, Y, C = data["train_x"], data["train_y"], data["num_classes"]
    N = X.shape[0]

    # ---- which labels may be used, and which images are in the pretraining pool
    if args.pool_file:
        labeled_idx = np.sort(np.load(args.pool_file))
    else:
        labeled_idx = labeled_subset(data, args.label_fraction, args.split_seed)
    labeled_mask = torch.zeros(N, dtype=torch.bool, device=dev)
    labeled_mask[torch.as_tensor(labeled_idx, device=dev, dtype=torch.long)] = True
    pool_idx = np.arange(N) if args.pool == "all" else labeled_idx
    assert len(pool_idx) > 0, "empty pretraining pool"
    uses_labels = args.method in ("supcon", "ce")
    if args.method == "ce":
        assert args.pool == "labeled" or len(labeled_idx) == N, "ce needs --pool labeled"
    # Images with a ground-truth label, used by every evaluation. On STL-10 only the labeled split has
    # labels, so probes, kNN and geometry use those 5,000 images and the test set.
    has_unl = "labeled_idx" in data
    if has_unl:
        assert args.holdout == 0
        ev = torch.as_tensor(data["labeled_idx"], device=dev)
        Xe, Ye = X[ev], Y[ev]
        eval_data = dict(data, train_x=Xe, train_y=Ye)
        eval_pool, eval_lab = np.arange(Xe.shape[0]), None
    else:
        Xe, Ye, eval_data, eval_pool, eval_lab = X, Y, data, pool_idx, labeled_idx

    B = args.batch_size
    total_steps = int(round(args.epochs * (N // B)))
    warmup = max(1, int(args.warmup_frac * total_steps))
    print(f"[run] {args.name}: {args.method} | pool {len(pool_idx)} | labeled {len(labeled_idx)} | "
          f"{total_steps} steps | lr {args.lr} wd {args.wd} tau {args.temperature}", flush=True)

    # ---- samplers
    if args.method == "supcon" and args.pool == "all" and args.labeled_share and 0 < len(labeled_idx) < N:
        n_lab = int(round(args.labeled_share * B))
        unl_idx = np.setdiff1d(np.arange(N), labeled_idx)
        samplers = [(IndexSampler(labeled_idx, n_lab, gen, dev), n_lab),
                    (IndexSampler(unl_idx, B - n_lab, gen, dev), B - n_lab)]
    else:
        samplers = [(IndexSampler(pool_idx, B, gen, dev), B)]

    def next_batch():
        return torch.cat([s.next(n) for s, n in samplers])

    # ---- model
    aug = GPUAugment(sup_pipeline(X.shape[-1]) if args.method == "ce" else args.aug, data["mean"], data["std"], dev, gen)
    if args.method in ("simclr", "supcon"):
        model = ContrastiveModel(args.backbone)
    elif args.method == "simsiam":
        model = SimSiamModel(args.backbone)
    elif args.method == "byol":
        model = BYOLModel(args.backbone)
    else:
        model = Classifier(args.backbone, C)
    if args.init_run:
        src = torch.load(os.path.join(args.ckpt_dir or args.out, args.init_run, "model.pt"), map_location="cpu")["model"]
        own = model.state_dict()
        keep = {k: v.float() if v.is_floating_point() else v for k, v in src.items()
                if k in own and own[k].shape == v.shape}
        missing = [k for k in own if k not in keep]
        model.load_state_dict(keep, strict=False)
        print(f"[init] {args.init_run}: loaded {len(keep)} tensors, {len(missing)} left at initialisation", flush=True)
    model = model.to(dev).to(memory_format=torch.channels_last)
    target = None
    if args.method == "byol":
        target = copy.deepcopy(model)
        for q in target.parameters():
            q.requires_grad_(False)
    net = torch.compile(model) if args.compile else model

    if args.method == "simsiam":   # the predictor keeps a constant learning rate
        groups = [{"params": [q for n, q in model.named_parameters() if not n.startswith("predictor")], "decay": True},
                  {"params": list(model.predictor.parameters()), "decay": False}]
    else:
        groups = [{"params": list(model.parameters()), "decay": True}]
    opt = torch.optim.SGD(groups, lr=args.lr, momentum=0.9, weight_decay=args.wd,
                          nesterov=(args.method == "ce"))
    scaler = torch.amp.GradScaler("cuda")

    curve_at = sorted({int(round(total_steps * (i + 1) / args.curve_points)) for i in range(args.curve_points)})
    curve, loss_log, run_loss, run_n = [], [], 0.0, 0
    geom_curve = []
    pool_t = torch.as_tensor(eval_pool, device=dev)
    lab_t = torch.as_tensor(np.arange(0) if has_unl else labeled_idx, device=dev, dtype=torch.long)

    def geom_point(at, ftr, fte):
        """Class tightness (CDNV) on the pretraining pool, on the labeled images and on the test set."""
        out = {"step": at, "pool_cdnv": geometry(ftr[pool_t], Ye[pool_t], C)[0]["cdnv"],
               "test_cdnv": geometry(fte, data["test_y"], C)[0]["cdnv"]}
        if 0 < lab_t.numel() < N:
            out["labeled_cdnv"] = geometry(ftr[lab_t], Ye[lab_t], C)[0]["cdnv"]
        return out
    t0 = time.time()
    model.train()
    for step in range(total_steps):
        lr = lr_at(step, total_steps, args.lr, warmup)
        for gr in opt.param_groups:
            gr["lr"] = lr if gr["decay"] else args.lr
        idx = next_batch()
        xb = X[idx]
        # forward passes run in mixed precision; every loss is computed in float32
        n = idx.shape[0]
        if args.method == "ce":
            with torch.autocast("cuda", dtype=torch.float16):
                logits = net(aug(xb).contiguous(memory_format=torch.channels_last))
            loss = F.cross_entropy(logits.float(), Y[idx])
        else:
            x = aug(torch.cat([xb, xb])).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16):
                out = net(x)
            if args.method in ("simclr", "supcon"):
                z = F.normalize(out[1].float(), dim=1)
                lab = labeled_mask[idx] if uses_labels else torch.zeros_like(idx, dtype=torch.bool)
                temp = args.temperature
                if args.temperature_unlabeled is not None:
                    temp = torch.where(lab, args.temperature, args.temperature_unlabeled)
                loss = grouped_contrastive_loss(z, make_groups(idx, Y[idx], lab, C), temp)
            elif args.method == "simsiam":
                _, z, pr = out
                loss = 0.5 * (negative_cosine(pr[:n], z[n:]) + negative_cosine(pr[n:], z[:n]))
            else:  # byol
                _, _, pr = out
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    _, zt, _ = target(x)
                loss = negative_cosine(pr[:n], zt[n:]) + negative_cosine(pr[n:], zt[:n])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if target is not None:
            m = 1 - (1 - 0.99) * (math.cos(math.pi * step / total_steps) + 1) / 2
            with torch.no_grad():
                for qt, qo in zip(target.parameters(), model.parameters()):
                    qt.mul_(m).add_(qo.detach(), alpha=1 - m)
                for bt, bo in zip(target.buffers(), model.buffers()):
                    bt.copy_(bo)

        run_loss += loss.detach()
        run_n += 1
        if (step + 1) % 100 == 0 or step + 1 == total_steps:
            avg = (run_loss / run_n).item()
            if not math.isfinite(avg):
                raise RuntimeError(f"non-finite loss at step {step + 1}")
            loss_log.append([step + 1, avg])
            run_loss, run_n = 0.0, 0
        if (step + 1) in curve_at and step + 1 < total_steps:
            ftr = extract_features(model.encoder, Xe, aug.normalize)
            fte = extract_features(model.encoder, data["test_x"], aug.normalize)
            acc = 100 * (knn_predict(ftr, Ye, fte, C) == data["test_y"]).float().mean().item()
            curve.append([step + 1, acc])
            geom_curve.append(geom_point(step + 1, ftr, fte))
            print(f"  step {step + 1:6d}/{total_steps}  loss {loss.item():.4f}  knn {acc:.2f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    torch.cuda.synchronize()
    train_time = time.time() - t0

    # ---- evaluation
    model.eval()
    t1 = time.time()
    head = getattr(model, "head", None) if args.method in ("simclr", "supcon") else None
    metrics, preds = evaluate_encoder(model.encoder, eval_data, aug.normalize, eval_lab, eval_pool,
                                      seed=args.seed, head=head, aug=aug)
    curve.append([total_steps, metrics["knn_acc"]])
    if args.method == "ce":
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            pr = torch.cat([model(aug.normalize(data["test_x"][i:i + 2000]).contiguous(memory_format=torch.channels_last)).argmax(1)
                            for i in range(0, data["test_x"].shape[0], 2000)])
        metrics["classifier_acc"] = 100 * (pr == data["test_y"]).float().mean().item()
        preds["classifier"] = pr

    result = {
        "name": args.name, "args": vars(args), "n_pool": int(len(pool_idx)), "n_labeled": int(len(labeled_idx)),
        "steps": total_steps, "train_time_s": train_time, "eval_time_s": time.time() - t1,
        "final_loss": loss_log[-1][1], "loss_log": loss_log, "knn_curve": curve, "geom_curve": geom_curve,
        "metrics": metrics,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }
    np.savez_compressed(os.path.join(out_dir, "preds.npz"), **to_numpy_preds(preds),
                        labeled_idx=np.asarray(labeled_idx, dtype=np.int32))
    if not args.no_save_ckpt:
        sd = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
        torch.save({"model": sd, "args": vars(args)}, os.path.join(ckpt_dir, "model.pt"))
    with open(res_path, "w") as f:
        json.dump(result, f, indent=1)
    keys = ("linear_acc", "knn_acc", "linear_lab_acc", "classifier_acc")
    print(f"[done] {args.name}: " + "  ".join(f"{k} {metrics[k]:.2f}" for k in keys if k in metrics)
          + f"  | train {train_time / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
