"""Semi-supervised fine-tuning of a pretrained encoder with the labeled subset only.

Protocol (LP-FT, Kumar et al. 2022): the classifier is initialised with the
label-limited linear probe and the whole network is then fine-tuned on the
labeled images with the standard supervised augmentation (flip + padded crop).

Example
-------
python -m dlt.finetune --name simclr_all_s0__ft --from_run simclr_all_s0 --label_fraction 0.1 --seed 0
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import IndexSampler, load_dataset, stratified_subset
from .evaluate import linear_probe
from .gpu_aug import GPUAugment
from .models import Classifier, extract_features


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--from_run", required=True)
    p.add_argument("--out", default="results/runs")
    p.add_argument("--ckpt_dir", default=None)
    p.add_argument("--data_dir", default=os.environ.get("DLT_DATA", "./data"))
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--label_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0, help="must equal the seed of the pretraining run (same split)")
    p.add_argument("--epochs", type=float, default=60.0, help="passes over the labeled subset")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--holdout", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    out_dir = os.path.join(args.out, args.name)
    res_path = os.path.join(out_dir, "result.json")
    if os.path.exists(res_path) and not args.force:
        print(f"[skip] {res_path} exists")
        return
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    data = load_dataset(args.dataset, args.data_dir)
    if args.holdout > 0:
        val_idx = stratified_subset(data["train_y"].numpy(), args.holdout / data["train_x"].shape[0], 12345)
        keep = np.setdiff1d(np.arange(data["train_x"].shape[0]), val_idx)
        data["test_x"], data["test_y"] = data["train_x"][val_idx], data["train_y"][val_idx]
        data["train_x"], data["train_y"] = data["train_x"][keep], data["train_y"][keep]
    for k in ("train_x", "train_y", "test_x", "test_y"):
        data[k] = data[k].to(dev)
    X, Y, C = data["train_x"], data["train_y"], data["num_classes"]
    labeled_idx = stratified_subset(Y.cpu().numpy(), args.label_fraction, args.seed)
    li = torch.as_tensor(labeled_idx, device=dev)

    ckpt_root = args.ckpt_dir or args.out
    ckpt = torch.load(os.path.join(ckpt_root, args.from_run, "model.pt"), map_location=dev)
    model = Classifier(args.backbone, C).to(dev)
    enc_sd = {k[len("encoder."):]: v.float() if v.is_floating_point() else v
              for k, v in ckpt["model"].items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(enc_sd)
    model = model.to(memory_format=torch.channels_last)
    aug = GPUAugment("sup", data["mean"], data["std"], dev, gen)

    # linear-probe initialisation of the classifier (labels of the subset only)
    ftr = extract_features(model.encoder, X, aug.normalize)
    fte = extract_features(model.encoder, data["test_x"], aug.normalize)
    stats = (ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True) + 1e-6)
    pred, info, (w, b, mu, sd) = linear_probe(ftr[li], Y[li], fte, C, stats, args.seed)
    lp_acc = 100 * (pred == data["test_y"]).float().mean().item()
    with torch.no_grad():
        model.fc.weight.copy_((w / sd.t()).t())
        model.fc.bias.copy_(b - ((mu / sd) @ w).squeeze(0))

    steps = int(round(args.epochs * len(labeled_idx) / args.batch_size))
    warm = max(1, int(0.05 * steps))
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda")
    sampler = IndexSampler(labeled_idx, args.batch_size, gen, dev)
    t0 = time.time()
    model.train()
    for step in range(steps):
        lr = args.lr * (step + 1) / warm if step < warm else \
            0.5 * args.lr * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
        for gr in opt.param_groups:
            gr["lr"] = lr
        idx = sampler.next()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(aug(X[idx]).contiguous(memory_format=torch.channels_last))
        loss = F.cross_entropy(logits.float(), Y[idx])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        pred = torch.cat([model(aug.normalize(data["test_x"][i:i + 2000]).contiguous(memory_format=torch.channels_last)).argmax(1)
                          for i in range(0, data["test_x"].shape[0], 2000)])
    acc = 100 * (pred == data["test_y"]).float().mean().item()
    result = {"name": args.name, "args": vars(args), "n_labeled": int(len(labeled_idx)), "steps": steps,
              "train_time_s": time.time() - t0,
              "metrics": {"finetune_acc": acc, "linear_lab_acc": lp_acc}}
    np.savez_compressed(os.path.join(out_dir, "preds.npz"), finetune=pred.cpu().numpy().astype(np.int16))
    with open(res_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[done] {args.name}: finetune_acc {acc:.2f} (linear probe on the same labels {lp_acc:.2f})", flush=True)


if __name__ == "__main__":
    main()
