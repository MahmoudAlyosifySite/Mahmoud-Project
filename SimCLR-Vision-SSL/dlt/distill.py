"""Two-stage pipeline in the style of SimCLR v2 (Chen et al., 2020b): fine-tune, then self-distil.

1. Teacher: the pretrained encoder is fine-tuned on the labeled subset (the default recipe of
   dlt.finetune: linear-probe initialisation, lr 0.01, 60 passes over the labeled images).
2. Student: a network of the same architecture, initialised from the same pretrained encoder,
   is trained on ALL training images (no labels) to match the teacher's class probabilities.

    python -m dlt.distill --name distill_simclr_all_e200_s0 --from_run simclr_all_e200_s0 \
        --label_fraction 0.1 --seed 0 --ckpt_dir <dir>
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

from .data import IndexSampler, labeled_subset, load_dataset
from .evaluate import evaluate_encoder, linear_probe, to_numpy_preds
from .gpu_aug import GPUAugment, sup_pipeline
from .models import Classifier, extract_features


def cosine(step, total, base, warm):
    if step < warm:
        return base * (step + 1) / warm
    return 0.5 * base * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


@torch.no_grad()
def predict(model, x, normalize):
    model.eval()
    out = []
    for i in range(0, x.shape[0], 2000):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(normalize(x[i:i + 2000]).contiguous(memory_format=torch.channels_last)).argmax(1))
    return torch.cat(out)


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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ft_epochs", type=float, default=60.0)
    p.add_argument("--ft_lr", type=float, default=0.01)
    p.add_argument("--epochs", type=float, default=100.0, help="distillation passes over all training images")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--temperature", type=float, default=1.0)
    args = p.parse_args(argv)

    out_dir = os.path.join(args.out, args.name)
    res_path = os.path.join(out_dir, "result.json")
    if os.path.exists(res_path):
        print(f"[skip] {res_path} exists")
        return
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    data = load_dataset(args.dataset, args.data_dir)
    for k in ("train_x", "train_y", "test_x", "test_y"):
        data[k] = data[k].to(dev)
    X, Y, C = data["train_x"], data["train_y"], data["num_classes"]
    N = X.shape[0]
    labeled_idx = labeled_subset(data, args.label_fraction, args.seed)
    li = torch.as_tensor(labeled_idx, device=dev)
    aug = GPUAugment(sup_pipeline(X.shape[-1]), data["mean"], data["std"], dev, gen)

    ckpt = torch.load(os.path.join(args.ckpt_dir or args.out, args.from_run, "model.pt"), map_location=dev)
    enc_sd = {k[len("encoder."):]: v.float() if v.is_floating_point() else v
              for k, v in ckpt["model"].items() if k.startswith("encoder.")}

    # ---- 1. teacher: LP-initialised fine-tuning on the labeled subset
    teacher = Classifier(args.backbone, C).to(dev)
    teacher.encoder.load_state_dict(enc_sd)
    teacher = teacher.to(memory_format=torch.channels_last)
    ftr = extract_features(teacher.encoder, X[li], aug.normalize)
    stats = (ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True) + 1e-6)
    _, _, (w, b, mu, sd) = linear_probe(ftr, Y[li], ftr[:10], C, stats, args.seed)
    with torch.no_grad():
        teacher.fc.weight.copy_((w / sd.t()).t())
        teacher.fc.bias.copy_(b - ((mu / sd) @ w).squeeze(0))
    steps = int(round(args.ft_epochs * len(labeled_idx) / 128))
    opt = torch.optim.SGD(teacher.parameters(), lr=args.ft_lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    scaler = torch.amp.GradScaler("cuda")
    sampler = IndexSampler(labeled_idx, 128, gen, dev)
    t0 = time.time()
    teacher.train()
    for step in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = cosine(step, steps, args.ft_lr, max(1, int(0.05 * steps)))
        idx = sampler.next()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = teacher(aug(X[idx]).contiguous(memory_format=torch.channels_last))
        loss = F.cross_entropy(logits.float(), Y[idx])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    teacher_pred = predict(teacher, data["test_x"], aug.normalize)
    teacher_acc = 100 * (teacher_pred == data["test_y"]).float().mean().item()
    print(f"[teacher] fine-tuned on {len(labeled_idx)} labels: {teacher_acc:.2f}", flush=True)
    teacher.eval()
    for q in teacher.parameters():
        q.requires_grad_(False)

    # ---- 2. student: same architecture, same pretrained initialisation, trained on all images without labels
    student = Classifier(args.backbone, C).to(dev)
    student.encoder.load_state_dict(enc_sd)
    student = student.to(memory_format=torch.channels_last)
    B = args.batch_size
    steps = int(round(args.epochs * (N // B)))
    opt = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda")
    sampler = IndexSampler(np.arange(N), B, gen, dev)
    T = args.temperature
    student.train()
    for step in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = cosine(step, steps, args.lr, max(1, int(0.05 * steps)))
        x = aug(X[sampler.next()]).contiguous(memory_format=torch.channels_last)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            target = torch.softmax(teacher(x).float() / T, dim=1)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = student(x)
        loss = -(target * torch.log_softmax(logits.float() / T, dim=1)).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if (step + 1) % 2000 == 0:
            print(f"  step {step + 1:6d}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    train_time = time.time() - t0

    pred = predict(student, data["test_x"], aug.normalize)
    if "labeled_idx" in data:
        ev = torch.as_tensor(data["labeled_idx"], device=dev)
        eval_data, e_lab, e_pool = dict(data, train_x=X[ev], train_y=Y[ev]), None, np.arange(len(ev))
    else:
        eval_data, e_lab, e_pool = data, labeled_idx, np.arange(N)
    student.eval()
    metrics, preds = evaluate_encoder(student.encoder, eval_data, aug.normalize, e_lab, e_pool, seed=args.seed, full=False)
    metrics["classifier_acc"] = 100 * (pred == data["test_y"]).float().mean().item()
    metrics["teacher_acc"] = teacher_acc
    preds["classifier"], preds["teacher"] = pred, teacher_pred
    result = {"name": args.name, "args": vars(args), "n_labeled": int(len(labeled_idx)), "steps": steps,
              "train_time_s": train_time, "metrics": metrics}
    np.savez_compressed(os.path.join(out_dir, "preds.npz"), **to_numpy_preds(preds))
    with open(res_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[done] {args.name}: student {metrics['classifier_acc']:.2f} (teacher {teacher_acc:.2f}) "
          f"| {train_time / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
