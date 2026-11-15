"""Semi-supervised baselines on the same labeled split: FixMatch and MixMatch.

Both use the ResNet-18 CIFAR encoder of the rest of the study (the original
papers use WRN-28-2) and are evaluated with the EMA model, as in the papers.

FixMatch (Sohn et al. 2020): labeled batch 64, mu = 7, weak = flip + translate,
strong = RandAugment (2 ops, random magnitude) + Cutout, threshold 0.95,
SGD-Nesterov lr 0.03, weight decay 5e-4, cosine schedule cos(7 pi k / 16 K).
MixMatch (Berthelot et al. 2019): batch 64 + 64, K = 2, T = 0.5, alpha = 0.75,
lambda_u = 75 with linear ramp-up, Adam lr 0.002, weight decay 0.02 * lr.

Example
-------
python -m dlt.semisup --name fixmatch_f0.1_s0 --method fixmatch --label_fraction 0.1 --seed 0
"""

import argparse
import copy
import json
import math
import os
import random
import time

import numpy as np
import PIL
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageOps
import torch
import torch.nn.functional as F
from PIL import Image

from .data import IndexSampler, labeled_subset, load_dataset, stratified_subset
from .evaluate import evaluate_encoder, to_numpy_preds
from .gpu_aug import AugConfig, GPUAugment
from .models import Classifier

# ----------------------------------------------------------------------------
# RandAugment as used by FixMatch (operations and ranges of the paper, Table 12)
# ----------------------------------------------------------------------------
PARAMETER_MAX = 10


def _fp(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _ip(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def _sign(v):
    return -v if random.random() < 0.5 else v


def _autocontrast(img, v, max_v, bias):
    return PIL.ImageOps.autocontrast(img)


def _brightness(img, v, max_v, bias):
    return PIL.ImageEnhance.Brightness(img).enhance(_fp(v, max_v) + bias)


def _color(img, v, max_v, bias):
    return PIL.ImageEnhance.Color(img).enhance(_fp(v, max_v) + bias)


def _contrast(img, v, max_v, bias):
    return PIL.ImageEnhance.Contrast(img).enhance(_fp(v, max_v) + bias)


def _equalize(img, v, max_v, bias):
    return PIL.ImageOps.equalize(img)


def _identity(img, v, max_v, bias):
    return img


def _posterize(img, v, max_v, bias):
    return PIL.ImageOps.posterize(img, _ip(v, max_v) + bias)


def _rotate(img, v, max_v, bias):
    return img.rotate(_sign(_ip(v, max_v) + bias))


def _sharpness(img, v, max_v, bias):
    return PIL.ImageEnhance.Sharpness(img).enhance(_fp(v, max_v) + bias)


def _shear_x(img, v, max_v, bias):
    return img.transform(img.size, Image.AFFINE, (1, _sign(_fp(v, max_v) + bias), 0, 0, 1, 0))


def _shear_y(img, v, max_v, bias):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, _sign(_fp(v, max_v) + bias), 1, 0))


def _solarize(img, v, max_v, bias):
    return PIL.ImageOps.solarize(img, 256 - (_ip(v, max_v) + bias))


def _translate_x(img, v, max_v, bias):
    return img.transform(img.size, Image.AFFINE, (1, 0, int(_sign(_fp(v, max_v) + bias) * img.size[0]), 0, 1, 0))


def _translate_y(img, v, max_v, bias):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, int(_sign(_fp(v, max_v) + bias) * img.size[1])))


def _cutout_abs(img, v):
    w, h = img.size
    x0 = int(max(0, np.random.uniform(0, w) - v / 2.0))
    y0 = int(max(0, np.random.uniform(0, h) - v / 2.0))
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle((x0, y0, int(min(w, x0 + v)), int(min(h, y0 + v))), (127, 127, 127))
    return img


RA_POOL = [(_autocontrast, None, None), (_brightness, 0.9, 0.05), (_color, 0.9, 0.05), (_contrast, 0.9, 0.05),
           (_equalize, None, None), (_identity, None, None), (_posterize, 4, 4), (_rotate, 30, 0),
           (_sharpness, 0.9, 0.05), (_shear_x, 0.3, 0), (_shear_y, 0.3, 0), (_solarize, 256, 0),
           (_translate_x, 0.3, 0), (_translate_y, 0.3, 0)]


def strong_augment(img, n=2, m=10):
    """flip + reflect-padded crop (pad = size/8) + RandAugment(n, random magnitude < m) + Cutout of size/2."""
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    size = img.size[0]
    pad = size // 8
    arr = np.pad(np.asarray(img), ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    i, j = np.random.randint(0, 2 * pad + 1), np.random.randint(0, 2 * pad + 1)
    img = Image.fromarray(arr[i:i + size, j:j + size])
    for op, max_v, bias in random.choices(RA_POOL, k=n):
        v = np.random.randint(1, m)
        if random.random() < 0.5:
            img = op(img, v, max_v, bias)
    return _cutout_abs(img, size // 2)


class StrongViews(torch.utils.data.Dataset):
    """Returns (index, strongly augmented uint8 image) for a fixed list of image indices."""

    def __init__(self, x_uint8_nhwc, indices):
        self.x = x_uint8_nhwc
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        img = strong_augment(Image.fromarray(self.x[idx]))
        return idx, torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)


def _worker_init(worker_id):
    seed = torch.utils.data.get_worker_info().seed % (2 ** 31)
    random.seed(seed)
    np.random.seed(seed)


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()):
        if be.dtype.is_floating_point:
            be.mul_(decay).add_(bm, alpha=1 - decay)
        else:
            be.copy_(bm)


@torch.no_grad()
def predict(model, x_uint8, normalize_fn):
    model.eval()
    out = []
    for i in range(0, x_uint8.shape[0], 2000):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(normalize_fn(x_uint8[i:i + 2000]).contiguous(memory_format=torch.channels_last)).argmax(1))
    return torch.cat(out)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--out", default="results/runs")
    p.add_argument("--ckpt_dir", default=None)
    p.add_argument("--data_dir", default=os.environ.get("DLT_DATA", "./data"))
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--method", required=True, choices=["fixmatch", "mixmatch"])
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--label_fraction", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=None,
                   help="default: the number of steps that processes as many images as a 100-epoch contrastive run")
    p.add_argument("--budget", type=float, default=1.0, help="multiplier on the default number of steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--holdout", type=int, default=0)
    p.add_argument("--init_run", default=None,
                   help="start from the encoder of this pretrained run (semi-supervised fine-tuning)")
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
    np.random.seed(args.seed)
    random.seed(args.seed)
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    data = load_dataset(args.dataset, args.data_dir)
    if args.holdout > 0:
        val_idx = stratified_subset(data["train_y"].numpy(), args.holdout / data["train_x"].shape[0], 12345)
        keep = np.setdiff1d(np.arange(data["train_x"].shape[0]), val_idx)
        data["test_x"], data["test_y"] = data["train_x"][val_idx], data["train_y"][val_idx]
        data["train_x"], data["train_y"] = data["train_x"][keep], data["train_y"][keep]
    x_nhwc = data["train_x"].permute(0, 2, 3, 1).contiguous().numpy()
    for k in ("train_x", "train_y", "test_x", "test_y"):
        data[k] = data[k].to(dev)
    X, Y, C = data["train_x"], data["train_y"], data["num_classes"]
    N = X.shape[0]
    labeled_idx = labeled_subset(data, args.label_fraction, args.seed)

    # image budget of a 100-epoch contrastive run: 100 * (N // 512) steps of 1024 views
    image_budget = 100 * (N // 512) * 1024 * args.budget
    per_step = 64 + 2 * 448 if args.method == "fixmatch" else 64 + 2 * 64
    steps = args.steps or int(round(image_budget / per_step))
    print(f"[run] {args.name}: {args.method} | labeled {len(labeled_idx)} | {steps} steps", flush=True)

    weak = GPUAugment(AugConfig(pad_crop=X.shape[-1] // 8, flip=True, pad_mode="reflection"), data["mean"], data["std"], dev, gen)
    model = Classifier(args.backbone, C)
    if args.init_run:
        src = torch.load(os.path.join(args.ckpt_dir or args.out, args.init_run, "model.pt"), map_location="cpu")["model"]
        enc = {k[len("encoder."):]: (v.float() if v.is_floating_point() else v)
               for k, v in src.items() if k.startswith("encoder.")}
        model.encoder.load_state_dict(enc)
        print(f"[init] encoder from {args.init_run} ({len(enc)} tensors)", flush=True)
    model = model.to(dev).to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model)
    for q in ema.parameters():
        q.requires_grad_(False)
    lab_sampler = IndexSampler(labeled_idx, 64, gen, dev)
    scaler = torch.amp.GradScaler("cuda")
    t0 = time.time()
    log = []

    if args.method == "fixmatch":
        opt = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, nesterov=True, weight_decay=5e-4)
        loader = torch.utils.data.DataLoader(
            StrongViews(x_nhwc, np.arange(N)), batch_size=448, shuffle=True, drop_last=True,
            num_workers=args.workers, worker_init_fn=_worker_init, persistent_workers=True, prefetch_factor=4)
        it = infinite(loader)
        mask_rate = 0.0
        for step in range(steps):
            for gr in opt.param_groups:
                gr["lr"] = 0.03 * math.cos(7 * math.pi * step / (16 * steps))
            li = lab_sampler.next()
            ui, xs = next(it)
            ui, xs = ui.to(dev, non_blocking=True), xs.to(dev, non_blocking=True)
            inputs = torch.cat([weak(X[li]), weak(X[ui]), weak.normalize(xs)]).contiguous(memory_format=torch.channels_last)
            model.train()
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(inputs).float()
            lx, lw, ls = logits[:64], logits[64:64 + 448], logits[64 + 448:]
            loss_x = F.cross_entropy(lx, Y[li])
            probs = torch.softmax(lw.detach(), dim=1)
            conf, pseudo = probs.max(1)
            mask = (conf >= 0.95).float()
            loss_u = (F.cross_entropy(ls, pseudo, reduction="none") * mask).mean()
            loss = loss_x + loss_u
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ema_update(ema, model, 0.999)
            mask_rate = 0.99 * mask_rate + 0.01 * mask.mean().item() if step % 20 == 0 else mask_rate
            if (step + 1) % max(1, steps // 10) == 0:
                acc = 100 * (predict(ema, data["test_x"], weak.normalize) == data["test_y"]).float().mean().item()
                log.append([step + 1, loss.item(), acc])
                print(f"  step {step + 1:6d}/{steps} loss {loss.item():.3f} ema-acc {acc:.2f} "
                      f"mask {mask.mean().item():.2f} ({time.time() - t0:.0f}s)", flush=True)
    else:  # mixmatch
        lr, alpha, lam_u, temp = 0.002, 0.75, 75.0, 0.5
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        unl_sampler = IndexSampler(np.arange(N), 64, gen, dev)
        beta = torch.distributions.Beta(alpha, alpha)
        for step in range(steps):
            li, ui = lab_sampler.next(), unl_sampler.next()
            xl, u1, u2 = weak(X[li]), weak(X[ui]), weak(X[ui])
            model.train()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                pu = (torch.softmax(model(u1.contiguous(memory_format=torch.channels_last)).float(), 1)
                      + torch.softmax(model(u2.contiguous(memory_format=torch.channels_last)).float(), 1)) / 2
            pt = pu ** (1 / temp)
            tu = pt / pt.sum(1, keepdim=True)
            all_x = torch.cat([xl, u1, u2])
            all_t = torch.cat([F.one_hot(Y[li], C).float(), tu, tu])
            lam = beta.sample().item()
            lam = max(lam, 1 - lam)
            perm = torch.randperm(all_x.shape[0], generator=gen, device=dev)
            mix_x = lam * all_x + (1 - lam) * all_x[perm]
            mix_t = lam * all_t + (1 - lam) * all_t[perm]
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(mix_x.contiguous(memory_format=torch.channels_last)).float()
            loss_x = -(F.log_softmax(logits[:64], 1) * mix_t[:64]).sum(1).mean()
            loss_u = ((torch.softmax(logits[64:], 1) - mix_t[64:]) ** 2).mean()
            w = lam_u * min(1.0, (step + 1) / steps)
            loss = loss_x + w * loss_u
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ema_update(ema, model, 0.999)
            with torch.no_grad():   # decoupled weight decay 0.02 * lr, as in the paper
                for q in model.parameters():
                    q.mul_(1 - 0.02 * lr)
            if (step + 1) % max(1, steps // 10) == 0:
                acc = 100 * (predict(ema, data["test_x"], weak.normalize) == data["test_y"]).float().mean().item()
                log.append([step + 1, loss.item(), acc])
                print(f"  step {step + 1:6d}/{steps} loss {loss.item():.3f} ema-acc {acc:.2f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    train_time = time.time() - t0

    ema.eval()
    pred = predict(ema, data["test_x"], weak.normalize)
    if "labeled_idx" in data:      # STL-10: only the labeled split has ground-truth labels
        ev = torch.as_tensor(data["labeled_idx"], device=dev)
        eval_data, e_lab, e_pool = dict(data, train_x=X[ev], train_y=Y[ev]), None, np.arange(len(ev))
    else:
        eval_data, e_lab, e_pool = data, labeled_idx, np.arange(N)
    metrics, preds = evaluate_encoder(ema.encoder, eval_data, weak.normalize, e_lab, e_pool, seed=args.seed)
    metrics["classifier_acc"] = 100 * (pred == data["test_y"]).float().mean().item()
    preds["classifier"] = pred
    result = {"name": args.name, "args": vars(args), "n_pool": N, "n_labeled": int(len(labeled_idx)),
              "steps": steps, "train_time_s": train_time, "log": log, "metrics": metrics,
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    np.savez_compressed(os.path.join(out_dir, "preds.npz"), **to_numpy_preds(preds),
                        labeled_idx=np.asarray(labeled_idx, dtype=np.int32))
    if args.ckpt_dir:
        os.makedirs(os.path.join(args.ckpt_dir, args.name), exist_ok=True)
        sd = {k: (v.half() if v.is_floating_point() else v) for k, v in ema.state_dict().items()}
        torch.save({"model": sd, "args": vars(args)}, os.path.join(args.ckpt_dir, args.name, "model.pt"))
    with open(res_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[done] {args.name}: classifier_acc {metrics['classifier_acc']:.2f}  linear_acc {metrics['linear_acc']:.2f}  "
          f"| train {train_time / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
