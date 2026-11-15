"""Resolution shift: high-resolution ImageNet dogs shrunk to 32x32.

Queries are the validation images of Imagewoof (ten dog breeds from ImageNet, >= 320 px).
Every image is centre-cropped to a square and downsampled to 32x32 (bicubic, antialiased),
which is what the retrieval demo of this project does with an uploaded photo.

For every CIFAR-10 encoder we report, for these out-of-distribution dogs and for the
in-distribution CIFAR-10 test dogs:
  * dog recall of the linear probe (share of images classified as "dog");
  * retrieval precision@5: share of the 5 nearest CIFAR-10 test images that are dogs.

CLIP ViT-B/32 (zero-shot, 10 CIFAR-10 class prompts) is evaluated on the same images at
several resolutions as an external reference. It is not a competitor: it was trained on
400M image-text pairs. It tells us how much of the failure is information lost by the
downsampling itself and how much is the encoder.

    python scripts/ood_resolution.py --runs simclr_all_e100_s0 ... --ckpt_dir <dir> --ood_dir <imagewoof2-320/val>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dlt.data import CIFAR10_CLASSES, load_dataset     # noqa: E402
from dlt.evaluate import linear_probe                    # noqa: E402
from dlt.gpu_aug import GPUAugment                       # noqa: E402
from dlt.models import Classifier, ContrastiveModel, extract_features   # noqa: E402

DOG = CIFAR10_CLASSES.index("dog")


def load_squares(folder, size=224, limit=None):
    """Centre-cropped square images, resized to ``size`` (uint8 [N,3,size,size])."""
    files = sorted(glob.glob(os.path.join(folder, "*", "*.JPEG")))
    if limit:
        files = files[:limit]
    out = []
    for f in files:
        im = Image.open(f).convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size), Image.BICUBIC)
        out.append(np.asarray(im))
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).contiguous(), files


def downsample(x_uint8, size):
    x = F.interpolate(x_uint8.float(), size=(size, size), mode="bicubic", antialias=True, align_corners=False)
    return x.round().clamp(0, 255).to(torch.uint8)


def boot_ci(v, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, dtype=np.float64)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return [100 * float(np.percentile(m, 2.5)), 100 * float(np.percentile(m, 97.5))]


@torch.no_grad()
def eval_encoder(run, ckpt_dir, data, ood32, dev="cuda"):
    ckpt = torch.load(os.path.join(ckpt_dir, run, "model.pt"), map_location=dev)
    sd = {k: v.float() if v.is_floating_point() else v for k, v in ckpt["model"].items()}
    backbone = ckpt["args"].get("backbone", "resnet18")
    model = Classifier(backbone, 10) if any(k.startswith("fc.") for k in sd) else ContrastiveModel(backbone)
    model.load_state_dict(sd)
    enc = model.encoder.to(dev).to(memory_format=torch.channels_last).eval()
    norm = GPUAugment("none", data["mean"], data["std"], dev).normalize
    ftr = extract_features(enc, data["train_x"], norm)
    fte = extract_features(enc, data["test_x"], norm)
    food = extract_features(enc, ood32, norm)
    stats = (ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True) + 1e-6)
    pred_te, _, (w, b, mu, sd_) = linear_probe(ftr, data["train_y"], fte, 10, stats, 0)
    pred_ood = (((food - mu) / sd_) @ w + b).argmax(1)
    yte = data["test_y"]
    dogs = yte == DOG
    # retrieval in the CIFAR-10 test index, cosine similarity on encoder features
    index = F.normalize(fte, dim=1)
    sim_ood = F.normalize(food, dim=1) @ index.t()
    nn_ood = yte[sim_ood.topk(5, dim=1).indices]
    sim_id = index[dogs] @ index.t()
    sim_id[torch.arange(int(dogs.sum())), dogs.nonzero(as_tuple=True)[0]] = -2.0   # drop the query itself
    nn_id = yte[sim_id.topk(5, dim=1).indices]
    hist = torch.bincount(pred_ood, minlength=10).float() / len(pred_ood)
    return {
        "test_acc": 100 * (pred_te == yte).float().mean().item(),
        "dog_recall_cifar": 100 * (pred_te[dogs] == DOG).float().mean().item(),
        "dog_recall_ood": 100 * (pred_ood == DOG).float().mean().item(),
        "dog_recall_ood_ci": boot_ci((pred_ood == DOG).float().cpu().numpy()),
        "p_at_5_cifar": 100 * (nn_id == DOG).float().mean().item(),
        "p_at_5_ood": 100 * (nn_ood == DOG).float().mean().item(),
        "top1_cifar": 100 * (nn_id[:, 0] == DOG).float().mean().item(),
        "top1_ood": 100 * (nn_ood[:, 0] == DOG).float().mean().item(),
        "ood_prediction_hist": {CIFAR10_CLASSES[i]: round(100 * hist[i].item(), 2) for i in range(10)},
    }


@torch.no_grad()
def eval_clip(data, ood224, resolutions, dev="cuda", name="openai/clip-vit-base-patch32"):
    from transformers import CLIPModel, CLIPTokenizer
    model = CLIPModel.from_pretrained(name).to(dev).eval()
    tok = CLIPTokenizer.from_pretrained(name)
    text = tok([f"a photo of a {c}" for c in CIFAR10_CLASSES], padding=True, return_tensors="pt").to(dev)
    tfeat = model.get_text_features(**text)
    tfeat = F.normalize(tfeat if torch.is_tensor(tfeat) else tfeat.pooler_output, dim=-1)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)

    def predict(x_uint8):
        preds = []
        for i in range(0, x_uint8.shape[0], 500):
            xb = x_uint8[i:i + 500].to(dev).float()
            if xb.shape[-1] != 224:
                xb = F.interpolate(xb, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 255)
            f = model.get_image_features(pixel_values=(xb / 255.0 - mean) / std)
            f = F.normalize(f if torch.is_tensor(f) else f.pooler_output, dim=-1)
            preds.append((f @ tfeat.t()).argmax(1))
        return torch.cat(preds)

    out = {}
    pte = predict(data["test_x"])
    yte = data["test_y"]
    out["test_acc"] = 100 * (pte == yte).float().mean().item()
    out["dog_recall_cifar"] = 100 * (pte[yte == DOG] == DOG).float().mean().item()
    out["dog_recall_ood_by_resolution"] = {}
    for r in resolutions:
        x = ood224 if r == 224 else downsample(ood224.to(dev), r)
        pr = predict(x)
        out["dog_recall_ood_by_resolution"][str(r)] = 100 * (pr == DOG).float().mean().item()
        if r == 32:
            hist = torch.bincount(pr, minlength=10).float() / len(pr)
            out["ood_prediction_hist"] = {CIFAR10_CLASSES[i]: round(100 * hist[i].item(), 2) for i in range(10)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--ood_dir", required=True)
    p.add_argument("--data_dir", default=os.environ.get("DLT_DATA", "./data"))
    p.add_argument("--out", default="results/ood_resolution.json")
    p.add_argument("--no_clip", action="store_true")
    args = p.parse_args()
    dev = "cuda"
    data = load_dataset("cifar10", args.data_dir)
    for k in ("train_x", "train_y", "test_x", "test_y"):
        data[k] = data[k].to(dev)
    ood224, files = load_squares(args.ood_dir)
    ood32 = downsample(ood224.to(dev), 32)
    res = {"n_ood": len(files), "encoders": {}}
    for run in args.runs:
        if not os.path.exists(os.path.join(args.ckpt_dir, run, "model.pt")):
            print(f"[skip] no checkpoint for {run}")
            continue
        res["encoders"][run] = r = eval_encoder(run, args.ckpt_dir, data, ood32)
        print(f"{run:34s} test {r['test_acc']:.2f} | dog recall: CIFAR {r['dog_recall_cifar']:.1f} -> OOD {r['dog_recall_ood']:.1f} "
              f"| P@5: CIFAR {r['p_at_5_cifar']:.1f} -> OOD {r['p_at_5_ood']:.1f}", flush=True)
    if not args.no_clip:
        res["clip"] = c = eval_clip(data, ood224, [16, 32, 64, 128, 224])
        print(f"CLIP ViT-B/32 zero-shot: test {c['test_acc']:.2f} | dog recall CIFAR {c['dog_recall_cifar']:.1f} | "
              f"OOD by resolution {c['dog_recall_ood_by_resolution']}")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
