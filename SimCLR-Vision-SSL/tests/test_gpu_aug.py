"""Checks the batched GPU augmentations against torchvision, and the loss against
the reference SupCon / NT-Xent implementations shipped with the original code.

Run:  python -m tests.test_gpu_aug
"""
import importlib.util
import os
import sys

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dlt import gpu_aug as A          # noqa: E402
from dlt.losses import grouped_contrastive_loss, make_groups   # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
x = torch.rand(64, 3, 32, 32, device=dev)
B = x.shape[0]


def close(a, b, tol, name):
    err = (a - b).abs().max().item()
    assert err < tol, f"{name}: max abs err {err:.2e} >= {tol}"
    print(f"  ok  {name:28s} max|err| = {err:.2e}")


print("colour ops vs torchvision.transforms.functional")
f = torch.empty(B, device=dev).uniform_(0.6, 1.4)
hue = torch.empty(B, device=dev).uniform_(-0.1, 0.1)
close(A.adjust_brightness(x.clone(), f), torch.stack([TF.adjust_brightness(x[i], f[i].item()) for i in range(B)]), 1e-5, "brightness")
close(A.adjust_contrast(x.clone(), f), torch.stack([TF.adjust_contrast(x[i], f[i].item()) for i in range(B)]), 1e-5, "contrast")
close(A.adjust_saturation(x.clone(), f), torch.stack([TF.adjust_saturation(x[i], f[i].item()) for i in range(B)]), 1e-5, "saturation")
close(A.adjust_hue(x.clone(), hue), torch.stack([TF.adjust_hue(x[i], hue[i].item()) for i in range(B)]), 1e-4, "hue")
close(A.rgb_to_gray(x), torch.stack([TF.rgb_to_grayscale(x[i]) for i in range(B)]), 1e-5, "grayscale")

print("geometry vs torchvision")
g = torch.Generator(device=dev).manual_seed(1)
top, left, h, w = A.sample_rrc(B, (0.2, 1.0), (3 / 4, 4 / 3), 32, g, dev)
assert ((top >= 0) & (left >= 0) & (top + h <= 32) & (left + w <= 32)).all()
ref = torch.stack([TF.resized_crop(x[i], int(top[i]), int(left[i]), int(h[i]), int(w[i]), [32, 32], antialias=False) for i in range(B)])
close(A.resized_crop(x, top, left, h, w), ref, 1e-4, "resized_crop")
flip = torch.rand(B, device=dev) < 0.5
ref_f = torch.where(flip.view(-1, 1, 1, 1), ref.flip(3), ref)
close(A.resized_crop(x, top, left, h, w, flip), ref_f, 1e-4, "resized_crop + flip")
area = (h * w) / 1024
print(f"  crop area fraction: min {area.min():.3f} mean {area.mean():.3f} max {area.max():.3f}")

sigma = torch.empty(B, device=dev).uniform_(0.1, 2.0)
ref = torch.stack([TF.gaussian_blur(x[i], [3, 3], [sigma[i].item()] * 2) for i in range(B)])
close(A.gaussian_blur3(x, sigma), ref, 1e-5, "gaussian_blur3")

print("random-crop with padding keeps pixels intact")
xi = (x * 255).round().byte()
out = A.padded_crop(xi.float() / 255, 4, None, g)
vals = set(torch.unique((out * 255).round().long()).tolist())
assert vals.issubset(set(range(256)))
zero_frac = (out.sum(1) == 0).float().mean().item()
print(f"  ok  padded_crop                  zero-padded pixel fraction = {zero_frac:.3f} (expected ~0.12)")

print("erasing")
m = A.sample_erasing(4096, 0.7, (0.02, 0.2), (0.3, 3.3), 32, g, dev)
any_ = m.flatten(1).any(1).float().mean().item()
frac = m.flatten(1).float().mean(1)
print(f"  ok  erasing: applied to {any_:.3f} of images (p=0.7), mean erased area {frac[frac > 0].mean():.3f}")
assert abs(any_ - 0.7) < 0.04

print("full pipelines run and are normalised")
mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
for name in A.PIPELINES:
    aug = A.GPUAugment(name, mean, std, dev, torch.Generator(device=dev).manual_seed(0))
    y = aug(xi)
    assert y.shape == xi.shape and torch.isfinite(y).all(), name
print(f"  ok  {len(A.PIPELINES)} pipelines")

print("loss vs reference implementations")
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref_sup = load(os.path.join(root, "src", "Final Work", "supcon", "loss_supcon.py"), "ref_sup").SupConLoss(temperature=0.1)
ref_nt = load(os.path.join(root, "src", "Final Work", "loss.py"), "ref_nt").NTXentLoss(temperature=0.5)
N, C = 96, 10
z1 = F.normalize(torch.randn(N, 128, device=dev), dim=1)
z2 = F.normalize(torch.randn(N, 128, device=dev), dim=1)
labels = torch.randint(0, C, (N,), device=dev)
idx = torch.arange(N, device=dev)
z = torch.cat([z1, z2])
all_lab = torch.ones(N, dtype=torch.bool, device=dev)
no_lab = torch.zeros(N, dtype=torch.bool, device=dev)
close(grouped_contrastive_loss(z, make_groups(idx, labels, all_lab, C), 0.1),
      ref_sup(torch.stack([z1, z2], 1), labels), 1e-4, "SupCon (all labeled)")
close(grouped_contrastive_loss(z, make_groups(idx, labels, no_lab, C), 0.5),
      ref_nt(z1, z2), 1e-4, "NT-Xent (no labels)")
half = torch.arange(N, device=dev) < N // 2
mask = (labels.view(-1, 1) == labels.view(1, -1)) & half.view(-1, 1) & half.view(1, -1)
mask = mask | torch.eye(N, dtype=torch.bool, device=dev)
close(grouped_contrastive_loss(z, make_groups(idx, labels, half, C), 0.1),
      ref_sup(torch.stack([z1, z2], 1), mask=mask.float()), 1e-4, "semi-supervised (half labeled)")
print("all checks passed")
