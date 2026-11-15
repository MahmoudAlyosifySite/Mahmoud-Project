"""Batched data augmentation on the GPU.

Every transform draws its random parameters *per image*, exactly like the
torchvision transforms it replaces (RandomResizedCrop, RandomHorizontalFlip,
ColorJitter, RandomGrayscale, GaussianBlur, RandomErasing), but the whole batch
is processed at once. ``tests/test_gpu_aug.py`` checks each operation against
torchvision for fixed parameters.

Order of operations inside a pipeline:
    rotation -> crop/flip -> colour jitter -> grayscale -> blur -> normalise -> erasing
"""

import math
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AugConfig:
    crop_scale: Optional[Tuple[float, float]] = None   # RandomResizedCrop area range
    crop_ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0)
    flip: bool = False                                 # RandomHorizontalFlip(p=0.5)
    jitter: Optional[Tuple[float, float, float, float]] = None  # brightness, contrast, saturation, hue
    jitter_p: float = 0.8
    gray_p: float = 0.0                                # RandomGrayscale
    blur: bool = False                                 # GaussianBlur(3, sigma in blur_sigma), always applied
    blur_sigma: Tuple[float, float] = (0.1, 2.0)
    erase_p: float = 0.0                               # RandomErasing after normalisation
    erase_scale: Tuple[float, float] = (0.02, 0.2)
    erase_ratio: Tuple[float, float] = (0.3, 3.3)
    rot90: bool = False                                # random rotation by 0/90/180/270 degrees
    pad_crop: int = 0                                  # RandomCrop(32, padding=pad_crop)
    pad_mode: str = "zeros"                            # "zeros" or "reflection"


# Colour distortion strength s = 0.5 (Chen et al. 2020, CIFAR-10 setting):
# ColorJitter(0.8 s, 0.8 s, 0.8 s, 0.2 s) applied with probability 0.8.
CJ = (0.4, 0.4, 0.4, 0.1)

# Pipelines. The "expNN" names are the experiment ids of the original
# 42-configuration augmentation sweep of this project.
PIPELINES = {
    "none": AugConfig(),
    "sup": AugConfig(pad_crop=4, flip=True),
    # --- canonical SimCLR / SupContrast CIFAR recipe (= exp8) ----------------
    "simclr": AugConfig(crop_scale=(0.2, 1.0), flip=True, jitter=CJ, gray_p=0.2),
    # stronger variant (ImageNet-style crop range, full-strength colour distortion)
    "simclr_strong": AugConfig(crop_scale=(0.08, 1.0), flip=True, jitter=(0.8, 0.8, 0.8, 0.2), gray_p=0.2),
    # --- original 8-configuration study --------------------------------------
    "exp1": AugConfig(crop_scale=(0.2, 1.0)),
    "exp2": AugConfig(crop_scale=(0.2, 1.0), flip=True),
    "exp3": AugConfig(crop_scale=(0.2, 1.0), jitter=CJ),
    "exp4": AugConfig(crop_scale=(0.2, 1.0), gray_p=0.2),
    "exp5": AugConfig(crop_scale=(0.2, 1.0), flip=True, jitter=CJ),
    "exp6": AugConfig(crop_scale=(0.2, 1.0), flip=True, gray_p=0.2),
    "exp7": AugConfig(crop_scale=(0.2, 1.0), jitter=CJ, gray_p=0.2),
    "exp8": AugConfig(crop_scale=(0.2, 1.0), flip=True, jitter=CJ, gray_p=0.2),
    # --- structural pipelines without colour jitter ("base") ------------------
    "exp9": AugConfig(crop_scale=(0.6, 1.0), blur=True),
    "exp10": AugConfig(crop_scale=(0.6, 1.0), erase_p=0.7),
    "exp13": AugConfig(crop_scale=(0.6, 1.0), flip=True, blur=True),
    "exp35": AugConfig(crop_scale=(0.8, 1.0)),
    "exp36": AugConfig(rot90=True),
    # --- the same pipelines with colour jitter added --------------------------
    "exp38": AugConfig(rot90=True, jitter=CJ),
    "exp39": AugConfig(crop_scale=(0.8, 1.0), jitter=CJ),
    "exp40": AugConfig(crop_scale=(0.6, 1.0), blur=True, jitter=CJ),
    "exp41": AugConfig(crop_scale=(0.6, 1.0), flip=True, blur=True, jitter=CJ),
    "exp42": AugConfig(crop_scale=(0.6, 1.0), erase_p=0.7, jitter=CJ),
}


def get_pipeline(name):
    """Look up a pipeline. ``name+nojitter`` / ``name+nogray`` remove one component."""
    base, *mods = name.split("+")
    cfg = PIPELINES[base]
    for m in mods:
        if m == "nojitter":
            cfg = replace(cfg, jitter=None)
        elif m == "nogray":
            cfg = replace(cfg, gray_p=0.0)
        elif m == "noflip":
            cfg = replace(cfg, flip=False)
        else:
            raise ValueError(f"unknown modifier {m}")
    return cfg


# ----------------------------------------------------------------------------
# colour helpers (float images in [0, 1], shape [B, 3, H, W])
# ----------------------------------------------------------------------------

def rgb_to_gray(x):
    r, g, b = x.unbind(dim=1)
    return (0.2989 * r + 0.587 * g + 0.114 * b).unsqueeze(1)


def _rgb2hsv(img):
    r, g, b = img.unbind(dim=-3)
    maxc = torch.max(img, dim=-3).values
    minc = torch.min(img, dim=-3).values
    eqc = maxc == minc
    cr = maxc - minc
    ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc)
    cr_div = torch.where(eqc, ones, cr)
    rc = (maxc - r) / cr_div
    gc = (maxc - g) / cr_div
    bc = (maxc - b) / cr_div
    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = torch.fmod((hr + hg + hb) / 6.0 + 1.0, 1.0)
    return torch.stack((h, s, maxc), dim=-3)


def _hsv2rgb(img):
    h, s, v = img.unbind(dim=-3)
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    i = i.to(torch.int32) % 6
    p = torch.clamp(v * (1.0 - s), 0.0, 1.0)
    q = torch.clamp(v * (1.0 - s * f), 0.0, 1.0)
    t = torch.clamp(v * (1.0 - s * (1.0 - f)), 0.0, 1.0)
    mask = i.unsqueeze(-3) == torch.arange(6, device=i.device).view(-1, 1, 1)
    a1 = torch.stack((v, q, p, p, t, v), dim=-3)
    a2 = torch.stack((t, v, v, q, p, p), dim=-3)
    a3 = torch.stack((p, p, t, v, v, q), dim=-3)
    a4 = torch.stack((a1, a2, a3), dim=-4)
    return torch.einsum("...ijk,...xijk->...xjk", mask.to(img.dtype), a4)


def adjust_brightness(x, f):
    return (x * f.view(-1, 1, 1, 1)).clamp_(0, 1)


def adjust_contrast(x, f):
    f = f.view(-1, 1, 1, 1)
    mean = rgb_to_gray(x).mean(dim=(1, 2, 3), keepdim=True)
    return (f * x + (1 - f) * mean).clamp_(0, 1)


def adjust_saturation(x, f):
    f = f.view(-1, 1, 1, 1)
    return (f * x + (1 - f) * rgb_to_gray(x)).clamp_(0, 1)


def adjust_hue(x, f):
    hsv = _rgb2hsv(x)
    h = (hsv[:, 0] + f.view(-1, 1, 1)) % 1.0
    return _hsv2rgb(torch.stack((h, hsv[:, 1], hsv[:, 2]), dim=1))


_JITTER_OPS = (adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue)


# ----------------------------------------------------------------------------
# parameter samplers
# ----------------------------------------------------------------------------

def _rand(shape, g, device):
    return torch.rand(shape, generator=g, device=device)


def _uniform(shape, lo, hi, g, device):
    return lo + (hi - lo) * _rand(shape, g, device)


def _first_success(ok):
    """For a [T, B] success table return (index of first success per column, any success)."""
    any_ok = ok.any(dim=0)
    first = ok.float().argmax(dim=0)
    return first, any_ok


def sample_rrc(B, scale, ratio, size, g, device, attempts=10):
    """Vectorised torchvision ``RandomResizedCrop.get_params`` -> (top, left, h, w).

    All ``attempts`` proposals are drawn at once and the first valid one is kept,
    which is the same distribution as torchvision's sequential rejection loop.
    """
    H = W = size
    area = float(H * W)
    log_lo, log_hi = math.log(ratio[0]), math.log(ratio[1])
    target = area * _uniform((attempts, B), scale[0], scale[1], g, device)
    aspect = torch.exp(_uniform((attempts, B), log_lo, log_hi, g, device))
    ww = torch.round(torch.sqrt(target * aspect))
    hh = torch.round(torch.sqrt(target / aspect))
    ok = (ww > 0) & (ww <= W) & (hh > 0) & (hh <= H)
    first, any_ok = _first_success(ok)
    w = ww.gather(0, first.view(1, -1)).squeeze(0)
    h = hh.gather(0, first.view(1, -1)).squeeze(0)
    # samples that never succeeded keep the whole image (torchvision fallback
    # for a square input whose ratio lies inside the allowed range)
    w = torch.where(any_ok, w, torch.full_like(w, float(W)))
    h = torch.where(any_ok, h, torch.full_like(h, float(H)))
    r = _rand((2, B), g, device)
    top = torch.floor(r[0] * (H - h + 1))
    left = torch.floor(r[1] * (W - w + 1))
    return top, left, h, w


def sample_erasing(B, p, scale, ratio, size, g, device, attempts=10):
    """Vectorised torchvision ``RandomErasing.get_params`` -> boolean mask [B,1,H,W]."""
    H = W = size
    area = float(H * W)
    log_lo, log_hi = math.log(ratio[0]), math.log(ratio[1])
    target = area * _uniform((attempts, B), scale[0], scale[1], g, device)
    aspect = torch.exp(_uniform((attempts, B), log_lo, log_hi, g, device))
    hh = torch.round(torch.sqrt(target * aspect))
    ww = torch.round(torch.sqrt(target / aspect))
    ok = (hh < H) & (ww < W)
    first, any_ok = _first_success(ok)
    eh = hh.gather(0, first.view(1, -1)).squeeze(0)
    ew = ww.gather(0, first.view(1, -1)).squeeze(0)
    r = _rand((3, B), g, device)
    top = torch.floor(r[0] * (H - eh + 1))
    left = torch.floor(r[1] * (W - ew + 1))
    apply = (r[2] < p) & any_ok
    ys = torch.arange(H, device=device).view(1, H, 1)
    xs = torch.arange(W, device=device).view(1, 1, W)
    mask = ((ys >= top.view(-1, 1, 1)) & (ys < (top + eh).view(-1, 1, 1))
            & (xs >= left.view(-1, 1, 1)) & (xs < (left + ew).view(-1, 1, 1)))
    return (mask & apply.view(-1, 1, 1)).unsqueeze(1)


# ----------------------------------------------------------------------------
# geometric helpers
# ----------------------------------------------------------------------------

def resized_crop(x, top, left, h, w, flip=None):
    """Crop boxes (top, left, h, w) and resize them back to the input size.

    Bilinear interpolation with half-pixel centres; sampling positions are
    clamped to the crop box, which is what crop-then-resize does in torchvision.
    """
    B, _, H, W = x.shape
    u = torch.arange(W, device=x.device, dtype=x.dtype) + 0.5
    v = torch.arange(H, device=x.device, dtype=x.dtype) + 0.5
    xs = left.view(-1, 1) + u.view(1, -1) * (w / W).view(-1, 1) - 0.5   # source pixel coordinates
    ys = top.view(-1, 1) + v.view(1, -1) * (h / H).view(-1, 1) - 0.5
    xs = torch.minimum(torch.maximum(xs, left.view(-1, 1)), (left + w - 1).view(-1, 1))
    ys = torch.minimum(torch.maximum(ys, top.view(-1, 1)), (top + h - 1).view(-1, 1))
    if flip is not None:
        xs = torch.where(flip.view(-1, 1), xs.flip(1), xs)
    gx = (2 * xs + 1) / W - 1
    gy = (2 * ys + 1) / H - 1
    grid = torch.stack((gx.view(B, 1, W).expand(B, H, W), gy.view(B, H, 1).expand(B, H, W)), dim=-1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def padded_crop(x, pad, flip, g, mode="zeros"):
    """RandomCrop(size, padding=pad) with zero or reflection padding, plus optional flip."""
    B, _, H, W = x.shape
    dev = x.device
    dx = torch.randint(-pad, pad + 1, (B,), generator=g, device=dev).to(x.dtype)
    dy = torch.randint(-pad, pad + 1, (B,), generator=g, device=dev).to(x.dtype)
    theta = torch.zeros(B, 2, 3, device=dev, dtype=x.dtype)
    sx = torch.ones(B, device=dev, dtype=x.dtype)
    if flip is not None:
        sx = torch.where(flip, -sx, sx)
    theta[:, 0, 0] = sx
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = 2 * dx / W
    theta[:, 1, 2] = 2 * dy / H
    grid = F.affine_grid(theta, (B, 3, H, W), align_corners=False)
    return F.grid_sample(x, grid, mode="nearest", padding_mode=mode, align_corners=False)


def gaussian_blur3(x, sigma):
    """3x3 Gaussian blur with a per-image sigma and reflect padding."""
    a = torch.exp(-0.5 / sigma ** 2)
    norm = 1 + 2 * a
    side = (a / norm).view(-1, 1, 1, 1)
    mid = (1 / norm).view(-1, 1, 1, 1)
    xp = F.pad(x, (1, 1, 0, 0), mode="reflect")
    x = side * (xp[..., :-2] + xp[..., 2:]) + mid * xp[..., 1:-1]
    xp = F.pad(x, (0, 0, 1, 1), mode="reflect")
    return side * (xp[..., :-2, :] + xp[..., 2:, :]) + mid * xp[..., 1:-1, :]


# ----------------------------------------------------------------------------
# the augmentation module
# ----------------------------------------------------------------------------

class GPUAugment:
    """Applies one pipeline to a uint8 batch [B,3,H,W] and returns normalised floats."""

    def __init__(self, cfg, mean, std, device, generator=None):
        self.cfg = get_pipeline(cfg) if isinstance(cfg, str) else cfg
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.device = device
        self.g = generator

    @torch.no_grad()
    def __call__(self, x_uint8):
        cfg, g, dev = self.cfg, self.g, self.device
        x = x_uint8.float().div_(255.0)
        B, _, H, W = x.shape

        if cfg.rot90:
            k = torch.randint(0, 4, (B,), generator=g, device=dev).view(-1, 1, 1, 1)
            out = x
            for r in (1, 2, 3):
                out = torch.where(k == r, torch.rot90(x, r, dims=(2, 3)), out)
            x = out

        flip = (_rand((B,), g, dev) < 0.5) if cfg.flip else None
        if cfg.crop_scale is not None:
            top, left, h, w = sample_rrc(B, cfg.crop_scale, cfg.crop_ratio, H, g, dev)
            x = resized_crop(x, top, left, h, w, flip)
        elif cfg.pad_crop > 0:
            x = padded_crop(x, cfg.pad_crop, flip, g, cfg.pad_mode)
        elif flip is not None:
            x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)

        if cfg.jitter is not None:
            b, c, s, hue = cfg.jitter
            factors = (
                _uniform((B,), max(0.0, 1 - b), 1 + b, g, dev),
                _uniform((B,), max(0.0, 1 - c), 1 + c, g, dev),
                _uniform((B,), max(0.0, 1 - s), 1 + s, g, dev),
                _uniform((B,), -hue, hue, g, dev),
            )
            apply = _rand((B,), g, dev) < cfg.jitter_p
            order = _rand((B, 4), g, dev).argsort(dim=1)     # random op order per image
            for pos in range(4):
                for op_id, op in enumerate(_JITTER_OPS):
                    sel = ((order[:, pos] == op_id) & apply).nonzero(as_tuple=True)[0]
                    x[sel] = op(x[sel], factors[op_id][sel])

        if cfg.gray_p > 0:
            m = _rand((B,), g, dev) < cfg.gray_p
            x = torch.where(m.view(-1, 1, 1, 1), rgb_to_gray(x).expand(-1, 3, -1, -1), x)

        if cfg.blur:
            sigma = _uniform((B,), cfg.blur_sigma[0], cfg.blur_sigma[1], g, dev)
            x = gaussian_blur3(x, sigma)

        x = (x - self.mean) / self.std

        if cfg.erase_p > 0:
            mask = sample_erasing(B, cfg.erase_p, cfg.erase_scale, cfg.erase_ratio, H, g, dev)
            x = x.masked_fill(mask, 0.0)
        return x

    @torch.no_grad()
    def normalize(self, x_uint8):
        return (x_uint8.float().div_(255.0) - self.mean) / self.std
