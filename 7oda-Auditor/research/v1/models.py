"""v1 models: challenge-conditioned teacher and student encoders, ArcFace head
and the self-supervised heads.

Both encoders read the SPEC 14.1 tensor ``x`` ``[B, 10, 256]`` (channels 0-8:
the aligned recording, channel 9: the challenge template) and return a
128-d L2-normalized ``embedding`` and 3 ``hand`` logits (right, left,
twoHanded), as SPEC 14.2 requires.

Conditioning: a small encoder reads the challenge channel and produces
time-aligned features at every stage resolution; FiLM layers modulate the
signal features with them (``y = x * (1 + gamma) + beta``, zero-initialized
so training starts from the unconditioned network). The challenge channel is
also part of the stem input.

Only ``torch`` is imported here so the export environment can load it.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EMB_DIM = 128
N_HAND = 3
IN_CH = 10
SIG_CH = 9


def l2_normalize(e: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """x / sqrt(sum x^2 + eps): plain ops that convert cleanly to Core ML."""
    return e / torch.sqrt(torch.sum(e * e, dim=1, keepdim=True) + eps)


class FiLM(nn.Module):
    def __init__(self, cond_ch: int, ch: int):
        super().__init__()
        self.proj = nn.Conv1d(cond_ch, 2 * ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        g, b = self.proj(cond).chunk(2, dim=1)
        return x * (1.0 + g) + b


class ChallengeEncoder(nn.Module):
    """Template channel -> features at resolutions T, T/2, T/4, ... ."""

    def __init__(self, width: int, n_levels: int, k: int = 9):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv1d(1, width, k, padding=k // 2), nn.ReLU(inplace=True),
                                 nn.Conv1d(width, width, k, padding=k // 2), nn.ReLU(inplace=True))
        self.down = nn.ModuleList([
            nn.Sequential(nn.Conv1d(width, width, 5, stride=2, padding=2), nn.ReLU(inplace=True))
            for _ in range(n_levels - 1)])

    def forward(self, tmpl):
        h = self.inp(tmpl)
        out = [h]
        for d in self.down:
            h = d(h)
            out.append(h)
        return out


# ---------------------------------------------------------------------------
# Teacher: 1D ResNet with FiLM (~10M parameters)
# ---------------------------------------------------------------------------

class BasicBlock1d(nn.Module):
    def __init__(self, cin, cout, stride, k, cond_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, k, padding=k // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.film = FiLM(cond_ch, cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                                       nn.BatchNorm1d(cout))

    def forward(self, x, cond):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.film(self.bn2(self.conv2(out)), cond)
        sc = x if self.short is None else self.short(x)
        return F.relu(out + sc, inplace=True)


class StatsPool(nn.Module):
    """Mean and standard deviation over time."""

    def forward(self, h):
        mu = h.mean(dim=2)
        var = (h * h).mean(dim=2) - mu * mu
        return torch.cat([mu, torch.sqrt(var.clamp_min(1e-5))], dim=1)


class Teacher(nn.Module):
    def __init__(self, widths=(128, 256, 384, 512), blocks=(2, 2, 2, 2), k: int = 5,
                 cond_width: int = 64, emb_dim: int = EMB_DIM, dropout: float = 0.1):
        super().__init__()
        self.widths = widths
        self.stem = nn.Sequential(nn.Conv1d(IN_CH, widths[0], 7, padding=3, bias=False),
                                  nn.BatchNorm1d(widths[0]), nn.ReLU(inplace=True))
        self.cond = ChallengeEncoder(cond_width, len(widths))
        self.stages = nn.ModuleList()
        cin = widths[0]
        for si, (w, nb) in enumerate(zip(widths, blocks)):
            stage = nn.ModuleList()
            for bi in range(nb):
                stride = 2 if (si > 0 and bi == 0) else 1
                stage.append(BasicBlock1d(cin, w, stride, k, cond_width))
                cin = w
            self.stages.append(stage)
        self.pool = StatsPool()
        feat = 2 * widths[-1]
        self.feat_dim = feat
        self.emb = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat, 512), nn.BatchNorm1d(512),
                                 nn.ReLU(inplace=True), nn.Linear(512, emb_dim), nn.BatchNorm1d(emb_dim))
        self.hand_head = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(inplace=True), nn.Linear(256, N_HAND))

    def feature_map(self, x):
        conds = self.cond(x[:, SIG_CH:SIG_CH + 1])
        h = self.stem(x)
        for si, stage in enumerate(self.stages):
            for blk in stage:
                h = blk(h, conds[si])
        return h

    def forward(self, x, return_all: bool = False):
        fmap = self.feature_map(x)
        feat = self.pool(fmap)
        emb_raw = self.emb(feat)
        out = {"embedding": l2_normalize(emb_raw.float()), "hand": self.hand_head(feat)}
        if return_all:
            out.update(feature_map=fmap, features=feat, embedding_raw=emb_raw)
        return out


# ---------------------------------------------------------------------------
# Student: depthwise-separable 1D CNN with FiLM (100k-300k parameters)
# ---------------------------------------------------------------------------

class DSBlock(nn.Module):
    """Inverted residual: pw expand -> dw conv -> pw project -> FiLM (+ residual)."""

    def __init__(self, cin, cout, stride, k, expand, cond_ch):
        super().__init__()
        mid = cin * expand
        self.pw1 = nn.Conv1d(cin, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(mid)
        self.dw = nn.Conv1d(mid, mid, k, stride=stride, padding=k // 2, groups=mid, bias=False)
        self.bn2 = nn.BatchNorm1d(mid)
        self.pw2 = nn.Conv1d(mid, cout, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(cout)
        self.film = FiLM(cond_ch, cout)
        self.residual = stride == 1 and cin == cout

    def forward(self, x, cond):
        h = F.relu(self.bn1(self.pw1(x)))
        h = F.relu(self.bn2(self.dw(h)))
        h = self.film(self.bn3(self.pw2(h)), cond)
        return x + h if self.residual else h


class Student(nn.Module):
    # (out_channels, stride, expand) per block; strides give T = 256 -> 16
    DEFAULT_PLAN = ((48, 2, 2), (48, 1, 2), (64, 2, 2), (64, 1, 2), (96, 2, 2), (96, 1, 2),
                    (128, 2, 2))

    def __init__(self, plan=DEFAULT_PLAN, stem: int = 32, k: int = 7, cond_width: int = 16,
                 emb_dim: int = EMB_DIM):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(IN_CH, stem, 5, padding=2, bias=False),
                                  nn.BatchNorm1d(stem), nn.ReLU())
        n_down = sum(1 for _, s, _ in plan if s == 2)
        self.cond_in = nn.Sequential(nn.Conv1d(1, cond_width, 9, padding=4), nn.ReLU())
        self.cond_down = nn.ModuleList([
            nn.Sequential(nn.Conv1d(cond_width, cond_width, 5, stride=2, padding=2, groups=1), nn.ReLU())
            for _ in range(n_down)])
        self.blocks = nn.ModuleList()
        self.block_level = []
        cin, level = stem, 0
        for cout, stride, expand in plan:
            if stride == 2:
                level += 1
            self.blocks.append(DSBlock(cin, cout, stride, k, expand, cond_width))
            self.block_level.append(level)
            cin = cout
        feat = 2 * cin
        self.feat_dim = feat
        self.emb = nn.Sequential(nn.Linear(feat, emb_dim), nn.BatchNorm1d(emb_dim))
        self.hand_head = nn.Sequential(nn.Linear(feat, 32), nn.ReLU(), nn.Linear(32, N_HAND))

    def forward_features(self, x):
        c = self.cond_in(x[:, SIG_CH:SIG_CH + 1])
        conds = [c]
        for d in self.cond_down:
            c = d(c)
            conds.append(c)
        h = self.stem(x)
        for blk, lvl in zip(self.blocks, self.block_level):
            h = blk(h, conds[lvl])
        # mean + max pooling over time
        return torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)

    def forward(self, x, return_all: bool = False):
        feat = self.forward_features(x)
        emb_raw = self.emb(feat)
        out = {"embedding": l2_normalize(emb_raw.float()), "hand": self.hand_head(feat)}
        if return_all:
            out.update(features=feat, embedding_raw=emb_raw)
        return out


class ExportWrapper(nn.Module):
    """Tuple outputs (embedding, hand) for tracing and Core ML conversion."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["embedding"], out["hand"]


# ---------------------------------------------------------------------------
# Heads used in training only
# ---------------------------------------------------------------------------

class ArcFace(nn.Module):
    """Additive angular margin softmax (ArcFace). Labels < 0 are ignored."""

    def __init__(self, emb_dim: int, n_classes: int, margin: float = 0.3, scale: float = 32.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, emb_dim) * 0.01)
        self.margin = margin
        self.scale = scale

    def set_margin(self, m: float):
        self.margin = m

    def logits(self, emb: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        w = F.normalize(self.weight.float(), dim=1)
        cos = (emb.float() @ w.t()).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        if labels is None or self.margin == 0:
            return self.scale * cos
        m = self.margin
        valid = labels >= 0
        lab = labels.clamp_min(0)
        cos_t = cos.gather(1, lab[:, None]).squeeze(1)
        sin_t = torch.sqrt(1.0 - cos_t * cos_t)
        phi = cos_t * math.cos(m) - sin_t * math.sin(m)
        # keep the target logit monotonic in theta beyond pi - m
        phi = torch.where(cos_t > math.cos(math.pi - m), phi, cos_t - math.sin(math.pi - m) * m)
        new_t = torch.where(valid, phi, cos_t)
        cos = cos.scatter(1, lab[:, None], new_t[:, None])
        return self.scale * cos

    def forward(self, emb, labels):
        logits = self.logits(emb, labels)
        if not torch.any(labels >= 0):
            return logits.sum() * 0.0, logits
        return F.cross_entropy(logits, labels, ignore_index=-1), logits


class ReconHead(nn.Module):
    """Masked-patch reconstruction head on a feature map with T/patch steps."""

    def __init__(self, in_ch: int, patch: int, out_ch: int = SIG_CH, t: int = 256):
        super().__init__()
        self.patch = patch
        self.out_ch = out_ch
        self.t = t
        self.proj = nn.Sequential(nn.Conv1d(in_ch, in_ch, 3, padding=1), nn.GELU(),
                                  nn.Conv1d(in_ch, out_ch * patch, 1))

    def forward(self, fmap):
        b, _, steps = fmap.shape
        y = self.proj(fmap)                                  # [B, out*patch, steps]
        y = y.view(b, self.out_ch, self.patch, steps)        # [B, C, P, S]
        y = y.permute(0, 1, 3, 2).reshape(b, self.out_ch, steps * self.patch)
        if y.shape[-1] != self.t:
            y = F.interpolate(y, size=self.t, mode="linear", align_corners=False)
        return y


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, out_dim))

    def forward(self, h):
        return F.normalize(self.net(h).float(), dim=1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def build_model(kind: str, **kw) -> nn.Module:
    if kind == "teacher":
        return Teacher(**kw)
    if kind == "student":
        return Student(**kw)
    if kind == "teacher_small":
        return Teacher(widths=(64, 128, 192, 256), **kw)
    if kind in ("mantis_pretrained", "mantis_scratch"):
        from .foundation import MantisTeacher   # needs the mantis-tsfm package
        return MantisTeacher(pretrained=kind == "mantis_pretrained", **kw)
    raise ValueError(kind)
