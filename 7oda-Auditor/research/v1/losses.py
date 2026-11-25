"""Losses for v1 training: self-supervised (masked reconstruction, contrastive
views) and distillation (embedding cosine, relational distances, hand logits).
ArcFace lives in ``models.ArcFace``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import SIGNAL_CHANNELS, T


# ---------------------------------------------------------------------------
# Masked reconstruction (MOMENT-style: mask ~30% of patches, reconstruct them)
# ---------------------------------------------------------------------------

def patch_mask(batch: int, patch: int = 8, ratio: float = 0.3, t: int = T, device="cpu",
               generator: torch.Generator | None = None) -> torch.Tensor:
    """Boolean mask ``[B, T]``: True on time steps of masked patches.

    Exactly ``round(ratio * n_patches)`` patches per sample.
    """
    n_patch = t // patch
    k = max(1, int(round(ratio * n_patch)))
    scores = torch.rand(batch, n_patch, device=device, generator=generator)
    idx = scores.argsort(dim=1)[:, :k]
    m = torch.zeros(batch, n_patch, dtype=torch.bool, device=device)
    m.scatter_(1, idx, True)
    return m.repeat_interleave(patch, dim=1)


def apply_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero the signal channels (0-8) at masked time steps; the template stays."""
    x = x.clone()
    x[:, :SIGNAL_CHANNELS] = x[:, :SIGNAL_CHANNELS] * (~mask)[:, None, :].to(x.dtype)
    return x


def masked_recon_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE over the signal channels at masked time steps only."""
    m = mask[:, None, :].to(pred.dtype)
    diff = (pred.float() - target[:, :SIGNAL_CHANNELS].float()) ** 2 * m
    denom = m.sum() * SIGNAL_CHANNELS
    return diff.sum() / denom.clamp_min(1.0)


# ---------------------------------------------------------------------------
# Contrastive views of the same recording (SimCLR / Mantis-style NT-Xent)
# ---------------------------------------------------------------------------

def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Normalized temperature-scaled cross entropy for two views ``[B, D]``."""
    z1 = F.normalize(z1.float(), dim=1)
    z2 = F.normalize(z2.float(), dim=1)
    b = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.t() / temperature
    sim.fill_diagonal_(float("-inf"))
    target = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)]).to(z.device)
    return F.cross_entropy(sim, target)


# ---------------------------------------------------------------------------
# Distillation
# ---------------------------------------------------------------------------

def cosine_distill(student_emb: torch.Tensor, teacher_emb: torch.Tensor) -> torch.Tensor:
    """Mean (1 - cos) between student and teacher embeddings."""
    return (1.0 - F.cosine_similarity(student_emb.float(), teacher_emb.float(), dim=1)).mean()


def relational_distill(student_emb: torch.Tensor, teacher_emb: torch.Tensor) -> torch.Tensor:
    """Distance-wise relational loss (RKD-D): match pairwise distances, each
    normalized by its batch mean, with a smooth L1 penalty."""
    s = student_emb.float()
    t = teacher_emb.float()
    ds = torch.cdist(s, s)
    dt = torch.cdist(t, t)
    off = ~torch.eye(s.shape[0], dtype=torch.bool, device=s.device)
    ds = ds[off]
    dt = dt[off]
    ds = ds / ds.mean().clamp_min(1e-6)
    dt = dt / dt.mean().clamp_min(1e-6)
    return F.smooth_l1_loss(ds, dt)


def similarity_distill(student_emb: torch.Tensor, teacher_emb: torch.Tensor) -> torch.Tensor:
    """MSE between the in-batch cosine-similarity matrices (scale-free)."""
    s = F.normalize(student_emb.float(), dim=1)
    t = F.normalize(teacher_emb.float(), dim=1)
    return F.mse_loss(s @ s.t(), t @ t.t())


def logit_distill(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temp: float = 2.0) -> torch.Tensor:
    """KL(teacher || student) on temperature-softened logits, times temp^2."""
    ps = F.log_softmax(student_logits.float() / temp, dim=1)
    pt = F.softmax(teacher_logits.float() / temp, dim=1)
    return F.kl_div(ps, pt, reduction="batchmean") * temp * temp
