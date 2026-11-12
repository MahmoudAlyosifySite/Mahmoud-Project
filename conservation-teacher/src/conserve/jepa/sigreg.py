"""A Gaussianity regulariser: SIGReg (LeJEPA, LeWorldModel).

Project a batch of embeddings on random unit directions and, for each direction,
measure the Epps-Pulley distance between the empirical characteristic function
and that of a standard normal.

Checked against the official LeWorldModel `SIGReg` (github.com/lucas-maes/le-wm,
`module.py`, commit 8edfeb3) on 2026-11-11 with `experiments/check_sigreg.py`.
It is a re-implementation, not the official file, and it now follows it: 17
trapezoid knots on [0, 3] with the Gaussian window, 1024 projections, the
statistic multiplied by the batch size of one time step, and the mean taken over
time steps and projections. An input of shape (T, B, D) is read as T time steps
of a batch of B; a 2-D input (B, D) is one batch.

Before 2026-11-11 (no Stage C number existed) this file pooled the three frames of
a triple into one batch of 3B, which made the statistic about three times larger
on non-Gaussian embeddings than the official one, and used 256 projections.
"""
from __future__ import annotations

import torch


def sigreg(z: torch.Tensor, num_proj: int = 1024, knots: int = 17, t_max: float = 3.0) -> torch.Tensor:
    """z: (batch, dim) or (time, batch, dim). Near zero when z looks like N(0, I); large when collapsed or lopsided."""
    if z.dim() == 2:
        z = z.unsqueeze(0)
    b, d = z.shape[-2], z.shape[-1]
    directions = torch.randn(d, num_proj, device=z.device, dtype=z.dtype)
    directions = directions / directions.norm(dim=0, keepdim=True)
    t = torch.linspace(0.0, t_max, knots, device=z.device, dtype=z.dtype)
    target = torch.exp(-0.5 * t**2)  # characteristic function of N(0, 1)
    xt = (z @ directions).unsqueeze(-1) * t  # (time, batch, proj, knots)
    err = (xt.cos().mean(-3) - target) ** 2 + xt.sin().mean(-3) ** 2  # (time, proj, knots)
    dt = t_max / (knots - 1)
    weights = torch.full((knots,), 2.0 * dt, device=z.device, dtype=z.dtype)
    weights[0] = weights[-1] = dt  # trapezoid rule on [0, t_max], doubled for the symmetric half
    return ((err * (weights * target)).sum(-1) * b).mean()
