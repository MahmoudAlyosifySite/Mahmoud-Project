"""A Gaussianity regulariser in the style of SIGReg (LeJEPA, LeWorldModel).

Re-implemented from the papers' description, not copied from their code: project
the batch of embeddings on random unit directions and, for each direction,
measure the Epps-Pulley distance between the empirical characteristic function
and that of a standard normal.

Before reporting any number that depends on it, check this against the official
implementation, or use the official LeWM code directly (see docs/roadmap.md,
Stage D).

Known differences from LeWorldModel as its paper describes it (arXiv 2603.19312,
Section 3.1 and Appendix A): LeWM uses 1024 projections, weight 0.1, and
trapezoid nodes on [0.2, 4]. This file defaults to 256 projections and 17 nodes
on [0, 3].
"""
from __future__ import annotations

import torch


def sigreg(z: torch.Tensor, num_proj: int = 256, knots: int = 17, t_max: float = 3.0) -> torch.Tensor:
    """z: (batch, dim). Near zero when z looks like N(0, I); large when it is collapsed or lopsided."""
    b, d = z.shape
    directions = torch.randn(d, num_proj, device=z.device, dtype=z.dtype)
    directions = directions / directions.norm(dim=0, keepdim=True)
    t = torch.linspace(0.0, t_max, knots, device=z.device, dtype=z.dtype)
    target = torch.exp(-0.5 * t**2)  # characteristic function of N(0, 1)
    xt = (z @ directions).unsqueeze(-1) * t  # (batch, proj, knots)
    err = (xt.cos().mean(0) - target) ** 2 + xt.sin().mean(0) ** 2
    dt = t_max / (knots - 1)
    weights = torch.full((knots,), 2.0 * dt, device=z.device, dtype=z.dtype)
    weights[0] = weights[-1] = dt  # trapezoid rule on [0, t_max], doubled for the symmetric half
    return ((err * (weights * target)).sum(-1) * b).mean()
