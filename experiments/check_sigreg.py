"""Compare src/conserve/jepa/sigreg.py with the official LeWorldModel SIGReg.

    python experiments/check_sigreg.py

`OfficialSIGReg` below is a transcription, read by eye, of `class SIGReg` in
`module.py` of github.com/lucas-maes/le-wm at commit
8edfeb336732b5f3ce7b8b210d0ba370a09e2cac (lines 10-36). It is not imported from
that repository. Its config (config/train/lewm.yaml at the same commit): knots 17,
num_proj 1024, loss weight 0.09 (the paper text says 0.1).

Prints the mean of both statistics over many random projections on several
embedding distributions, and their ratio. Diagnostic only; writes nothing to the ledger.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from conserve.jepa.sigreg import sigreg  # noqa: E402


class OfficialSIGReg(torch.nn.Module):
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):  # proj: (T, B, D)
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--b", type=int, default=256)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--reps", type=int, default=200)
    args = p.parse_args()
    torch.manual_seed(0)
    T, B, D = 3, args.b, args.d
    dists = {
        "gaussian": lambda: torch.randn(T, B, D),
        "scaled x0.5": lambda: 0.5 * torch.randn(T, B, D),
        "shifted +1": lambda: torch.randn(T, B, D) + 1.0,
        "uniform": lambda: (torch.rand(T, B, D) - 0.5) * 12 ** 0.5,
        "collapsed rank 4": lambda: torch.randn(T, B, 4) @ torch.randn(4, D) / 2.0,
    }
    official = OfficialSIGReg(knots=17, num_proj=1024)
    print(f"{'distribution':18s} {'official(1024)':>15s} {'ours(1024)':>11s} {'ours/official':>14s}"
          f"   (both: T={T} time steps of a batch of {B})")
    for name, make in dists.items():
        off, ours = [], []
        for _ in range(args.reps):
            x = make()
            off.append(official(x).item())
            ours.append(sigreg(x).item())
        o, m = sum(off) / len(off), sum(ours) / len(ours)
        print(f"{name:18s} {o:15.4f} {m:11.4f} {m / max(o, 1e-12):14.2f}")


if __name__ == "__main__":
    main()
