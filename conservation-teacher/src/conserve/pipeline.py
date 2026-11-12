"""Fit on lawful clips, score any clip, compare scores. Shared by every experiment.

Two layers:
  fit_latents / score_latent      work on latent sequences from any encoder
  fit_world / evaluate_family     wrap them for the synthetic billiards latents
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .events import clean_mask, fit_event_threshold
from .features import FeatureMap
from .invariants import InvariantHead, fit_invariants, long_lag_slowness
from .latents import Mixer, frame_latent, phase_space
from .metrics import auroc
from .predictors import ResidualMLP
from .scores import level_shift, oracle_errors, pe_scores
from .worlds import CONDITIONS, FAMILIES, BilliardsConfig, ScenarioConfig, Trajectory, make_dataset, world_for

INV_SCORES = ("inv", "inv_rel")
PE_SCORES = ("pe_max", "pe_mean", "pe_rel")
COMPARISONS = (("impossible", "possible"), ("impossible", "improbable"), ("improbable", "possible"))


@dataclass(frozen=True)
class PilotConfig:
    n_train: int = 300
    n_val: int = 100
    n_test: int = 200
    alpha: float | None = None  # None: raw frame latents. A number: mixed latents with that nonlinearity.
    noise: float = 0.0  # per-frame latent noise, in units of each latent dimension's spread
    span: int = 1  # baseline, in steps, of the velocity estimate used by the invariants
    pca_dim: int | None = None
    lag_low: int = 30
    lag_high: int = 90
    max_invariants: int = 16
    min_correlation: float = 0.9  # long-lag autocorrelation an invariant must keep on fresh lawful clips
    window: int = 12
    hidden: tuple[int, ...] = (128, 128)
    predictor_iters: int = 150

    @property
    def lag_range(self) -> tuple[int, int]:
        return (self.lag_low, self.lag_high)

    @property
    def max_slowness(self) -> float:
        return 2.0 * (1.0 - self.min_correlation)


@dataclass
class FittedLatents:
    threshold: float
    head_all: InvariantHead  # every candidate, with its validation slowness
    head: InvariantHead  # the candidates that passed validation
    predictor: ResidualMLP | None
    pe_floor: float


def fit_latents(
    zs: list[np.ndarray],
    zs_val: list[np.ndarray],
    pc: PilotConfig,
    seed: int,
    with_predictor: bool = True,
) -> FittedLatents:
    """Fit the event threshold, the invariants and (optionally) a predictor on lawful latents."""
    rng = np.random.default_rng([seed, 0])
    threshold = fit_event_threshold(zs)
    us = [phase_space(z, pc.span) for z in zs]
    cleans = [clean_mask(z, threshold, pc.span) for z in zs]
    us_val = [phase_space(z, pc.span) for z in zs_val]
    cleans_val = [clean_mask(z, threshold, pc.span) for z in zs_val]
    feat = FeatureMap.fit(np.concatenate([u[c] for u, c in zip(us, cleans)]), pc.pca_dim)
    head_all = fit_invariants(us, cleans, feat, rng, pc.lag_range, max_k=pc.max_invariants)
    slowness_val = long_lag_slowness(head_all, us_val, cleans_val, rng, pc.lag_range)
    head_all = replace(head_all, slowness_val=slowness_val)
    head = head_all.select(slowness_val <= pc.max_slowness)
    predictor, pe_floor = None, 0.0
    if with_predictor:
        u1 = [phase_space(z, 1) for z in zs]
        predictor = ResidualMLP(pc.hidden, pc.predictor_iters, seed).fit(u1, zs)
        pe_floor = float(np.median(np.concatenate([predictor.errors(u, z) for u, z in zip(u1[:50], zs[:50])])))
    return FittedLatents(threshold, head_all, head, predictor, pe_floor)


def score_latent(fl: FittedLatents, z: np.ndarray, pc: PilotConfig, errors: np.ndarray | None = None) -> dict:
    """Score one clip's latents. `errors` are the encoder's own prediction errors, if it has a predictor."""
    out = level_shift(fl.head(phase_space(z, pc.span)), clean_mask(z, fl.threshold, pc.span), pc.window)
    if errors is None and fl.predictor is not None:
        errors = fl.predictor.errors(phase_space(z, 1), z)
    if errors is not None:
        out.update(pe_scores(errors, fl.pe_floor))
    return out


def compare(scores: dict[str, dict[str, np.ndarray]], names) -> dict[str, dict[str, float]]:
    """AUROC of every score for every pair of conditions."""
    return {s: {f"{a}_vs_{b}": auroc(scores[a][s], scores[b][s]) for a, b in COMPARISONS} for s in names}


# ---------------------------------------------------------------------------
# Synthetic billiards latents
# ---------------------------------------------------------------------------

SCORES = INV_SCORES + PE_SCORES + ("oracle_pe",)


@dataclass
class FittedWorld:
    cfg: BilliardsConfig
    mixer: Mixer | None
    fit: FittedLatents

    @property
    def head(self) -> InvariantHead:
        return self.fit.head

    @property
    def head_all(self) -> InvariantHead:
        return self.fit.head_all


def _latent(traj: Trajectory, mixer: Mixer | None, rng: np.random.Generator) -> np.ndarray:
    z = frame_latent(traj)
    return z if mixer is None else mixer(z, rng)


def fit_world(family: str, sc: ScenarioConfig, pc: PilotConfig, seed: int) -> FittedWorld:
    """Fit everything on lawful, ordinary clips of the family's world. No impossible clip is seen."""
    rng = np.random.default_rng([seed, 2])
    train = make_dataset(family, "possible", pc.n_train, sc, seed * 1000 + 1)
    val = make_dataset(family, "possible", pc.n_val, sc, seed * 1000 + 2)
    mixer = None
    if pc.alpha is not None:
        raw = [frame_latent(c) for c in train]
        mixer = Mixer(raw[0].shape[1], pc.alpha, pc.noise, seed).fit(raw)
    zs = [_latent(c, mixer, rng) for c in train]
    zs_val = [_latent(c, mixer, rng) for c in val]
    return FittedWorld(world_for(family, sc), mixer, fit_latents(zs, zs_val, pc, seed))


def score_clip(fw: FittedWorld, traj: Trajectory, pc: PilotConfig, rng: np.random.Generator) -> dict:
    out = score_latent(fw.fit, _latent(traj, fw.mixer, rng), pc)
    out["oracle_pe"] = float(oracle_errors(traj, fw.cfg).max())
    return out


def evaluate_family(fw: FittedWorld, family: str, sc: ScenarioConfig, pc: PilotConfig, seed: int) -> dict:
    """Score fresh clips of all three conditions and return every AUROC."""
    scores: dict[str, dict[str, np.ndarray]] = {}
    fi = FAMILIES.index(family)
    for ci, condition in enumerate(CONDITIONS):
        clips = make_dataset(family, condition, pc.n_test, sc, seed * 1000 + 10 + 10 * fi + ci)
        rng = np.random.default_rng([seed, 1, fi, ci])
        rows = [score_clip(fw, c, pc, rng) for c in clips]
        scores[condition] = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    return {"aurocs": compare(scores, SCORES), "scores": scores}


def invariant_values(fw: FittedWorld, clips: list[Trajectory], pc: PilotConfig, rng: np.random.Generator):
    """For every clip: (values of the kept invariants at t = 1..T-1, clean mask)."""
    out = []
    for c in clips:
        z = _latent(c, fw.mixer, rng)
        out.append((fw.head(phase_space(z, pc.span)), clean_mask(z, fw.fit.threshold, pc.span)))
    return out


def ground_truth_r2(fw: FittedWorld, clips: list[Trajectory], pc: PilotConfig, rng: np.random.Generator) -> dict:
    """How well the kept invariants explain the true energy and ball count (diagnostic only)."""
    if fw.head.k == 0:
        return {"energy": 0.0, "count": 0.0}
    rows, energy, count = [], [], []
    for c, (values, clean) in zip(clips, invariant_values(fw, clips, pc, rng)):
        rows.append(values[clean])
        energy.append(c.kinetic_energy[1:][clean])
        count.append(c.count[1:][clean])
    x = np.concatenate(rows)
    x = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    out = {}
    for name, target in (("energy", np.concatenate(energy)), ("count", np.concatenate(count).astype(float))):
        coef, *_ = np.linalg.lstsq(x, target, rcond=None)
        resid = target - x @ coef
        out[name] = float(1.0 - resid.var() / max(target.var(), 1e-300))
    return out
