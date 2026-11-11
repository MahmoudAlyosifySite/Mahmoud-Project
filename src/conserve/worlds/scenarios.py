"""Three-way scenarios: possible, improbable-but-lawful, impossible.

Each family pairs one kind of violation with a lawful control that is just as
unusual. A score that measures lawfulness should fire on the impossible clips
and stay quiet on the improbable ones.

family        impossible                         improbable (lawful)
energy        a ball changes speed in mid-flight one ball is far faster than any in training
permanence    a ball vanishes                    same
continuity    a ball teleports                   same
stochastic    a ball gains speed at a rough wall a rough-wall bounce at a rare angle
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .billiards import BilliardsConfig, Intervention, Trajectory, sample_initial, simulate

FAMILIES = ("energy", "permanence", "continuity", "stochastic")
CONDITIONS = ("possible", "improbable", "impossible")


@dataclass(frozen=True)
class ScenarioConfig:
    T: int = 120
    t_event_low: int = 40
    t_event_high: int = 80
    ball_counts: tuple[int, ...] = (2, 3)
    improbable_scale: float = 3.0  # speed multiplier of the odd ball
    energy_factors: tuple[float, ...] = (0.5, 1.6)
    teleport_min_dist: float = 0.3
    gain_factor: float = 1.6
    rare_sigmas: tuple[float, float] = (3.0, 4.0)
    rough_jitter_deg: float = 12.0
    settle: int = 20  # frames required after a bounce-triggered event


def world_for(family: str, sc: ScenarioConfig, base: BilliardsConfig = BilliardsConfig()) -> BilliardsConfig:
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    jitter = sc.rough_jitter_deg if family == "stochastic" else 0.0
    return replace(base, wall_jitter_deg=jitter)


def make_trajectory(
    family: str,
    condition: str,
    sc: ScenarioConfig,
    rng: np.random.Generator,
    base: BilliardsConfig = BilliardsConfig(),
) -> Trajectory:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    cfg = world_for(family, sc, base)
    for _ in range(200):
        n_balls = int(rng.choice(sc.ball_counts))
        pos, vel, alive = sample_initial(cfg, rng, n_balls)
        ball = int(rng.choice(np.flatnonzero(alive)))
        t0 = int(rng.integers(sc.t_event_low, sc.t_event_high))
        interventions: tuple[Intervention, ...] = ()
        if condition == "improbable":
            if family == "stochastic":
                sigma = np.deg2rad(sc.rough_jitter_deg)
                angle = rng.choice([-1.0, 1.0]) * rng.uniform(*sc.rare_sigmas) * sigma
                interventions = (Intervention("rare_bounce", t0, ball, float(angle)),)
            else:
                vel[ball] = vel[ball] * sc.improbable_scale
        elif condition == "impossible":
            if family == "energy":
                interventions = (Intervention("energy_jump", t0, ball, float(rng.choice(sc.energy_factors))),)
            elif family == "permanence":
                interventions = (Intervention("vanish", t0, ball),)
            elif family == "continuity":
                interventions = (Intervention("teleport", t0, ball),)
            else:
                interventions = (Intervention("bounce_gain", t0, ball, sc.gain_factor),)
        traj = simulate(
            cfg, sc.T, rng, init=(pos, vel, alive), interventions=interventions,
            teleport_min_dist=sc.teleport_min_dist,
        )
        scripted = bool(interventions)
        if scripted and (traj.meta["unfired"] or traj.t_event > sc.T - sc.settle):
            continue  # the armed ball never reached a wall in time: draw again
        traj.meta.update(family=family, condition=condition, ball=ball if scripted or condition == "improbable" else -1)
        return traj
    raise RuntimeError(f"could not build a {condition} clip for family {family}")


def make_dataset(
    family: str,
    condition: str,
    n: int,
    sc: ScenarioConfig,
    seed: int,
    base: BilliardsConfig = BilliardsConfig(),
) -> list[Trajectory]:
    rng = np.random.default_rng(seed)
    return [make_trajectory(family, condition, sc, rng, base) for _ in range(n)]
