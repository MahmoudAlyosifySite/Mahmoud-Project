"""Billiards: equal-mass discs in the unit box.

Lawful dynamics conserve two things exactly: the number of balls and the total
kinetic energy (walls and ball-ball collisions are elastic). Rough walls make
the outcome of a bounce random but still conserve both. Everything else in this
repository is built on that fact.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class BilliardsConfig:
    max_balls: int = 3
    radius: float = 0.06
    speed_low: float = 0.006  # box lengths per step
    speed_high: float = 0.014
    ball_collisions: bool = True
    wall_jitter_deg: float = 0.0  # std of the reflection-angle noise; 0 means specular walls


@dataclass(frozen=True)
class Intervention:
    """A scripted change to the state.

    kind:
      "energy_jump"  multiply the ball's velocity by `value` at step t          (unlawful)
      "vanish"       remove the ball at step t                                   (unlawful)
      "teleport"     move the ball to a free position at step t                  (unlawful)
      "bounce_gain"  multiply the ball's speed by `value` at its next wall hit   (unlawful)
      "rare_bounce"  rotate the reflection by `value` radians at its next hit    (lawful, rare)
    """

    kind: str
    t: int
    ball: int
    value: float = 1.0


@dataclass
class Trajectory:
    pos: np.ndarray  # (T, N, 2), zero for absent balls
    vel: np.ndarray  # (T, N, 2), zero for absent balls
    alive: np.ndarray  # (T, N) bool
    wall_hit: np.ndarray  # (T, N) bool: the ball hit a wall during the step that produced frame t
    t_event: int = -1  # step at which a scripted intervention took effect, -1 if none
    meta: dict = field(default_factory=dict)

    @property
    def kinetic_energy(self) -> np.ndarray:
        return 0.5 * (self.alive * (self.vel**2).sum(-1)).sum(-1)

    @property
    def count(self) -> np.ndarray:
        return self.alive.sum(-1)


def sample_initial(cfg: BilliardsConfig, rng: np.random.Generator, n_balls: int):
    """Random non-overlapping start with `n_balls` of the `max_balls` slots filled."""
    n = cfg.max_balls
    if not 1 <= n_balls <= n:
        raise ValueError(f"n_balls must be in [1, {n}], got {n_balls}")
    alive = np.zeros(n, bool)
    alive[rng.choice(n, size=n_balls, replace=False)] = True
    pos = np.zeros((n, 2))
    vel = np.zeros((n, 2))
    lo, hi = cfg.radius, 1.0 - cfg.radius
    placed: list[np.ndarray] = []
    for i in np.flatnonzero(alive):
        for _ in range(1000):
            p = rng.uniform(lo, hi, size=2)
            if all(np.linalg.norm(p - q) > 2.2 * cfg.radius for q in placed):
                break
        else:
            raise RuntimeError("could not place the balls without overlap")
        placed.append(p)
        pos[i] = p
        speed = rng.uniform(cfg.speed_low, cfg.speed_high)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        vel[i] = speed * np.array([np.cos(angle), np.sin(angle)])
    return pos, vel, alive


def _rotate_keep_inward(v: np.ndarray, delta: float, axis: int, inward: float) -> np.ndarray:
    """Rotate `v` by `delta` radians, keeping its speed and keeping it pointed into the box."""
    c, s = np.cos(delta), np.sin(delta)
    out = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
    speed = float(np.linalg.norm(v))
    floor = 0.1 * speed
    if out[axis] * inward < floor:
        other = 1 - axis
        sign = 1.0 if out[other] >= 0 else -1.0
        out[axis] = inward * floor
        out[other] = sign * np.sqrt(max(speed**2 - floor**2, 0.0))
    return out


def step(pos, vel, alive, cfg: BilliardsConfig, rng=None, forced=None):
    """Advance one step and return (pos, vel, wall_hit).

    With `rng=None` wall reflections are specular, which is the mean outcome on a
    rough wall. `forced` maps a ball index to ("angle", radians) or ("gain", factor)
    and is applied at that ball's wall hit during this step, if it has one.
    """
    pos = pos + vel * alive[:, None]
    vel = vel.copy()
    wall_hit = np.zeros(len(alive), bool)
    lo, hi = cfg.radius, 1.0 - cfg.radius
    for i in np.flatnonzero(alive):
        hit = None
        for axis in (0, 1):
            if pos[i, axis] < lo:
                pos[i, axis] = 2.0 * lo - pos[i, axis]
                vel[i, axis] = -vel[i, axis]
                hit = (axis, 1.0)
            elif pos[i, axis] > hi:
                pos[i, axis] = 2.0 * hi - pos[i, axis]
                vel[i, axis] = -vel[i, axis]
                hit = (axis, -1.0)
        if hit is None:
            continue
        wall_hit[i] = True
        action = forced.get(int(i)) if forced else None
        delta = 0.0
        if action is not None and action[0] == "angle":
            delta = float(action[1])
        elif rng is not None and cfg.wall_jitter_deg > 0:
            delta = float(rng.normal(0.0, np.deg2rad(cfg.wall_jitter_deg)))
        if delta != 0.0:
            vel[i] = _rotate_keep_inward(vel[i], delta, *hit)
        if action is not None and action[0] == "gain":
            vel[i] = vel[i] * float(action[1])
    pos = np.clip(pos, lo, hi) * alive[:, None] + pos * (~alive[:, None])
    if cfg.ball_collisions:
        idx = np.flatnonzero(alive)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                d = pos[i] - pos[j]
                dist = float(np.linalg.norm(d))
                if 0.0 < dist < 2.0 * cfg.radius:
                    normal = d / dist
                    closing = float((vel[i] - vel[j]) @ normal)
                    if closing < 0.0:  # equal masses: swap the normal components
                        vel[i] = vel[i] - closing * normal
                        vel[j] = vel[j] + closing * normal
    return pos, vel, wall_hit


def _free_position(pos, alive, ball, cfg, rng, min_dist):
    lo, hi = cfg.radius, 1.0 - cfg.radius
    others = [pos[j] for j in np.flatnonzero(alive) if j != ball]
    for _ in range(2000):
        p = rng.uniform(lo, hi, size=2)
        if np.linalg.norm(p - pos[ball]) < min_dist:
            continue
        if all(np.linalg.norm(p - q) > 2.2 * cfg.radius for q in others):
            return p
    raise RuntimeError("could not find a free position for the teleport")


def simulate(
    cfg: BilliardsConfig,
    T: int,
    rng: np.random.Generator,
    n_balls: int | None = None,
    init=None,
    interventions: tuple[Intervention, ...] = (),
    teleport_min_dist: float = 0.3,
) -> Trajectory:
    """Roll the world forward for T frames, applying scripted interventions.

    An intervention at step t changes the state after the lawful step that
    produced frame t, so frames before t are untouched.
    """
    if init is None:
        pos, vel, alive = sample_initial(cfg, rng, cfg.max_balls if n_balls is None else n_balls)
    else:
        pos, vel, alive = (np.array(x, copy=True) for x in init)
    n = cfg.max_balls
    P = np.zeros((T, n, 2))
    V = np.zeros((T, n, 2))
    A = np.zeros((T, n), bool)
    H = np.zeros((T, n), bool)
    P[0], V[0], A[0] = pos * alive[:, None], vel * alive[:, None], alive
    by_time: dict[int, list[Intervention]] = {}
    for iv in interventions:
        by_time.setdefault(iv.t, []).append(iv)
    armed: dict[int, Intervention] = {}  # waiting for that ball's next wall hit
    t_event = -1
    for t in range(1, T):
        forced = {
            b: ("gain" if iv.kind == "bounce_gain" else "angle", iv.value) for b, iv in armed.items()
        }
        pos, vel, hit = step(pos, vel, alive, cfg, rng, forced)
        for b in list(armed):
            if hit[b]:
                t_event = t
                del armed[b]
        for iv in by_time.get(t, ()):
            if not alive[iv.ball]:
                raise ValueError(f"intervention targets absent ball {iv.ball}")
            if iv.kind == "energy_jump":
                vel[iv.ball] = vel[iv.ball] * iv.value
                t_event = t
            elif iv.kind == "vanish":
                alive = alive.copy()
                alive[iv.ball] = False
                t_event = t
            elif iv.kind == "teleport":
                pos[iv.ball] = _free_position(pos, alive, iv.ball, cfg, rng, teleport_min_dist)
                t_event = t
            elif iv.kind in ("bounce_gain", "rare_bounce"):
                armed[iv.ball] = iv
            else:
                raise ValueError(f"unknown intervention kind {iv.kind!r}")
        P[t], V[t], A[t], H[t] = pos * alive[:, None], vel * alive[:, None], alive, hit
    return Trajectory(P, V, A, H, t_event, meta={"unfired": bool(armed)})
