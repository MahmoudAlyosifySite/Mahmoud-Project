"""Rigid phone + hand contacts: 6-DOF mass-spring-damper with cubic contact springs.

State q = (x, y, z, rx, ry, rz): small displacement and rotation of the phone
about its resting pose in the hand, device axes, about the phone's centre.

    M q'' + C q' + K q + D^T f_nl(D q) = b u(t)

M = phone mass/inertia + sum_c m_c J_c^T J_c       (hand mass coupled at each contact)
K = sum_c J_c^T (kn_c n n^T + kt_c (I - n n^T)) J_c  (C likewise with cn, ct)
J_c = [I, -[r_c]x] maps q to the displacement of contact point r_c,
D   = rows n_c^T J_c (normal displacement of each contact),
f_nl = k3_c delta^3 (cubic stiffening of each normal spring),
b   = [a, r_act x a] for an actuator force along unit axis a at r_act.

The fast path discretises the linear part exactly (zero-order hold, matrix
exponential of the augmented system) at the simulation rate and treats the
cubic term explicitly (held over a step); the reference path is classic RK4
on the continuous equations, used by the tests to check the fast path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import DELTA_REF, Contacts, Phone


def skew(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, float)
    z = np.zeros(r.shape[:-1])
    return np.stack([np.stack([z, -r[..., 2], r[..., 1]], -1),
                     np.stack([r[..., 2], z, -r[..., 0]], -1),
                     np.stack([-r[..., 1], r[..., 0], z], -1)], -2)


def point_jacobian(r: np.ndarray) -> np.ndarray:
    """(..., 3, 6): displacement of point r for a small motion q of the body."""
    r = np.asarray(r, float)
    eye = np.broadcast_to(np.eye(3), r.shape[:-1] + (3, 3))
    return np.concatenate([eye, -skew(r)], -1)


def pad_contacts(cs: list[Contacts]) -> dict:
    """Stack per-trial contact sets, padding with inert contacts."""
    nc = max(len(c.kn) for c in cs)
    out = {}
    for f in Contacts.__dataclass_fields__:
        arrs = []
        for c in cs:
            a = getattr(c, f)
            pad = nc - a.shape[0]
            if pad:
                fill = np.tile([0.0, 0.0, 1.0], (pad, 1)) if f == "normal" else np.zeros((pad,) + a.shape[1:])
                a = np.concatenate([a, fill])
            arrs.append(a)
        out[f] = np.stack(arrs)
    return out


@dataclass
class System:
    M: np.ndarray      # (B, 6, 6)
    K: np.ndarray
    C: np.ndarray
    D: np.ndarray      # (B, nc, 6)
    k3: np.ndarray     # (B, nc)
    b: np.ndarray      # (6,)
    J_imu: np.ndarray  # (3, 6)


def build_system(cs: list[Contacts], phone: Phone) -> System:
    p = pad_contacts(cs)
    n = p["normal"]
    J = point_jacobian(p["pos"])                                    # (B, nc, 3, 6)
    nn = n[..., :, None] * n[..., None, :]
    eye = np.eye(3)
    Kc = p["kn"][..., None, None] * nn + p["kt"][..., None, None] * (eye - nn)
    Cc = p["cn"][..., None, None] * nn + p["ct"][..., None, None] * (eye - nn)
    K = np.einsum("bcia,bcij,bcjd->bad", J, Kc, J)
    C = np.einsum("bcia,bcij,bcjd->bad", J, Cc, J)
    M = np.einsum("bc,bcia,bcid->bad", p["mass"], J, J)
    M[:, [0, 1, 2], [0, 1, 2]] += phone.mass
    M[:, [3, 4, 5], [3, 4, 5]] += np.asarray(phone.inertia)
    D = np.einsum("bci,bcid->bcd", n, J)
    a = np.asarray(phone.act_axis, float)
    b = np.concatenate([a, np.cross(np.asarray(phone.act_pos), a)])
    return System(M=M, K=K, C=C, D=D, k3=p["k3"], b=b, J_imu=point_jacobian(np.asarray(phone.imu_pos)))


def expm_batch(A: np.ndarray) -> np.ndarray:
    """Matrix exponential of a stack of matrices (scaling and squaring, Taylor 14).
    The scaling is chosen per matrix, so each result is independent of the rest of the stack."""
    norm = np.abs(A).sum(-2).max(-1)                                   # (B,) 1-norms
    s = np.maximum(0, np.ceil(np.log2(np.maximum(norm, 1e-300) / 0.25))).astype(int)
    As = A / (2.0 ** s)[:, None, None]
    n = A.shape[-1]
    eye = np.broadcast_to(np.eye(n), A.shape)
    E = eye + As / 14.0
    for k in range(13, 0, -1):
        E = eye + (As @ E) / k
    for j in range(int(s.max()) if s.size else 0):
        E = np.where((s > j)[:, None, None], E @ E, E)
    return E


@dataclass
class Discrete:
    Phi: np.ndarray    # (B, 12, 12 + nc)  [state | nonlinear force] -> next state
    Gu: np.ndarray     # (B, 12)
    Cz: np.ndarray     # (B, 6, 12 + nc)   outputs at the IMU: accel (m/s^2) and gyro (rad/s)
    du: np.ndarray     # (B, 6)
    D: np.ndarray      # (B, nc, 6)
    k3: np.ndarray     # (B, nc)
    nonlinear: bool


def discretize(sysm: System, h: float) -> Discrete:
    B, nc = sysm.k3.shape
    Minv = np.linalg.inv(sysm.M)
    A = np.zeros((B, 12, 12))
    A[:, :6, 6:] = np.eye(6)
    A[:, 6:, :6] = -Minv @ sysm.K
    A[:, 6:, 6:] = -Minv @ sysm.C
    Bu = np.zeros((B, 12))
    Bu[:, 6:] = Minv @ sysm.b
    Bn = np.zeros((B, 12, nc))
    Bn[:, 6:, :] = -Minv @ np.swapaxes(sysm.D, 1, 2)
    n = 12 + 1 + nc
    Aug = np.zeros((B, n, n))
    Aug[:, :12, :12] = A
    Aug[:, :12, 12] = Bu
    Aug[:, :12, 13:] = Bn
    E = expm_batch(Aug * h)
    Phi = np.concatenate([E[:, :12, :12], E[:, :12, 13:]], -1)
    Gu = E[:, :12, 12]
    # outputs: accel at the IMU = J_imu q'' ; gyro = rotational velocity
    JM = sysm.J_imu @ Minv                                            # (B, 3, 6)
    Cz = np.zeros((B, 6, 12 + nc))
    Cz[:, :3, :6] = -JM @ sysm.K
    Cz[:, :3, 6:12] = -JM @ sysm.C
    Cz[:, :3, 12:] = -JM @ np.swapaxes(sysm.D, 1, 2)
    Cz[:, 3:, 9:12] = np.eye(3)
    du = np.zeros((B, 6))
    du[:, :3] = JM @ sysm.b
    return Discrete(Phi=Phi, Gu=Gu, Cz=Cz, du=du, D=sysm.D, k3=sysm.k3, nonlinear=bool(np.any(sysm.k3 > 0)))


def _cubic(d: np.ndarray, k3: np.ndarray) -> np.ndarray:
    lim = d.dtype.type(4 * DELTA_REF)
    dc = np.clip(d, -lim, lim)
    return k3 * dc * dc * dc


def integrate(disc: Discrete, u_grid: np.ndarray, u_mid: np.ndarray, k_start: int = 0,
              out_dtype=np.float32) -> np.ndarray:
    """Time-step the discrete system. u_grid/u_mid: (T, B) force at grid points and
    step midpoints. Returns outputs (T, B, 6) at the grid points; zero before k_start
    (the state is at rest until the first actuator event)."""
    T, B = u_grid.shape
    nc = disc.k3.shape[1]
    y = np.zeros((T, B, 6), out_dtype)
    f32 = np.float32           # single precision is ample here and ~1.4x faster
    x = np.zeros((B, 12), f32)
    z = np.zeros((B, 12 + nc), f32)
    Phi, Gu, Cz, du, D6, k3 = (a.astype(f32) for a in (disc.Phi, disc.Gu, disc.Cz, disc.du, disc.D, disc.k3))
    u_grid = u_grid.astype(f32)
    u_mid = u_mid.astype(f32)
    for k in range(max(0, k_start), T):
        z[:, :12] = x
        if disc.nonlinear:
            z[:, 12:] = _cubic(np.einsum("bcj,bj->bc", D6, x[:, :6]), k3)
        y[k] = np.einsum("boi,bi->bo", Cz, z) + du * u_grid[k][:, None]
        x = np.einsum("bij,bj->bi", Phi, z) + Gu * u_mid[k][:, None]
    return y


def integrate_reference(sysm: System, u_fine: np.ndarray, h: float, out_every: int) -> np.ndarray:
    """RK4 on the continuous nonlinear equations for ONE trial (B = 1).

    u_fine: force sampled at spacing h/2 (so u at t, t+h/2, t+h is available).
    Returns outputs (6, n_out) at every `out_every`-th step (accel m/s^2, gyro rad/s).
    """
    M, K, C, D, k3, b = sysm.M[0], sysm.K[0], sysm.C[0], sysm.D[0], sysm.k3[0], sysm.b
    Minv = np.linalg.inv(M)
    JM = sysm.J_imu @ Minv

    def acc(q, v, u):
        fn = _cubic(D @ q, k3)
        return Minv @ (b * u - K @ q - C @ v - D.T @ fn), fn

    n_steps = (len(u_fine) - 1) // 2
    q = np.zeros(6)
    v = np.zeros(6)
    out = []
    for k in range(n_steps):
        u0, um, u1 = u_fine[2 * k], u_fine[2 * k + 1], u_fine[2 * k + 2]
        if k % out_every == 0:
            a0, fn = acc(q, v, u0)
            out.append(np.concatenate([JM @ (b * u0 - K @ q - C @ v - D.T @ fn), v[3:]]))
        k1q, (k1v, _) = v, acc(q, v, u0)
        k2q, (k2v, _) = v + 0.5 * h * k1v, acc(q + 0.5 * h * k1q, v + 0.5 * h * k1v, um)
        k3q, (k3v, _) = v + 0.5 * h * k2v, acc(q + 0.5 * h * k2q, v + 0.5 * h * k2v, um)
        k4q, (k4v, _) = v + h * k3v, acc(q + h * k3q, v + h * k3v, u1)
        q = q + h / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
        v = v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return np.array(out).T
