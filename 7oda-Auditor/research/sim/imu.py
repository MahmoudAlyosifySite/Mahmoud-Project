"""Hand motion (tremor, sway, fidgets), IMU sampling, device-motion fusion output.

Sign conventions follow Core Motion: accelerometer values are in g and equal
gravity + userAcceleration, where gravity is the unit vector towards the
ground in device axes (a phone flat on a table, screen up, reads z = -1) and
userAcceleration = -(kinematic acceleration)/g0. Gyro values are rad/s.
The attitude quaternion maps device axes to a z-up reference frame with an
arbitrary heading, so gravity = R(q)^T (0, 0, -1) at every sample.
"""
from __future__ import annotations

import numpy as np

MOTION_FS = 200.0      # Hz, grid for the slow kinematic signals
MOTION_NFFT = 1024


# --------------------------------------------------------------------------- rotations

def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


R_UPRIGHT = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])   # portrait, screen facing the user


def pose_matrix(tilt_deg, inplane_deg, side_deg, yaw_deg=0.0):
    """Device -> reference rotation. tilt: screen tilted back from vertical
    (0 upright, 90 flat face up); inplane: rotation about the screen normal;
    side: rotation about the device's long axis."""
    t, i, s, y = np.deg2rad([tilt_deg, inplane_deg, side_deg, yaw_deg])
    return rot_z(y) @ rot_x(-t) @ R_UPRIGHT @ rot_y(s) @ rot_z(i)


def rodrigues(theta: np.ndarray) -> np.ndarray:
    """(..., 3) rotation vectors -> (..., 3, 3) matrices."""
    ang = np.linalg.norm(theta, axis=-1)[..., None, None]
    k = np.where(ang > 1e-12, theta[..., None, :] / np.maximum(ang, 1e-300), 0.0)[..., 0, :]
    K = np.zeros(theta.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    eye = np.broadcast_to(np.eye(3), K.shape)
    return eye + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """(3, 3) -> (w, x, y, z) with w >= 0."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, (..., 4) as (w, x, y, z)."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], -1)


def quat_from_rotvec(theta: np.ndarray) -> np.ndarray:
    ang = np.linalg.norm(theta, axis=-1, keepdims=True)
    k = np.where(ang > 1e-12, theta / np.maximum(ang, 1e-300), 0.0)
    return np.concatenate([np.cos(ang / 2), np.sin(ang / 2) * k], -1)


def quat_to_gravity(q: np.ndarray) -> np.ndarray:
    """Gravity (towards the ground) in device axes for device->reference quaternions."""
    w, x, y, z = np.moveaxis(q, -1, 0)
    # third row of R(q), negated: R^T (0,0,-1) = -(R[2,0], R[2,1], R[2,2])
    return -np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)


# --------------------------------------------------------------------------- filters

def butter_lowpass_sos(fc: float, fs: float, order: int):
    """Digital Butterworth low-pass as biquad sections (b0, b1, b2, a1, a2), bilinear with prewarping."""
    if order % 2:
        raise ValueError("order must be even")
    wc = 2 * fs * np.tan(np.pi * fc / fs)
    k = 2 * fs
    sos = []
    for i in range(order // 2):
        theta = np.pi * (2 * i + 1) / (2 * order)
        q = 1 / (2 * np.sin(theta))                      # section quality factor
        # analog: wc^2 / (s^2 + wc/q s + wc^2), bilinear s = k (z-1)/(z+1)
        a0 = k * k + wc / q * k + wc * wc
        b0 = wc * wc / a0
        sos.append((b0, 2 * b0, b0, (2 * wc * wc - 2 * k * k) / a0, (k * k - wc / q * k + wc * wc) / a0))
    return sos


def sosfilt_time(sos, y: np.ndarray, k_start: int = 0) -> np.ndarray:
    """Causal filtering along axis 0 of (T, ...) with zero initial state (input zero before k_start)."""
    out = y.copy()
    for (b0, b1, b2, a1, a2) in sos:
        x1 = np.zeros(y.shape[1:], y.dtype)
        x2 = np.zeros_like(x1)
        y1 = np.zeros_like(x1)
        y2 = np.zeros_like(x1)
        src = out.copy()
        for k in range(k_start, y.shape[0]):
            x0 = src[k]
            y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            out[k] = y0
            x2, x1, y2, y1 = x1, x0, y1, y0
    return out


# --------------------------------------------------------------------------- slow kinematics

def hand_motion_batch(tps, rngs, duration: float, fidgets):
    """Kinematic hand-frame motion for a batch on the MOTION_FS grid starting at t = -0.40 s
    (relative to hapticStart). Tremor: band-limited noise peaked at the person's tremor
    frequency (8-12 Hz); sway: slow (< 3 Hz) rotation whose derivative is added to the gyro;
    fidget: an occasional 8-25 degree hand movement. Returns dict(t, acc (B,3,M) in g of
    kinematic acceleration, gyro (B,3,M) rad/s, theta (B,3,M) rad rotation vector, fidget list)."""
    B = len(tps)
    M = int(np.ceil((duration + 0.85) * MOTION_FS)) + 2
    t = -0.40 + np.arange(M) / MOTION_FS
    f = np.fft.rfftfreq(MOTION_NFFT, 1 / MOTION_FS)
    nf = f.size
    z = np.stack([r.standard_normal((2, 9, nf)) for r in rngs])          # (B, 2, 9, nf)
    spec = z[:, 0] + 1j * z[:, 1]
    ftr = np.array([tp.tremor_f for tp in tps])[:, None]
    bw = np.array([tp.tremor_bw for tp in tps])[:, None]
    fc = np.array([tp.sway_fc for tp in tps])[:, None]
    S_tr = np.where((f >= 2.0) & (f <= 30.0), 1.0 / (1.0 + ((f - ftr) / (bw / 2)) ** 2) + 0.03, 0.0)
    S_sw = np.where((f >= 0.05) & (f <= 3.0), 1.0 / (1.0 + (f / fc) ** 2) ** 3, 0.0)
    S_tl = np.where((f >= 0.1) & (f <= 4.0), 1.0, 0.0) * np.ones((B, 1))
    S = np.concatenate([np.repeat(S_tr[:, None], 6, 1), np.repeat(S_sw[:, None], 3, 1)], 1)   # (B, 9, nf)
    spec = spec * np.sqrt(S)
    x = np.fft.irfft(spec, MOTION_NFFT, axis=-1)
    scale = np.sqrt(np.mean(x * x, axis=-1, keepdims=True)) + 1e-300
    x = x / scale
    dsw = np.fft.irfft(spec[:, 6:9] * (2j * np.pi * f), MOTION_NFFT, axis=-1) / scale[:, 6:9]
    tr_acc = np.stack([tp.tremor_acc for tp in tps])[:, :, None]
    tr_gyr = np.stack([tp.tremor_gyro for tp in tps])[:, :, None]
    sway = np.stack([tp.sway_rad for tp in tps])[:, :, None]
    table = np.array([tp.table for tp in tps])[:, None, None]
    acc = x[:, 0:3, :M] * tr_acc
    gyro = x[:, 3:6, :M] * tr_gyr + dsw[:, :, :M] * sway
    theta = x[:, 6:9, :M] * sway
    # small low-frequency translation of the hand (removed by the 5 Hz high-pass, kept for realism)
    zt = np.stack([r.standard_normal((2, 3, nf)) for r in rngs])
    xt = np.fft.irfft((zt[:, 0] + 1j * zt[:, 1]) * np.sqrt(S_tl)[:, None], MOTION_NFFT, axis=-1)
    xt = xt / (np.sqrt(np.mean(xt * xt, axis=-1, keepdims=True)) + 1e-300)
    acc = acc + xt[:, :, :M] * np.array([r.uniform(0.0002, 0.0008) for r in rngs])[:, None, None]
    acc = np.where(table, 0.0, acc)
    gyro = np.where(table, 0.0, gyro)
    theta = np.where(table, 0.0, theta)
    infos = []
    for b in range(B):
        if not fidgets[b] or tps[b].table:
            infos.append(None)
            continue
        r = rngs[b]
        axis = r.standard_normal(3)
        axis /= np.linalg.norm(axis)
        amp = np.deg2rad(r.uniform(8.0, 25.0))
        t0, T = r.uniform(-0.25, duration + 0.1), r.uniform(0.2, 0.5)
        s = np.clip((t - t0) / T, 0.0, 1.0)
        theta[b] += axis[:, None] * amp * (3 * s * s - 2 * s ** 3)
        gyro[b] += axis[:, None] * amp * (6 * s - 6 * s * s) / T
        acc[b] += axis[[1, 2, 0], None] * r.uniform(0.02, 0.08) * np.sin(np.pi * s)
        infos.append(dict(start=float(t0), length=float(T), angleDeg=float(np.rad2deg(amp))))
    return dict(t=t, acc=acc, gyro=gyro, theta=theta, fidget=infos)


def interp_rows(t_grid: np.ndarray, rows: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Linear interpolation of (C, M) rows sampled on a uniform grid at times t -> (C, len(t))."""
    dt = t_grid[1] - t_grid[0]
    p = np.clip((t - t_grid[0]) / dt, 0, len(t_grid) - 1.000001)
    i = np.floor(p).astype(int)
    w = p - i
    return rows[:, i] * (1 - w) + rows[:, i + 1] * w


# --------------------------------------------------------------------------- sampling

def sample_times(rng, t_lo: float, t_hi: float, fs: float, phase: float, jitter: float) -> np.ndarray:
    ts = 1.0 / fs
    n = int(np.floor((t_hi - t_lo) / ts)) + 2
    t = t_lo + phase + np.arange(n) * ts + rng.normal(0, jitter, n)
    return t[(t >= t_lo) & (t <= t_hi)]


def gap_mask(rng, t: np.ndarray, windows, drop_rate: float) -> np.ndarray:
    keep = rng.uniform(size=t.size) >= drop_rate
    for a, b in windows:
        keep &= ~((t >= a) & (t < b))
    return keep


def gap_windows(rng, t_lo, t_hi, fs, burst_rate, long_gap_rate):
    w = []
    ts = 1.0 / fs
    if rng.uniform() < burst_rate:
        a = rng.uniform(t_lo, t_hi)
        w.append((a, a + rng.integers(3, 9) * ts))
    if rng.uniform() < long_gap_rate:
        a = rng.uniform(t_lo, t_hi)
        w.append((a, a + rng.integers(30, 61) * ts))
    return w


def duplicate_indices(rng, n: int, rate: float) -> np.ndarray:
    """Index map that repeats some samples (duplicated timestamps)."""
    rep = 1 + (rng.uniform(size=n) < rate).astype(int)
    return np.repeat(np.arange(n), rep)
