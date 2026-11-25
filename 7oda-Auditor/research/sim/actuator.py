"""Haptic actuator: Core Haptics events -> force on the phone body.

The actuator's moving mass follows a prescribed relative displacement xi(t)
(the closed-loop drive), and the phone feels the reaction u(t) = -xi''(t)
(the moving mass is folded into the amplitude). Writing the force as a second
derivative of a compactly supported displacement keeps momentum exact: every
event has zero net impulse and zero net displacement, so all low-frequency
content the 100 Hz sensors can see comes from the hand's own response, as on
the real phone.

Transient (tap): xi = X sin^2(pi s) (sin(2 pi f (t - t_on)) + a), s = (t - t_on)/T_b,
a 2-4 cycle burst at f = f0 (0.7 + 0.6 sharpness) with force amplitude
F = f_max * intensity * G(f); `a` (phone.tap_asym) is the moving mass's net
excursion during the click, the only source of sub-carrier force content
besides the burst envelope (its low-frequency force grows as Omega^2).
Continuous (ramp): xi = X e(t) sin(2 pi f (t - t_on)) where e(t) is the
intensity curve (SPEC 3.3) passed through a first-order lag with the actuator
time constant tau_lra (rise and fall).
G(f) is the actuator's resonance gain normalised to 1 at f0.
"""
from __future__ import annotations

import numpy as np

from .params import Phone


def lra_gain(f, phone: Phone):
    r = np.asarray(f, float) / phone.f0
    q = phone.q_lra
    return 1.0 / (q * np.sqrt((1 - r * r) ** 2 + (r / q) ** 2))


def event_frequency(sharpness, f0):
    return f0 * (0.7 + 0.6 * np.asarray(sharpness, float))


def tap_cycles(sharpness, phone: Phone):
    return np.clip(phone.tap_cycles + 0.8 * (0.5 - np.asarray(sharpness, float)), 2.0, 4.0)


def _split_events(challenge: dict):
    taps = [e for e in challenge["events"] if e["kind"] == "tap"]
    ramps = [e for e in challenge["events"] if e["kind"] == "ramp"]
    return taps, ramps


def _amplitude_to_x(f, delta, zoh_step):
    """Displacement amplitude whose discrete second difference (spacing delta)
    has unit force amplitude at frequency f; optionally compensates the
    zero-order-hold droop of an integrator step `zoh_step`."""
    g = (2.0 * np.sin(np.pi * f * delta) / delta) ** 2
    if zoh_step:
        g = g * np.sinc(f * zoh_step)
    return 1.0 / g


def _lag_envelope(tau, d, p0, p1, p2, tl):
    """First-order lag (time constant tl) of the piecewise-linear ramp command,
    zero before the ramp; closed form, vectorised."""
    half = d / 2
    sa = (p1 - p0) / half
    sb = (p2 - p1) / half
    ea = (p0 - sa * tl) * (1 - np.exp(-half / tl)) + sa * half          # e(d/2), from e(0) = 0
    eb = p1 + sb * half - sb * tl + (ea - p1 + sb * tl) * np.exp(-half / tl)  # e(d)
    t1 = tau
    e_a = p0 + sa * t1 - sa * tl + (-p0 + sa * tl) * np.exp(-np.clip(t1, 0, None) / tl)
    t2 = tau - half
    e_b = p1 + sb * t2 - sb * tl + (ea - p1 + sb * tl) * np.exp(-np.clip(t2, 0, None) / tl)
    t3 = tau - d
    e_c = eb * np.exp(-np.clip(t3, 0, None) / tl)
    return np.where(tau < 0, 0.0, np.where(tau < half, e_a, np.where(tau < d, e_b, e_c)))


def force_waveform(challenges, onsets, phone: Phone, r0: float, delta: float, n_points: int,
                   f0=None, amp=None, zoh_step=None) -> np.ndarray:
    """Actuator force u (N, along the actuator axis) on a uniform grid.

    challenges: list of B ChallengeSpec dicts; onsets: (B,) haptic onset time of
    pattern time 0 on the grid's clock; the grid is r0 + k*delta, k < n_points.
    f0: (B,) per-trial resonance (default phone.f0); amp: (B,) amplitude factors.
    Returns (B, n_points).
    """
    B = len(challenges)
    f0 = np.full(B, phone.f0) if f0 is None else np.asarray(f0, float)
    amp = np.ones(B) if amp is None else np.asarray(amp, float)
    onsets = np.asarray(onsets, float)
    xi = np.zeros(B * (n_points + 2))
    stride = n_points + 2                      # one zero guard sample on each side
    idx_parts, val_parts = [], []

    # ---- taps
    tap_rows = []
    for b, ch in enumerate(challenges):
        taps, _ = _split_events(ch)
        for e in taps:
            tap_rows.append((b, e["time"], e["intensity"], e["sharpness"]))
    if tap_rows:
        tr = np.array(tap_rows)
        bi = tr[:, 0].astype(int)
        f = event_frequency(tr[:, 3], f0[bi])
        tb = tap_cycles(tr[:, 3], phone) / f
        famp = phone.f_max * tr[:, 2] * lra_gain(f, phone) * amp[bi]
        x = famp * _amplitude_to_x(f, delta, zoh_step)
        t_on = onsets[bi] + tr[:, 1]
        k0 = np.floor((t_on - r0) / delta).astype(int)
        L = int(np.ceil(tb.max() / delta)) + 3
        k = k0[:, None] + np.arange(L)[None, :]
        tau = r0 + k * delta - t_on[:, None]
        s = tau / tb[:, None]
        v = np.where((s > 0) & (s < 1), x[:, None] * np.sin(np.pi * s) ** 2
                     * (np.sin(2 * np.pi * f[:, None] * tau) + phone.tap_asym), 0.0)
        idx_parts.append(bi[:, None] * stride + 1 + np.clip(k, -1, n_points))
        val_parts.append(np.where((k >= 0) & (k < n_points), v, 0.0))

    # ---- ramps
    ramp_rows = []
    for b, ch in enumerate(challenges):
        _, ramps = _split_events(ch)
        for e in ramps:
            c = e["curve"]
            ramp_rows.append((b, e["time"], e["duration"], e["sharpness"], c[0]["intensity"], c[1]["intensity"],
                              c[2]["intensity"]))
    if ramp_rows:
        rr = np.array(ramp_rows)
        bi = rr[:, 0].astype(int)
        d = rr[:, 2]
        f = event_frequency(rr[:, 3], f0[bi])
        famp = phone.f_max * lra_gain(f, phone) * amp[bi]
        x = famp * _amplitude_to_x(f, delta, zoh_step)
        tl = phone.tau_lra
        t_on = onsets[bi] + rr[:, 1]
        k0 = np.floor((t_on - r0) / delta).astype(int)
        L = int(np.ceil((d.max() + 10 * tl) / delta)) + 3
        k = k0[:, None] + np.arange(L)[None, :]
        tau = r0 + k * delta - t_on[:, None]
        env = _lag_envelope(tau, d[:, None], rr[:, 4:5], rr[:, 5:6], rr[:, 6:7], tl)
        env = np.where(tau <= d[:, None] + 10 * tl, env, 0.0)        # per-ramp support: batch independent
        v = x[:, None] * env * np.sin(2 * np.pi * f[:, None] * np.clip(tau, 0, None))
        idx_parts.append(bi[:, None] * stride + 1 + np.clip(k, -1, n_points))
        val_parts.append(np.where((k >= 0) & (k < n_points), v, 0.0))

    if idx_parts:
        idx = np.concatenate([p.ravel() for p in idx_parts])
        val = np.concatenate([p.ravel() for p in val_parts])
        xi += np.bincount(idx, weights=val, minlength=xi.size)[: xi.size]
    xi = xi.reshape(B, stride)
    return -(xi[:, 2:] - 2.0 * xi[:, 1:-1] + xi[:, :-2]) / (delta * delta)
