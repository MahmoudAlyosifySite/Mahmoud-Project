"""Small deterministic synthetic recordings for tests and fixtures.

This is *not* the physics simulator (``research/sim``). It only produces
recordings in the exact SPEC 4 JSON form with enough structure for the
pipeline to have something to measure:

- every challenge event excites a damped sinusoid (tap) or a sustained
  sinusoid following the ramp curve, at an audio-range frequency that the
  ~100 Hz sampling aliases;
- per-axis accel/gyro gains, decay time, intensity exponent, sharpness
  response, tremor, grip angle and noise are "hand" parameters, so different
  synthetic people and hands produce different features;
- gravity, tremor, white noise and a gyro bias are added; timestamps are on a
  boot clock with jitter; the vibration starts ``latency`` seconds after
  ``hapticStart``.

Options cover sensor gaps, duplicate and unsorted timestamps, large motion,
the phone resting on a table, weak signal, slow sampling and a missing gyro
stream. Every random draw comes from :class:`auditor_ref.rng.Xoshiro256`, so
output is identical for identical arguments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from .challenge import curve_at, generate
from .recording import make_recording
from .rng import Xoshiro256


@dataclass(frozen=True)
class HandParams:
    """How a synthetic hand (or table) responds to the vibration."""

    freq: float = 172.0                     # Hz, response frequency at sharpness 0
    freq_sharpness: float = 45.0            # Hz added per unit sharpness
    decay: float = 0.022                    # s, amplitude time constant
    accel_gain: tuple[float, float, float] = (0.040, 0.022, 0.012)   # g at intensity 1
    gyro_gain: tuple[float, float, float] = (0.070, 0.044, 0.110)    # rad/s at intensity 1
    exponent: float = 1.25                  # amplitude ~ intensity ** exponent
    sharp_gain: float = 0.45                # amplitude * (1 + sharp_gain * (S - 0.5))
    ramp_gain: float = 0.55                 # sustained amplitude relative to a tap
    ramp_exponent: float = 1.1
    tremor_accel: float = 0.0012            # g
    tremor_gyro: float = 0.0035             # rad/s
    tremor_freq: float = 9.6                # Hz
    gravity: tuple[float, float, float] = (0.08, -0.60, -0.795)
    accel_noise: float = 0.0008             # g (white, per sample)
    gyro_noise: float = 0.0014              # rad/s
    gyro_bias: tuple[float, float, float] = (0.0021, -0.0013, 0.0008)


#: The phone flat on a hard table: crisp response, no tremor, tiny noise.
TABLE = HandParams(
    freq=205.0, freq_sharpness=30.0, decay=0.012,
    accel_gain=(0.010, 0.006, 0.016), gyro_gain=(0.010, 0.012, 0.008),
    exponent=1.0, sharp_gain=0.2, ramp_gain=0.5, ramp_exponent=1.0,
    tremor_accel=0.0, tremor_gyro=0.0, gravity=(0.004, -0.003, -0.99999),
    accel_noise=0.00025, gyro_noise=0.00035, gyro_bias=(0.0011, 0.0004, -0.0007),
)


def make_hand(seed: int, base: HandParams | None = None, spread: float = 1.0) -> HandParams:
    """A random synthetic hand around ``base`` (deterministic in ``seed``)."""
    b = base or HandParams()
    r = Xoshiro256(seed)

    def jit(v: float, rel: float) -> float:
        return v * math.exp(spread * rel * r.normal())

    g = [b.gravity[0] + spread * 0.15 * r.normal(), b.gravity[1] + spread * 0.15 * r.normal(), b.gravity[2] + spread * 0.08 * r.normal()]
    gn = math.sqrt(sum(v * v for v in g))
    return replace(
        b,
        freq=b.freq + spread * 18.0 * r.normal(),
        freq_sharpness=jit(b.freq_sharpness, 0.2),
        decay=jit(b.decay, 0.25),
        accel_gain=tuple(jit(v, 0.35) for v in b.accel_gain),
        gyro_gain=tuple(jit(v, 0.35) for v in b.gyro_gain),
        exponent=b.exponent + spread * 0.2 * r.normal(),
        sharp_gain=b.sharp_gain + spread * 0.15 * r.normal(),
        ramp_gain=jit(b.ramp_gain, 0.25),
        ramp_exponent=b.ramp_exponent + spread * 0.15 * r.normal(),
        tremor_accel=jit(b.tremor_accel, 0.3),
        tremor_gyro=jit(b.tremor_gyro, 0.3),
        tremor_freq=b.tremor_freq + spread * 0.8 * r.normal(),
        gravity=(g[0] / gn, g[1] / gn, g[2] / gn),
    )


@dataclass(frozen=True)
class SynthOptions:
    """Recording conditions."""

    haptic_start: float = 81234.5612        # s on the boot clock
    latency: float = 0.04                   # s, vibration onset after hapticStart
    rate: float = 100.0                     # Hz, nominal sensor rate
    jitter: float = 0.0003                  # s, timestamp jitter (std)
    signal_scale: float = 1.0               # response amplitude multiplier
    gap: tuple[float, float] | None = None  # (start, length) in s from the window start
    gap_streams: tuple[str, ...] = ("accel", "gyro")
    duplicates: float = 0.0                 # fraction of samples duplicated (same t)
    unsorted: bool = False                  # shuffle the sample order of every stream
    motion: float = 0.0                     # rad/s amplitude of slow hand motion
    tilt_deg: float = 0.0                   # gravity rotation over the window
    drop_gyro: bool = False                 # gyro stream without samples
    pre_extra: float = 0.1                  # s recorded before hapticStart - preRoll
    post_extra: float = 0.1                 # s recorded after the post-roll
    pre_roll: float = 0.25
    post_roll: float = 0.25


_ACC_PHASE = (0.0, 0.7, 1.9)
_GYR_PHASE = (0.3, 1.2, 2.5)


def _response(t: float, events: list[dict], onset0: float, hand: HandParams, scale: float, phase: float) -> float:
    """Vibration response (unit gain) at absolute time ``t``."""
    total = 0.0
    T = hand.decay
    for ev in events:
        start = onset0 + ev["time"]
        tau = t - start
        if tau < 0.0:
            continue
        if ev["kind"] == "tap":
            amp = (ev["intensity"] ** hand.exponent) * (1.0 + hand.sharp_gain * (ev["sharpness"] - 0.5))
            f = hand.freq + hand.freq_sharpness * ev["sharpness"]
            total += scale * amp * math.exp(-tau / T) * math.sin(2.0 * math.pi * f * tau + phase)
        else:
            d = ev["duration"]
            f = hand.freq + hand.freq_sharpness * ev["sharpness"]
            if tau <= d:
                level = curve_at(ev, tau) ** hand.ramp_exponent
                env = 1.0 - math.exp(-tau / T)
            else:
                level = curve_at(ev, d) ** hand.ramp_exponent
                env = (1.0 - math.exp(-d / T)) * math.exp(-(tau - d) / T)
            total += scale * hand.ramp_gain * level * env * math.sin(2.0 * math.pi * f * tau + phase)
    return total


def _rotate_x(v: tuple[float, float, float], angle: float) -> tuple[float, float, float]:
    c = math.cos(angle)
    s = math.sin(angle)
    return (v[0], c * v[1] - s * v[2], s * v[1] + c * v[2])


def _times(rng: Xoshiro256, start: float, end: float, opts: SynthOptions) -> list[float]:
    period = 1.0 / opts.rate
    t = start + rng.uniform() * period
    out = []
    k = 0
    while True:
        tk = t + k * period + opts.jitter * max(-3.0, min(3.0, rng.normal()))
        if tk > end:
            break
        out.append(tk)
        k += 1
    return out


def _uuid(rng: Xoshiro256) -> str:
    a = rng.next()
    b = rng.next()
    a = (a & ~(0xF << 12)) | (0x4 << 12)              # version 4
    b = (b & ~(0x3 << 62)) | (0x2 << 62)              # RFC 4122 variant
    h = "%016x%016x" % (a, b)
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}".upper()


def _apply_stream_options(rng: Xoshiro256, stream: dict, opts: SynthOptions, name: str, window_start: float) -> dict:
    keys = list(stream.keys())
    n = len(stream["t"])
    idx = list(range(n))
    if opts.gap is not None and name in opts.gap_streams:
        g0 = window_start + opts.gap[0]
        g1 = g0 + opts.gap[1]
        idx = [i for i in idx if not (g0 <= stream["t"][i] < g1)]
    if opts.duplicates > 0.0:
        out = []
        for i in idx:
            out.append(i)
            if rng.uniform() < opts.duplicates:
                out.append(-1 - i)   # marker: duplicate of sample i with perturbed values
        idx = out
    if opts.unsorted:
        for i in range(len(idx) - 1, 0, -1):
            j = rng.int_in_range(0, i)
            idx[i], idx[j] = idx[j], idx[i]
    res = {k: [] for k in keys}
    for i in idx:
        if i >= 0:
            for k in keys:
                res[k].append(stream[k][i])
        else:
            src = -1 - i
            for k in keys:
                v = stream[k][src]
                res[k].append(v if k == "t" else v + 1e-4 * rng.normal())
    return res


def synth_recording(
    seed: int,
    hand: HandParams | None = None,
    opts: SynthOptions | None = None,
    *,
    challenge: dict | None = None,
    challenge_seed: int | None = None,
    purpose: str = "check",
    label: dict | None = None,
    device: dict | None = None,
    created_at: str = "2026-11-24T10:00:00Z",
) -> dict:
    """One synthetic recording (SPEC 4 JSON form), deterministic in its arguments."""
    hand = hand or HandParams()
    opts = opts or SynthOptions()
    rng = Xoshiro256(seed)
    rec_id = _uuid(rng)
    if challenge is None:
        challenge = generate(challenge_seed if challenge_seed is not None else rng.next())
    events = challenge["events"]
    hs = opts.haptic_start
    w0 = hs - opts.pre_roll - opts.pre_extra
    w1 = hs + challenge["duration"] + opts.post_roll + opts.post_extra
    span = w1 - w0
    onset0 = hs + opts.latency
    tremor_phase = [rng.uniform() * 2.0 * math.pi for _ in range(6)]
    motion_phase = [rng.uniform() * 2.0 * math.pi for _ in range(3)]

    def gravity_at(t: float) -> tuple[float, float, float]:
        frac = (t - w0) / span
        return _rotate_x(hand.gravity, math.radians(opts.tilt_deg) * frac)

    def rotation_at(t: float) -> list[float]:
        if opts.motion == 0.0:
            return [0.0, 0.0, 0.0]
        u = t - w0
        return [
            opts.motion * math.sin(2.0 * math.pi * 0.7 * u + motion_phase[0]),
            0.6 * opts.motion * math.sin(2.0 * math.pi * 1.1 * u + motion_phase[1]),
            0.4 * opts.motion * math.sin(2.0 * math.pi * 0.5 * u + motion_phase[2]),
        ]

    def tremor(t: float, i: int, amp: float) -> float:
        return amp * math.sin(2.0 * math.pi * hand.tremor_freq * (t - w0) + tremor_phase[i])

    accel = {"t": [], "x": [], "y": [], "z": []}
    for t in _times(rng, w0, w1, opts):
        g = gravity_at(t)
        accel["t"].append(t)
        for c, key in enumerate(("x", "y", "z")):
            r = _response(t, events, onset0, hand, opts.signal_scale, _ACC_PHASE[c])
            accel[key].append(g[c] + hand.accel_gain[c] * r + tremor(t, c, hand.tremor_accel)
                              + hand.accel_noise * rng.normal())
    gyro = {"t": [], "x": [], "y": [], "z": []}
    if not opts.drop_gyro:
        for t in _times(rng, w0, w1, opts):
            rot = rotation_at(t)
            gyro["t"].append(t)
            for c, key in enumerate(("x", "y", "z")):
                r = _response(t, events, onset0, hand, opts.signal_scale, _GYR_PHASE[c])
                gyro[key].append(rot[c] + hand.gyro_bias[c] + hand.gyro_gain[c] * r
                                 + tremor(t, 3 + c, hand.tremor_gyro) + hand.gyro_noise * rng.normal())
    motion = {k: [] for k in ("t", "gravityX", "gravityY", "gravityZ", "userAccelX", "userAccelY", "userAccelZ",
                              "rotationX", "rotationY", "rotationZ", "attitudeW", "attitudeX", "attitudeY", "attitudeZ")}
    for t in _times(rng, w0, w1, opts):
        g = gravity_at(t)
        rot = rotation_at(t)
        motion["t"].append(t)
        gn = [g[c] + 2e-5 * rng.normal() for c in range(3)]
        nn = math.sqrt(sum(v * v for v in gn))
        for c, key in enumerate(("gravityX", "gravityY", "gravityZ")):
            motion[key].append(gn[c] / nn)
        for c, key in enumerate(("userAccelX", "userAccelY", "userAccelZ")):
            r = _response(t, events, onset0, hand, opts.signal_scale, _ACC_PHASE[c])
            motion[key].append(hand.accel_gain[c] * r + tremor(t, c, hand.tremor_accel))
        for c, key in enumerate(("rotationX", "rotationY", "rotationZ")):
            r = _response(t, events, onset0, hand, opts.signal_scale, _GYR_PHASE[c])
            motion[key].append(rot[c] + hand.gyro_gain[c] * r + tremor(t, 3 + c, hand.tremor_gyro)
                               + 0.5 * hand.gyro_noise * rng.normal())
        # an attitude consistent enough for display purposes (not used by the pipeline)
        ang = math.acos(max(-1.0, min(1.0, -g[2])))
        motion["attitudeW"].append(math.cos(ang / 2.0))
        motion["attitudeX"].append(math.sin(ang / 2.0))
        motion["attitudeY"].append(0.0)
        motion["attitudeZ"].append(0.0)

    accel = _apply_stream_options(rng, accel, opts, "accel", w0)
    gyro = _apply_stream_options(rng, gyro, opts, "gyro", w0)
    motion = _apply_stream_options(rng, motion, opts, "motion", w0)
    return make_recording(
        challenge, hs, accel, gyro, motion,
        purpose=purpose, label=label, device=device, rec_id=rec_id, created_at=created_at,
    )


def label(person_id: str, person_name: str, hand: str = "right", accessory: str = "none",
          posture: str = "sitting", session_id: str = "S1") -> dict:
    """A SPEC 4 label dict."""
    return {"personID": person_id, "personName": person_name, "hand": hand,
            "accessory": accessory, "posture": posture, "sessionID": session_id}
