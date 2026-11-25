"""Signal processing (SPEC section 5): grid, filtering, envelope, template,
alignment and event indices.

``process(recording)`` returns a :class:`Processed` object that mirrors the
Swift ``ProcessedSignal`` (SPEC 10.1). Arrays are numpy float64; 3-channel
quantities have shape ``(3, N)`` in the order x, y, z.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import dsp
from ._numerics import TieAudit, mean, median, note_argmax, seq_sum
from .challenge import curve_at
from .recording import sort_dedup


@dataclass(frozen=True)
class SignalConfig:
    """Defaults of SPEC 5.1."""

    fs: float = 100.0
    pre_roll: float = 0.25
    post_roll: float = 0.25
    hp_cutoff: float = 5.0
    hp_q: float = dsp.Q_BUTTERWORTH
    motion_low_pass: float = 3.0
    motion_q: float = dsp.Q_BUTTERWORTH
    tremor_f0: float = math.sqrt(8.0 * 12.0)
    tremor_q: float = math.sqrt(8.0 * 12.0) / 4.0
    pad: int = 30
    gap_tolerance: float = 0.025
    lag_min: int = -10
    lag_max: int = 25
    template_decay: float = 0.03
    accel_noise_floor: float = 1e-6
    gyro_noise_floor: float = 1e-6

    def highpass(self) -> dsp.Coeffs:
        return dsp.biquad_highpass(self.hp_cutoff, self.fs, self.hp_q)

    def motion_lowpass(self) -> dsp.Coeffs:
        return dsp.biquad_lowpass(self.motion_low_pass, self.fs, self.motion_q)

    def tremor_bandpass(self) -> dsp.Coeffs:
        return dsp.biquad_bandpass(self.tremor_f0, self.fs, self.tremor_q)


DEFAULT_CONFIG = SignalConfig()


@dataclass
class Alignment:
    """Result of SPEC 5.6."""

    lag: int
    ncc: float
    peak_margin: float
    ncc_by_lag: dict[int, float] = field(default_factory=dict)


@dataclass
class Processed:
    """Python mirror of ``ProcessedSignal`` (SPEC 10.1)."""

    config: SignalConfig
    challenge: dict
    n: int
    grid_start: float
    grid: np.ndarray
    accel: np.ndarray          # (3, N) raw accel on the grid (g)
    gyro: np.ndarray           # (3, N) raw gyro on the grid (rad/s)
    gravity: np.ndarray        # (3, N) device-motion gravity
    rotation: np.ndarray       # (3, N) device-motion rotation rate
    accel_missing: np.ndarray  # (N,) bool
    gyro_missing: np.ndarray   # (N,) bool
    accel_hp: np.ndarray       # (3, N) aH
    gyro_hp: np.ndarray        # (3, N) gH
    energy_accel: np.ndarray   # EA
    energy_gyro: np.ndarray    # EG
    noise_accel: float         # nA
    noise_gyro: float          # nG
    envelope: np.ndarray       # env (normalised energy)
    amplitude: np.ndarray      # x = sqrt(env)
    template: np.ndarray       # h at lag 0
    alignment: Alignment
    gap_fraction: float
    sample_rate: float
    cache: dict = field(default_factory=dict, repr=False)

    # --- convenience -----------------------------------------------------
    @property
    def lag(self) -> int:
        return self.alignment.lag

    @property
    def ncc(self) -> float:
        return self.alignment.ncc

    @property
    def peak_margin(self) -> float:
        return self.alignment.peak_margin

    # --- SPEC 5.7 --------------------------------------------------------
    def index(self, t: float) -> int:
        """``idx(t) = Int((preRoll + t) * fs + 0.5) + lag`` (unclamped)."""
        cfg = self.config
        return int((cfg.pre_roll + t) * cfg.fs + 0.5) + self.alignment.lag

    def clamp(self, i: int) -> int:
        """Clamp an index to ``[0, N]`` (used for range bounds)."""
        return min(max(i, 0), self.n)

    @property
    def end_index(self) -> int:
        """``endIdx = idx(duration)`` (unclamped)."""
        return self.index(self.challenge["duration"])

    @property
    def onset_first(self) -> int:
        """``idx(events[0].time)`` (unclamped)."""
        events = self.challenge.get("events") or []
        return self.index(events[0]["time"] if events else 0.0)

    @property
    def quiet_range(self) -> tuple[int, int]:
        """Quiet window Q of SPEC 5.7 as a half-open ``(start, end)``."""
        end = self.clamp(self.onset_first - 2)
        if end < 10:
            start = max(0, min(self.end_index + 5, self.n - 10))
            return (start, self.n)
        return (0, end)

    @property
    def event_range(self) -> tuple[int, int]:
        """``[onsetFirst, endIdx)`` clamped to ``[0, N]``."""
        a = self.clamp(self.onset_first)
        b = self.clamp(self.end_index)
        return (a, max(a, b))


def make_grid(haptic_start: float, duration: float, cfg: SignalConfig) -> tuple[float, int, np.ndarray]:
    """``t0 = hapticStart - preRoll``; ``N = Int((preRoll + duration + postRoll) * fs + 0.5)``;
    ``grid[k] = t0 + Double(k) / fs``."""
    t0 = haptic_start - cfg.pre_roll
    n = int((cfg.pre_roll + duration + cfg.post_roll) * cfg.fs + 0.5)
    n = max(n, 0)
    grid = t0 + np.arange(n, dtype=np.float64) / cfg.fs
    return t0, n, grid


def template(challenge: dict, n: int, cfg: SignalConfig) -> np.ndarray:
    """Template amplitude ``h[n]`` at lag 0 (SPEC 5.6).

    ``h[n] = (sum over taps) + (sum over ramps)``, each sum accumulated in
    event order from 0.0; ``tau = Double(n)/fs - preRoll``.
    """
    events = challenge.get("events") or []
    taps = [e for e in events if e["kind"] == "tap"]
    ramps = [e for e in events if e["kind"] == "ramp"]
    decay = cfg.template_decay
    out = np.zeros(n, dtype=np.float64)
    for k in range(n):
        tau = float(k) / cfg.fs - cfg.pre_roll
        tap_sum = 0.0
        for e in taps:
            ti = e["time"]
            if tau >= ti:
                tap_sum = tap_sum + e["intensity"] * math.exp(-(tau - ti) / decay)
        ramp_sum = 0.0
        for e in ramps:
            t = e["time"]
            d = e["duration"]
            if t <= tau < t + d:
                ramp_sum = ramp_sum + curve_at(e, tau - t)
            elif tau >= t + d:
                ramp_sum = ramp_sum + curve_at(e, d) * math.exp(-(tau - t - d) / decay)
        out[k] = tap_sum + ramp_sum
    return out


def pearson(x: np.ndarray, h: np.ndarray) -> float:
    """Two-pass Pearson correlation; 0 if either centred sum of squares is 0.

    ``r = sum(dx*dh) / sqrt(sum(dx*dx) * sum(dh*dh))`` with
    ``dx = x - mean(x)``, ``dh = h - mean(h)``; sums left to right.
    """
    mx = mean(x)
    mh = mean(h)
    dx = x - mx
    dh = h - mh
    sxx = seq_sum(dx * dx)
    shh = seq_sum(dh * dh)
    if sxx == 0.0 or shh == 0.0:
        return 0.0
    return seq_sum(dx * dh) / math.sqrt(sxx * shh)


def align(amplitude: np.ndarray, tmpl: np.ndarray, cfg: SignalConfig, audit: TieAudit | None = None) -> Alignment:
    """Lag search of SPEC 5.6 (pairs ``x[n]`` with ``h[n - L]``)."""
    n = amplitude.size
    lags = list(range(cfg.lag_min, cfg.lag_max + 1))
    ncc_by_lag: dict[int, float] = {}
    for lag in lags:
        lo = max(0, lag)
        hi = min(n, n + lag)
        if hi - lo < 10:
            ncc_by_lag[lag] = 0.0
            continue
        ncc_by_lag[lag] = pearson(amplitude[lo:hi], tmpl[lo - lag:hi - lag])
    best_lag, best_val, peak_margin = select_lag(ncc_by_lag)
    if audit is not None:
        vals = [ncc_by_lag[v] for v in lags]
        note_argmax(audit, "alignment.lag", vals, lags.index(best_lag), scale=1e-3)
    return Alignment(lag=best_lag, ncc=best_val, peak_margin=peak_margin, ncc_by_lag=ncc_by_lag)


def select_lag(ncc_by_lag: dict[int, float]) -> tuple[int, float, float]:
    """``lag`` = argmax (ties -> smaller ``|L|``, then smaller ``L``), ``ncc`` = max,
    ``peakMargin = ncc - max{ncc(L') : |L' - lag| >= 4}`` (``ncc`` if no such ``L'``)."""
    best_lag = None
    best_val = -math.inf
    for lag in sorted(ncc_by_lag, key=lambda v: (abs(v), v)):
        val = ncc_by_lag[lag]
        if best_lag is None or val > best_val:
            best_lag = lag
            best_val = val
    if best_lag is None:
        raise ValueError("empty lag range")
    far = [ncc_by_lag[v] for v in ncc_by_lag if abs(v - best_lag) >= 4]
    peak_margin = best_val - max(far) if far else best_val
    return best_lag, best_val, peak_margin


def process(rec: dict, config: SignalConfig | None = None, audit: TieAudit | None = None) -> Processed:
    """Run SPEC 5.2-5.7 on a recording dict (SPEC 4 JSON form)."""
    cfg = config or DEFAULT_CONFIG
    challenge = rec["challenge"]
    duration = float(challenge["duration"])
    t0, n, grid = make_grid(float(rec["hapticStart"]), duration, cfg)

    acc = sort_dedup(rec["accel"])
    gyr = sort_dedup(rec["gyro"])
    mot = sort_dedup(rec["motion"])

    def _resample3(stream: dict, keys: tuple[str, str, str]) -> tuple[np.ndarray, np.ndarray]:
        chans = []
        miss = np.ones(n, dtype=bool)
        for i, k in enumerate(keys):
            vals, m = dsp.resample(stream["t"], stream[k], grid, cfg.gap_tolerance)
            chans.append(vals)
            if i == 0:
                miss = m
        return np.vstack(chans) if n else np.zeros((3, 0)), miss

    accel, accel_missing = _resample3(acc, ("x", "y", "z"))
    gyro, gyro_missing = _resample3(gyr, ("x", "y", "z"))
    gravity, _ = _resample3(mot, ("gravityX", "gravityY", "gravityZ"))
    rotation, _ = _resample3(mot, ("rotationX", "rotationY", "rotationZ"))

    gap_fraction = float(np.count_nonzero(accel_missing | gyro_missing)) / float(n) if n else 0.0
    ta = acc["t"]
    sample_rate = 1.0 / median(ta[1:] - ta[:-1]) if ta.size >= 2 else 0.0

    hp = cfg.highpass()
    accel_hp = np.vstack([dsp.filtfilt_z(hp, accel[c], cfg.pad, 0.0) for c in range(3)]) if n else np.zeros((3, 0))
    gyro_hp = np.vstack([dsp.filtfilt_z(hp, gyro[c], cfg.pad, 0.0) for c in range(3)]) if n else np.zeros((3, 0))
    ea = accel_hp[0] * accel_hp[0] + accel_hp[1] * accel_hp[1] + accel_hp[2] * accel_hp[2]
    eg = gyro_hp[0] * gyro_hp[0] + gyro_hp[1] * gyro_hp[1] + gyro_hp[2] * gyro_hp[2]

    pre_end = int(cfg.pre_roll * cfg.fs + 0.5) - 5
    if pre_end < 1:
        noise_a = cfg.accel_noise_floor
        noise_g = cfg.gyro_noise_floor
    else:
        noise_a = max(mean(ea[0:pre_end]), cfg.accel_noise_floor)
        noise_g = max(mean(eg[0:pre_end]), cfg.gyro_noise_floor)
    env = ea / noise_a + eg / noise_g
    amp = np.sqrt(env)

    tmpl = template(challenge, n, cfg)
    alignment = align(amp, tmpl, cfg, audit)

    return Processed(
        config=cfg,
        challenge=challenge,
        n=n,
        grid_start=t0,
        grid=grid,
        accel=accel,
        gyro=gyro,
        gravity=gravity,
        rotation=rotation,
        accel_missing=accel_missing,
        gyro_missing=gyro_missing,
        accel_hp=accel_hp,
        gyro_hp=gyro_hp,
        energy_accel=ea,
        energy_gyro=eg,
        noise_accel=float(noise_a),
        noise_gyro=float(noise_g),
        envelope=env,
        amplitude=amp,
        template=tmpl,
        alignment=alignment,
        gap_fraction=gap_fraction,
        sample_rate=float(sample_rate),
    )
