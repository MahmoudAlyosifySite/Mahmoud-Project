"""Phone, hand and person parameters for the physics simulator.

Everything here is plain numpy and deterministic: a phone or a person is a pure
function of its seed, and every session, grip and accessory realization is a
pure function of (person seed, hand, session, grip). Trial-level randomness is
the only thing that comes from the caller's generator.

Units: metres, kilograms, seconds, newtons, radians (angles in the dataclasses
are stored in degrees where noted). Device axes follow iOS: x to the right of
the screen, y towards the top, z out of the screen.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np

G0 = 9.80665                 # m/s^2 per g
DELTA_REF = 5e-6             # m, contact displacement at which the cubic term adds `eps` of the linear force
HANDS = ("right", "left")
HAND_ID = {"right": 0, "left": 1}
POSTURES = ("sitting", "standing", "freshGrip", "unspecified")
ACCESSORIES = ("none", "watch", "ring", "bracelet", "other")
CONDITIONS = ("handheld", "twoHanded", "onTable")

# Contact template for a one-handed right-hand grip, in fractions of the phone
# width W, height H and thickness T, measured from the phone's centre.
#               name      position (W, H, T)      inward normal (unnormalized)  surface axis
_CONTACTS = (
    ("palm",    (+0.25, -0.22, -0.50), (0.0, 0.0, 1.0), 2),
    ("thenar",  (+0.50, -0.30, -0.20), (-1.0, 0.0, 0.25), 0),
    ("index",   (-0.50, +0.12, -0.10), (1.0, 0.0, 0.30), 0),
    ("middle",  (-0.50, -0.04, -0.25), (1.0, 0.0, 0.35), 0),
    ("ring",    (-0.50, -0.19, -0.35), (1.0, 0.0, 0.40), 0),
    ("little",  (-0.18, -0.50, -0.30), (0.1, 1.0, 0.25), 1),
    ("thumb",   (+0.22, +0.02, +0.50), (0.0, 0.0, -1.0), 2),
)
CONTACT_NAMES = tuple(c[0] for c in _CONTACTS)
N_CONTACTS = len(_CONTACTS)
# Per-contact ranges (log-uniform for stiffness and damping, uniform for mass).
_KN = np.array([(2000, 8000), (1000, 4000), (300, 1500), (300, 1500), (300, 1500), (300, 1500), (100, 800)], float)
_CN = np.array([(4, 16), (2, 8), (0.8, 3), (0.8, 3), (0.8, 3), (0.8, 3), (0.3, 1.5)], float)
_MASS = np.array([(0.020, 0.070), (0.005, 0.015), (0.001, 0.004), (0.001, 0.004), (0.001, 0.004),
                  (0.001, 0.003), (0.001, 0.004)], float)
_FINGERS = (2, 3, 4)
# Within-person variability (docs/SIMULATION.md section 3): log-normal sd of multiplicative factors (common
# stiffness, per-contact stiffness/damping, coupled masses, cubic term), metres for contact positions, degrees
# for grip angles. Session ("day") and grip values except `eps` are scaled by the person's consistency; the
# session level is also scaled by `day_scale`.
SESSION_JITTER = dict(common=0.12, contact=0.08, mass=0.06, pos=0.0025, ang=4.0, eps=0.2)
GRIP_JITTER = dict(common=0.08, contact=0.06, mass=0.04, pos=0.0015, ang=3.0, eps=0.1)


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _logu(rng, lo, hi, size=None):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size))


def _rng(*key) -> np.random.Generator:
    """Generator for a derived, order-independent stream (key = tuple of ints)."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(k) & 0xFFFFFFFF for k in key])))


# --------------------------------------------------------------------------- phone

@dataclass(frozen=True)
class Phone:
    seed: int
    imu_mode: str                 # "none" (raw point sampling, carrier aliases) | "lpf" (low-pass before decimation)
    model: str
    mass: float                   # kg, includes the actuator's moving mass
    dims: tuple                   # (W, H, T) m
    inertia: tuple                # principal moments about the centre (kg m^2), device axes
    f0: float                     # Hz, actuator resonance
    q_lra: float                  # effective quality factor of the (closed-loop) actuator
    f_max: float                  # N, peak actuator force at intensity 1 driven at f0
    tap_cycles: float             # cycles in a transient at sharpness 0.5 (2..4)
    tap_asym: float               # net excursion of the moving mass during a transient (fraction of amplitude)
    tau_lra: float                # s, rise/fall time constant of continuous events
    act_axis: tuple               # unit force direction (mostly x, small cross-coupling)
    act_pos: tuple                # m, actuator position relative to the centre
    imu_pos: tuple                # m, IMU position relative to the centre
    latency_mean: float           # s, haptic onset latency after hapticStart
    latency_sd: float             # s, trial-to-trial latency jitter
    fs_imu: float                 # Hz, true sensor rate (nominal 100)
    accel_noise: float            # g, white noise per sample and axis at the 100 Hz output
    gyro_noise: float             # rad/s, white noise per sample and axis
    accel_bias: tuple             # g
    gyro_bias_sd: float           # rad/s, per-session raw gyro bias spread
    lpf_fc: float                 # Hz, anti-alias cutoff (lpf mode)
    lpf_order: int                # Butterworth order (lpf mode), 2 or 4
    gyro_offset: float            # s, gyro sampling offset relative to the accelerometer
    timestamp_jitter: float       # s, sd of sample timestamp jitter
    drop_rate: float              # per-sample probability of a single missing sample
    burst_rate: float             # per-recording probability of a short delivery gap (3..8 samples)
    long_gap_rate: float          # per-recording probability of a long gap (30..60 samples)
    dup_rate: float               # per-sample probability of a duplicated timestamp
    fidget_rate: float            # per-handheld-recording probability of a hand movement during the pattern
    table: dict = field(default_factory=dict)   # on-table support parameters

    def with_mode(self, imu_mode: str) -> "Phone":
        return make_phone(self.seed, imu_mode)


def make_phone(seed: int = 0, imu_mode: str = "none", **overrides) -> Phone:
    """Draw a phone. `imu_mode` is "none" (raw aliasing) or "lpf" (low-pass ~30-50 Hz before decimation).

    The phone's physical parameters depend only on `seed`; the IMU mode changes
    the anti-alias stage and the effective noise (aliased broadband noise is
    larger without the filter).
    """
    if imu_mode not in ("none", "lpf"):
        raise ValueError(f"imu_mode must be 'none' or 'lpf', got {imu_mode!r}")
    r = _rng(0x50484F4E, seed)
    W, H, T = r.uniform(0.068, 0.078), r.uniform(0.140, 0.160), r.uniform(0.0075, 0.0090)
    m = r.uniform(0.165, 0.235)
    inertia = (m * (H * H + T * T) / 12 * r.uniform(0.9, 1.1),
               m * (W * W + T * T) / 12 * r.uniform(0.9, 1.1),
               m * (W * W + H * H) / 12 * r.uniform(0.9, 1.1))
    axis = _unit([1.0, r.uniform(-0.12, 0.12), r.uniform(-0.12, 0.12)])
    act_pos = (r.uniform(-0.015, 0.015), r.uniform(-0.060, -0.045), r.uniform(-0.002, 0.002))
    imu_pos = (r.uniform(-0.015, 0.015), r.uniform(0.0, 0.030), r.uniform(-0.002, 0.002))
    noise_a, noise_g = r.uniform(0.0006, 0.0012), r.uniform(0.0005, 0.0010)
    alias_noise = r.uniform(1.4, 1.8)
    table = dict(
        kn=r.uniform(2e4, 1e5),           # N/m per corner
        kt_ratio=r.uniform(0.3, 1.0),
        zeta=r.uniform(0.02, 0.08),       # light damping
        inset=r.uniform(0.80, 0.92),      # corner positions as a fraction of the half-dimensions
        tilt_deg=r.uniform(0.0, 0.6),
    )
    p = Phone(
        seed=int(seed), imu_mode=imu_mode, model=f"sim-phone-{int(seed)}",
        mass=m, dims=(W, H, T), inertia=inertia,
        f0=r.uniform(140.0, 200.0), q_lra=r.uniform(2.0, 4.0), f_max=r.uniform(1.5, 3.0),
        tap_cycles=r.uniform(2.5, 3.5), tap_asym=r.uniform(0.2, 0.6), tau_lra=r.uniform(0.007, 0.014),
        act_axis=tuple(axis), act_pos=act_pos, imu_pos=imu_pos,
        latency_mean=r.uniform(0.012, 0.035), latency_sd=r.uniform(0.002, 0.004),
        fs_imu=100.0 * (1 + r.uniform(-0.003, 0.003)),
        accel_noise=noise_a * (alias_noise if imu_mode == "none" else 1.0),
        gyro_noise=noise_g * (alias_noise if imu_mode == "none" else 1.0),
        accel_bias=tuple(r.uniform(-0.008, 0.008, 3)), gyro_bias_sd=0.006,
        lpf_fc=r.uniform(30.0, 50.0), lpf_order=2,
        gyro_offset=r.uniform(0.0, 0.002), timestamp_jitter=r.uniform(0.0001, 0.0004),
        drop_rate=r.uniform(0.001, 0.004), burst_rate=0.05, long_gap_rate=0.005, dup_rate=0.002,
        fidget_rate=0.03, table=table,
    )
    return replace(p, **overrides) if overrides else p


# --------------------------------------------------------------------------- people

@dataclass(frozen=True, eq=False)
class HandParams:
    side: str
    pos: np.ndarray        # (N_CONTACTS, 3) m
    normal: np.ndarray     # (N_CONTACTS, 3) unit
    kn: np.ndarray         # N/m, normal stiffness
    kt_ratio: np.ndarray   # tangential / normal stiffness
    cn: np.ndarray         # N s/m, normal damping
    ct_ratio: np.ndarray   # tangential / normal damping
    mass: np.ndarray       # kg, effective hand mass coupled at each contact
    eps: np.ndarray        # cubic nonlinearity: fraction of the linear force at DELTA_REF
    angles: dict           # posture -> (tilt, inplane, side) in degrees
    standing_k: float      # stiffness factor when standing (grip changes with posture)
    tremor_scale: float    # hand-specific tremor factor


@dataclass(frozen=True, eq=False)
class Person:
    pid: str
    name: str
    seed: int
    dominant: str
    hands: dict            # side -> HandParams
    tremor_f: float        # Hz, physiological tremor peak (8-12)
    tremor_bw: float       # Hz
    tremor_acc: np.ndarray # (3,) g rms per axis
    tremor_gyro: np.ndarray  # (3,) rad/s rms per axis
    sway_deg: np.ndarray   # (3,) deg rms of slow sway per axis (sitting)
    sway_fc: float         # Hz, sway corner frequency
    standing_sway: float   # sway multiplier when standing
    consistency: float     # scales session and grip jitter (some people regrip more consistently)
    accessories: dict      # name -> parameters
    day_scale: float = 1.0  # extra factor on day-to-day (session) variability; 0 = every day is the same day


def _mirror_hand(right: HandParams, r: np.random.Generator, dominant: str) -> HandParams:
    """Left hand: the right hand mirrored in x, then perturbed with its own differences."""
    flip = np.array([-1.0, 1.0, 1.0])
    pos = right.pos * flip + r.normal(0, 0.002, right.pos.shape)
    normal = _unit(right.normal * flip + r.normal(0, 0.08, right.normal.shape))
    dom = 1.08 if dominant == "left" else 1 / 1.08
    angles = {k: (v[0] + r.normal(0, 4.0), -v[1] + r.normal(0, 3.0), -v[2] + r.normal(0, 3.0))
              for k, v in right.angles.items()}
    return HandParams(
        side="left", pos=pos, normal=normal,
        kn=right.kn * np.exp(r.normal(0, 0.15, N_CONTACTS)) * dom,
        kt_ratio=np.clip(right.kt_ratio + r.normal(0, 0.05, N_CONTACTS), 0.2, 0.9),
        cn=right.cn * np.exp(r.normal(0, 0.15, N_CONTACTS)) * dom,
        ct_ratio=np.clip(right.ct_ratio + r.normal(0, 0.05, N_CONTACTS), 0.2, 0.9),
        mass=right.mass * np.exp(r.normal(0, 0.10, N_CONTACTS)),
        eps=right.eps * np.exp(r.normal(0, 0.2, N_CONTACTS)),
        angles=angles, standing_k=float(np.exp(r.normal(0, 0.08))),
        tremor_scale=float(np.exp(r.normal(0, 0.15))),
    )


def make_person(seed: int, pid: str | None = None, name: str | None = None, dims=(0.0715, 0.147, 0.0078)) -> Person:
    """Draw one person (both hands, tremor, sway, own accessories) from `seed`.

    `dims` is the phone size the contact template is laid out on; the
    contacts are re-scaled to the actual phone when a trial is realized.
    """
    r = _rng(0x50455253, seed)
    W, H, T = dims
    scale = np.array([W, H, T])
    firm = np.exp(r.normal(0, 0.25))           # grip firmness: stiffness and damping together
    damp = np.exp(r.normal(0, 0.20))           # tissue damping
    size = np.exp(r.normal(0, 0.08))           # hand size: positions and masses
    eps_p = r.uniform(0.0, 0.5)
    pos = np.array([c[1] for c in _CONTACTS]) * scale
    jitter = r.normal(0, 0.003, pos.shape)
    for i, c in enumerate(_CONTACTS):
        jitter[i, c[3]] = 0.0                   # stay on the contact surface
    pos = pos + jitter
    pos[0, :2] *= size                          # palm sits further from the centre in a bigger hand
    normal = np.array([c[2] for c in _CONTACTS], float)
    normal[_FINGERS, 2] *= size                 # bigger hands wrap further around the back
    normal = _unit(normal + r.normal(0, 0.06, normal.shape))
    kn = _logu(r, _KN[:, 0], _KN[:, 1]) * firm
    cn = _logu(r, _CN[:, 0], _CN[:, 1]) * firm ** 0.5 * damp
    mass = r.uniform(_MASS[:, 0], _MASS[:, 1]) * size ** 3
    tilt = r.uniform(35.0, 62.0)
    angles = {"sitting": (tilt, r.normal(4.0, 5.0), r.normal(6.0, 6.0))}
    angles["standing"] = (tilt - r.uniform(3.0, 15.0), angles["sitting"][1] + r.normal(0, 2.0),
                          angles["sitting"][2] + r.normal(0, 2.0))
    dominant = "right" if r.uniform() < 0.88 else "left"
    right = HandParams(
        side="right", pos=pos, normal=normal, kn=kn, kt_ratio=r.uniform(0.3, 0.7, N_CONTACTS),
        cn=cn, ct_ratio=r.uniform(0.3, 0.8, N_CONTACTS), mass=mass,
        eps=eps_p * r.uniform(0.5, 1.5, N_CONTACTS), angles=angles,
        standing_k=float(np.exp(r.normal(0, 0.08))), tremor_scale=float(np.exp(r.normal(0, 0.15))),
    )
    if dominant == "right":
        right = replace(right, kn=right.kn * 1.08, cn=right.cn * 1.08)
    left = _mirror_hand(right, r, dominant)
    wacc = r.dirichlet([3.0, 3.0, 3.0])
    wgyr = r.dirichlet([3.0, 3.0, 3.0])
    tremor_acc = _logu(r, 0.0008, 0.0040) * np.sqrt(wacc)
    tremor_gyro = _logu(r, 0.002, 0.010) * np.sqrt(wgyr)
    accessories = {
        "watch": dict(mass=r.uniform(0.03, 0.07), coupling=r.uniform(0.15, 0.35), damp=r.uniform(1.0, 1.2),
                      angle=r.normal(0, 2.0)),
        "ring": dict(finger=int(r.choice(_FINGERS)), k=r.uniform(1.6, 3.0), c=r.uniform(0.6, 0.9),
                     mass=r.uniform(0.002, 0.006)),
        "bracelet": dict(mass=r.uniform(0.01, 0.03), coupling=r.uniform(0.1, 0.3), damp=r.uniform(1.0, 1.1)),
        "other": dict(mass=r.uniform(0.005, 0.03), coupling=r.uniform(0.1, 0.3), damp=r.uniform(1.0, 1.1)),
    }
    pid = pid if pid is not None else f"sim{int(seed) & 0xFFFFFFFF:08x}"
    return Person(
        pid=pid, name=name if name is not None else pid, seed=int(seed), dominant=dominant,
        hands={"right": right, "left": left},
        tremor_f=r.uniform(8.0, 12.0), tremor_bw=r.uniform(1.5, 3.5),
        tremor_acc=tremor_acc, tremor_gyro=tremor_gyro,
        sway_deg=r.uniform(0.15, 0.6, 3), sway_fc=r.uniform(0.4, 1.2), standing_sway=r.uniform(1.3, 2.2),
        consistency=float(_logu(r, 0.7, 1.4)), accessories=accessories,
    )


def make_population(n_people: int, seed: int = 0, prefix: str = "P", start: int = 1,
                    day_scale: float = 1.0) -> list[Person]:
    """`n_people` persons with ids/names `P01`, `P02`, ... drawn from `seed`.
    `day_scale` multiplies the day-to-day (session-level) variability of everyone."""
    ss = np.random.SeedSequence([0x504F5055, int(seed) & 0xFFFFFFFF])
    seeds = ss.generate_state(n_people, dtype=np.uint32)
    width = max(2, len(str(start + n_people - 1)))
    out = [make_person(int(s), pid=f"{prefix}{i:0{width}d}", name=f"{prefix}{i:0{width}d}")
           for i, s in zip(range(start, start + n_people), seeds)]
    return [replace(p, day_scale=float(day_scale)) for p in out] if day_scale != 1.0 else out


# --------------------------------------------------------------------------- realization

@dataclass
class Contacts:
    """Contact set of one trial (arrays of length nc)."""
    pos: np.ndarray
    normal: np.ndarray
    kn: np.ndarray
    kt: np.ndarray
    cn: np.ndarray
    ct: np.ndarray
    mass: np.ndarray
    k3: np.ndarray          # N/m^3, cubic normal stiffness


@dataclass
class TrialParams:
    contacts: Contacts
    gravity_angles: tuple   # (tilt, inplane, side) degrees, mean pose of this trial
    tremor_acc: np.ndarray  # (3,) g rms
    tremor_gyro: np.ndarray  # (3,) rad/s rms
    tremor_f: float
    tremor_bw: float
    sway_rad: np.ndarray    # (3,) rad rms
    sway_fc: float
    table: bool


def _jitter_hand(h: HandParams, r: np.random.Generator, s_common: float, s_contact: float,
                 s_mass: float, s_pos: float, s_ang: float, s_eps: float) -> HandParams:
    common = np.exp(r.normal(0, s_common))
    return replace(
        h,
        pos=h.pos + r.normal(0, s_pos, h.pos.shape),
        kn=h.kn * common * np.exp(r.normal(0, s_contact, N_CONTACTS)),
        cn=h.cn * common ** 0.7 * np.exp(r.normal(0, s_contact, N_CONTACTS)),
        mass=h.mass * np.exp(r.normal(0, s_mass, N_CONTACTS)),
        eps=h.eps * np.exp(r.normal(0, s_eps, N_CONTACTS)),
        angles={k: tuple(np.asarray(v) + r.normal(0, s_ang, 3)) for k, v in h.angles.items()},
        tremor_scale=h.tremor_scale * float(np.exp(r.normal(0, 0.25 if s_common > 0.1 else 0.1))),
    )


@lru_cache(maxsize=4096)
def _session_hand(person: Person, side: str, session: int) -> HandParams:
    """Day-level grip and tissue state of one hand (deterministic)."""
    c = person.consistency * person.day_scale
    j = SESSION_JITTER
    r = _rng(0x53455353, person.seed, HAND_ID[side], session)
    return _jitter_hand(person.hands[side], r, j["common"] * c, j["contact"] * c, j["mass"] * c, j["pos"] * c,
                        j["ang"] * c, j["eps"] * person.day_scale)


@lru_cache(maxsize=16384)
def _grip_hand(person: Person, side: str, session: int, grip: int) -> HandParams:
    """One grip instance (the phone picked up again) within a session (deterministic)."""
    c = person.consistency
    j = GRIP_JITTER
    r = _rng(0x47524950, person.seed, HAND_ID[side], session, grip)
    return _jitter_hand(_session_hand(person, side, session), r, j["common"] * c, j["contact"] * c, j["mass"] * c,
                        j["pos"] * c, j["ang"] * c, j["eps"])


def _scale_to_phone(pos: np.ndarray, phone: Phone, ref=(0.0715, 0.147, 0.0078)) -> np.ndarray:
    return pos * (np.asarray(phone.dims) / np.asarray(ref))


def _hand_contacts(h: HandParams, phone: Phone, r: np.random.Generator, accessory: str, person: Person,
                   k_factor: float = 1.0) -> Contacts:
    kn = h.kn * np.exp(r.normal(0, 0.03, N_CONTACTS)) * k_factor
    cn = h.cn * np.exp(r.normal(0, 0.03, N_CONTACTS)) * k_factor ** 0.7
    mass = h.mass.copy()
    pos = _scale_to_phone(h.pos, phone) + r.normal(0, 0.0003, h.pos.shape)
    acc = person.accessories.get(accessory) if accessory not in (None, "none") else None
    if acc is not None:
        if accessory == "ring":
            i = acc["finger"]
            kn[i] *= acc["k"]
            cn[i] *= acc["c"]
            mass[i] += acc["mass"]
        else:  # watch, bracelet, other: extra mass near the wrist, coupled through the palm
            mass[0] += acc["mass"] * acc["coupling"]
            cn[0] *= acc["damp"]
    kt = kn * h.kt_ratio
    ct = cn * h.ct_ratio
    k3 = h.eps * kn / DELTA_REF ** 2
    return Contacts(pos=pos, normal=h.normal.copy(), kn=kn, kt=kt, cn=cn, ct=ct, mass=mass, k3=k3)


def _concat(a: Contacts, b: Contacts) -> Contacts:
    return Contacts(*[np.concatenate([getattr(a, f), getattr(b, f)]) for f in Contacts.__dataclass_fields__])


def table_contacts(phone: Phone, session: int, r: np.random.Generator) -> Contacts:
    """Phone lying on a hard table: four stiff, lightly damped corner supports, no added mass."""
    t = phone.table
    rs = _rng(0x5441424C, phone.seed, session)
    W, H, T = phone.dims
    sx, sy = t["inset"] * W / 2, t["inset"] * H / 2
    pos = np.array([[sx, sy, -T / 2], [-sx, sy, -T / 2], [sx, -sy, -T / 2], [-sx, -sy, -T / 2]])
    kn = t["kn"] * np.exp(rs.normal(0, 0.15, 4)) * np.exp(r.normal(0, 0.02, 4))
    m_quarter = phone.mass / 4
    cn = 2 * t["zeta"] * np.sqrt(kn * m_quarter)
    return Contacts(pos=pos, normal=np.tile([0.0, 0.0, 1.0], (4, 1)), kn=kn, kt=kn * t["kt_ratio"],
                    cn=cn, ct=cn * t["kt_ratio"], mass=np.zeros(4), k3=np.zeros(4))


def realize(person: Person | None, hand: str, phone: Phone, rng: np.random.Generator, *, posture: str = "sitting",
            accessory: str = "none", session: int = 0, condition: str = "handheld", grip: int | None = None,
            angles_from: Person | None = None) -> TrialParams:
    """Draw the physical parameters of one trial.

    Hierarchy: person/hand (fixed) -> session (day) -> grip instance -> trial.
    `grip=None` draws a fresh grip from `rng`; an int selects a reproducible grip
    within the session (use the same int for all patterns of one check).
    `angles_from` copies another person's grip angles (mimicry proxy).
    """
    if condition == "onTable":
        tilt = phone.table["tilt_deg"]
        return TrialParams(contacts=table_contacts(phone, session, rng), gravity_angles=(90.0 - tilt, 0.0, 0.0),
                           tremor_acc=np.zeros(3), tremor_gyro=np.zeros(3), tremor_f=10.0, tremor_bw=2.0,
                           sway_rad=np.zeros(3), sway_fc=1.0, table=True)
    if person is None:
        raise ValueError("a person is required for handheld conditions")
    phys_posture = "standing" if posture == "standing" else "sitting"

    def grip_of(side):
        if grip is None:
            g = int(rng.integers(1 << 30)) | (1 << 30)    # fresh, never collides with small user ints
        else:
            g = int(grip)
        return _grip_hand(person, side, int(session), g)

    sides = ("right", "left") if condition == "twoHanded" else (hand,)
    contacts, angs, tscale = None, [], []
    for side in sides:
        h = grip_of(side)
        kf = h.standing_k if phys_posture == "standing" else 1.0
        if condition == "twoHanded":
            kf *= 0.65
            h = replace(h, mass=h.mass * 0.8)
        c = _hand_contacts(h, phone, rng, accessory, person, kf)
        contacts = c if contacts is None else _concat(contacts, c)
        a = np.asarray(h.angles[phys_posture], float)
        if angles_from is not None:   # victim's typical angles plus the attacker's own grip deviation
            own = np.asarray(person.hands[side].angles[phys_posture], float)
            a = np.asarray(angles_from.hands[side].angles[phys_posture], float) + (a - own)
        if accessory == "watch":
            a = a + np.array([person.accessories["watch"]["angle"], 0.0, 0.0])
        angs.append(a)
        tscale.append(h.tremor_scale)
    ang = np.mean(angs, axis=0)
    if condition == "twoHanded":
        ang[1:] *= 0.3
    ang = ang + rng.normal(0, 0.7, 3)              # pattern-to-pattern pose wobble within a grip
    ang[0] = float(np.clip(ang[0], 10.0, 70.0))
    tf = float(np.mean(tscale)) * float(np.exp(rng.normal(0, 0.1)))
    sway = np.deg2rad(person.sway_deg) * (person.standing_sway if phys_posture == "standing" else 1.0)
    if condition == "twoHanded":
        tf *= 0.7
        sway = sway * 0.8
    return TrialParams(contacts=contacts, gravity_angles=tuple(ang), tremor_acc=person.tremor_acc * tf,
                       tremor_gyro=person.tremor_gyro * tf, tremor_f=person.tremor_f, tremor_bw=person.tremor_bw,
                       sway_rad=sway, sway_fc=person.sway_fc, table=False)
