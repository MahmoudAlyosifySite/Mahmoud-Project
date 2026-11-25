"""simulate / simulate_batch: trials -> Recording dicts in the SPEC section 4 form."""
from __future__ import annotations

import datetime as _dt
import uuid
from typing import Any, Sequence

import numpy as np

from . import actuator, dynamics, imu
from .params import G0, Person, Phone, _rng, realize

PRE_WINDOW = 0.35        # snapshot starts preRoll + 0.1 s before hapticStart (SPEC 12, CheckRunner)
POST_WINDOW = 0.35
SIM_R0 = -0.36           # simulation grid start relative to hapticStart
SCHEMA = "7oda-auditor.recording.v1"
BASE_DATE = _dt.datetime(2026, 12, 1, 9, 0, 0, tzinfo=_dt.timezone.utc)


def _challenge(c) -> dict:
    if isinstance(c, dict):
        return c
    from auditor_ref.challenge import generate
    return generate(int(c))


def _bcast(v, n: int, name: str) -> list:
    if isinstance(v, (str, bytes, dict, Person)) or v is None or np.isscalar(v):
        return [v] * n
    v = list(v)
    if len(v) != n:
        raise ValueError(f"{name}: expected {n} values, got {len(v)}")
    return v


def session_clock(phone: Phone, session: int):
    """Deterministic (uptime base in s, UTC datetime at that uptime) of a session ("day")."""
    r = _rng(0x434C4F43, phone.seed, session)
    up = float(r.uniform(2e3, 4e5))
    day = BASE_DATE + _dt.timedelta(days=int(session) * 3 + int(r.integers(0, 2)), seconds=float(r.uniform(0, 8 * 3600)))
    return up, day


def iso(d: _dt.datetime) -> str:
    return d.astimezone(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def simulate_batch(persons, hands, challenges, phone: Phone, rng: np.random.Generator | None = None, *,
                   postures="sitting", accessories="none", sessions=0, conditions="handheld", grips=None,
                   purposes=None, angles_from=None, seeds=None, start_times=None, include_meta: bool = True,
                   sim_rate: float = 2000.0, fidget=None):
    """Simulate B trials at once (vectorised integration). Returns a list of Recording dicts.

    Every argument may be a single value (shared) or a sequence of length B:
      persons      Person (or None for onTable)
      hands        "right" | "left" (label hand; for twoHanded the person's both hands are used)
      challenges   ChallengeSpec dict or int seed (auditor_ref.challenge.generate)
      postures     "sitting" | "standing" | "freshGrip" | "unspecified"
      accessories  "none" | "watch" | "ring" | "bracelet" | "other"
      sessions     int day index (session-level grip state, clock, gyro bias)
      conditions   "handheld" | "twoHanded" | "onTable"
      grips        int grip instance within the session (same int = same grip), None = fresh grip
      purposes     SPEC purpose; default "calibration" for onTable else "check"
      angles_from  Person whose grip angles are copied (mimicry), or None
      seeds        per-trial seeds (default: drawn from rng); a trial is a pure function of its inputs
                   and its seed, independent of the rest of the batch
      start_times  hapticStart (s, uptime clock); default: random within the session
      fidget       force (True/False) a hand movement during the pattern; default random (phone.fidget_rate)
    """
    lengths = [len(x) for x in (persons, hands, challenges, postures, accessories, sessions, conditions, grips,
                                purposes, angles_from, seeds, start_times, fidget)
               if x is not None and not isinstance(x, (str, bytes, dict, Person)) and not np.isscalar(x)]
    B = max(lengths) if lengths else 1
    if rng is None:
        rng = np.random.Generator(np.random.PCG64(0))
    persons = _bcast(persons, B, "persons")
    hands = _bcast(hands, B, "hands")
    chs = [_challenge(c) for c in _bcast(challenges, B, "challenges")]
    postures = _bcast(postures, B, "postures")
    accessories = _bcast(accessories, B, "accessories")
    sessions = [int(s) for s in _bcast(sessions, B, "sessions")]
    conditions = _bcast(conditions, B, "conditions")
    grips = _bcast(grips, B, "grips")
    purposes = _bcast(purposes, B, "purposes")
    angles_from = _bcast(angles_from, B, "angles_from")
    fidget = _bcast(fidget, B, "fidget")
    seeds = [int(s) for s in (_bcast(seeds, B, "seeds") if seeds is not None else rng.integers(0, 2 ** 63, B))]
    start_times = _bcast(start_times, B, "start_times")
    rngs = [np.random.Generator(np.random.PCG64(s)) for s in seeds]

    # ---- per-trial physical parameters
    tps, lat, f0, amp = [], np.empty(B), np.empty(B), np.empty(B)
    for i in range(B):
        r = rngs[i]
        tps.append(realize(persons[i], hands[i], phone, r, posture=postures[i], accessory=accessories[i],
                           session=sessions[i], condition=conditions[i], grip=grips[i], angles_from=angles_from[i]))
        lat[i] = np.clip(phone.latency_mean + r.normal(0, phone.latency_sd), 0.010, 0.040)
        drift = _rng(0x46524551, phone.seed, sessions[i]).normal(0, 0.005)
        f0[i] = phone.f0 * (1 + drift + r.normal(0, 0.001))
        amp[i] = np.exp(r.normal(0, 0.03))

    # ---- actuator force on the half-step grid, integrate
    h = 1.0 / sim_rate
    dur = np.array([c["duration"] for c in chs])
    T = int(np.ceil((dur.max() + PRE_WINDOW + POST_WINDOW + 0.03) / h)) + 2
    u_half = actuator.force_waveform(chs, lat, phone, SIM_R0, h / 2, 2 * T + 1, f0=f0, amp=amp, zoh_step=h)
    u_grid = np.ascontiguousarray(u_half[:, 0:2 * T:2].T)
    u_mid = np.ascontiguousarray(u_half[:, 1:2 * T:2].T)
    first = min(min(e["time"] for e in c["events"]) for c in chs)
    k_start = max(0, int(np.floor((lat.min() + first - SIM_R0) / h)) - 2)
    disc = dynamics.discretize(dynamics.build_system([tp.contacts for tp in tps], phone), h)
    y = dynamics.integrate(disc, u_grid, u_mid, k_start)
    if phone.imu_mode == "lpf":
        y = imu.sosfilt_time(imu.butter_lowpass_sos(phone.lpf_fc, sim_rate, phone.lpf_order), y, k_start)

    fid = [((not tps[i].table) and rngs[i].uniform() < phone.fidget_rate) if fidget[i] is None else bool(fidget[i])
           for i in range(B)]
    motion = imu.hand_motion_batch(tps, rngs, float(dur.max()), fid)
    out = []
    for i in range(B):
        mo = dict(t=motion["t"], acc=motion["acc"][i], gyro=motion["gyro"][i], theta=motion["theta"][i],
                  fidget=motion["fidget"][i])
        out.append(_assemble(i, y[:, i, :], h, chs[i], tps[i], persons[i], hands[i], postures[i], accessories[i],
                             sessions[i], conditions[i], grips[i], purposes[i], phone, rngs[i], lat[i], f0[i],
                             seeds[i], start_times[i], mo, include_meta))
    return out


def simulate(person: Person | None, hand: str, challenge_spec, phone: Phone, rng: np.random.Generator, *,
             posture: str = "sitting", accessory: str = "none", session: int = 0, condition: str = "handheld",
             grip: int | None = None, purpose: str | None = None, angles_from: Person | None = None,
             start_time: float | None = None, include_meta: bool = True, fidget=None) -> dict:
    """One recording (same model as simulate_batch with B = 1)."""
    return simulate_batch([person], [hand], [challenge_spec], phone, rng, postures=[posture], accessories=[accessory],
                          sessions=[session], conditions=[condition], grips=[grip], purposes=[purpose],
                          angles_from=[angles_from], start_times=[start_time], include_meta=include_meta,
                          fidget=[fidget])[0]


def _assemble(i, yb, h, ch, tp, person, hand, posture, accessory, session, condition, grip, purpose, phone, r,
              latency, f0, seed, start_time, motion, include_meta):
    D = ch["duration"]
    up0, day0 = session_clock(phone, session)
    hs = float(start_time) if start_time is not None else up0 + float(r.uniform(5.0, 3000.0))
    t_lo, t_hi = -PRE_WINDOW, D + POST_WINDOW
    ts = 1.0 / phone.fs_imu
    phase = r.uniform(0, ts)

    # sampling instants (relative to hapticStart); gaps are shared, single drops per stream
    t_a = imu.sample_times(r, t_lo, t_hi, phone.fs_imu, phase, phone.timestamp_jitter)
    t_g = imu.sample_times(r, t_lo, t_hi, phone.fs_imu, (phase + phone.gyro_offset) % ts, phone.timestamp_jitter)
    t_m = imu.sample_times(r, t_lo, t_hi, phone.fs_imu, r.uniform(0, ts), phone.timestamp_jitter)
    windows = imu.gap_windows(r, t_lo, t_hi, phone.fs_imu, phone.burst_rate, phone.long_gap_rate)
    t_a = t_a[imu.gap_mask(r, t_a, windows, phone.drop_rate)]
    t_g = t_g[imu.gap_mask(r, t_g, windows, phone.drop_rate)]
    t_m = t_m[imu.gap_mask(r, t_m, windows, phone.drop_rate)]

    def vib(t, cols):
        k = np.clip(np.rint((t - SIM_R0) / h).astype(int), 0, yb.shape[0] - 1)
        return yb[k][:, cols].T.astype(float)

    # pose: mean grip pose, arbitrary heading, sway/fidget rotation (device-frame perturbation)
    yaw = float(_rng(0x59415720, phone.seed, session).uniform(0, 360)) + r.normal(0, 3.0)
    R0 = imu.pose_matrix(*tp.gravity_angles, yaw_deg=yaw)
    q0 = imu.mat_to_quat(R0)
    qs = imu.quat_mul(q0[None, :], imu.quat_from_rotvec(motion["theta"].T))     # (M, 4)
    grav_grid = imu.quat_to_gravity(qs).T                                       # (3, M)

    rs = _rng(0x42494153, phone.seed, session)
    gyro_bias = rs.normal(0, phone.gyro_bias_sd, 3)
    accel_bias = np.asarray(phone.accel_bias) + rs.normal(0, 0.001, 3)

    def unit(v):
        return v / np.linalg.norm(v, axis=0, keepdims=True)

    grav_a = unit(imu.interp_rows(motion["t"], grav_grid, t_a))
    acc = (grav_a - vib(t_a, [0, 1, 2]) / G0 - imu.interp_rows(motion["t"], motion["acc"], t_a)
           + accel_bias[:, None] + r.normal(0, phone.accel_noise, (3, t_a.size)))
    gyr = (vib(t_g, [3, 4, 5]) + imu.interp_rows(motion["t"], motion["gyro"], t_g) + gyro_bias[:, None]
           + r.normal(0, phone.gyro_noise, (3, t_g.size)))
    user = (-vib(t_m, [0, 1, 2]) / G0 - imu.interp_rows(motion["t"], motion["acc"], t_m)
            + r.normal(0, phone.accel_noise, (3, t_m.size)))
    rot = (vib(t_m, [3, 4, 5]) + imu.interp_rows(motion["t"], motion["gyro"], t_m)
           + r.normal(0, 3e-4, 3)[:, None] + r.normal(0, phone.gyro_noise, (3, t_m.size)))
    att = imu.interp_rows(motion["t"], qs.T, t_m)
    att = att / np.linalg.norm(att, axis=0, keepdims=True)
    grav_m = imu.quat_to_gravity(att.T).T                       # fusion gravity, exactly consistent with attitude

    ia = imu.duplicate_indices(r, t_a.size, phone.dup_rate)
    ig = imu.duplicate_indices(r, t_g.size, phone.dup_rate)
    im = imu.duplicate_indices(r, t_m.size, phone.dup_rate)
    rid = str(uuid.UUID(bytes=r.bytes(16), version=4)).upper()
    created = day0 + _dt.timedelta(seconds=hs - up0)
    if purpose is None:
        purpose = "calibration" if condition == "onTable" else "check"
    label = None
    if condition != "onTable" and person is not None:
        label = {"personID": person.pid, "personName": person.name, "hand": hand, "accessory": accessory or "none",
                 "posture": posture, "sessionID": f"{person.pid}-S{session}"}

    def L(a):
        return a.tolist()

    rec = {
        "schema": SCHEMA, "id": rid, "createdAt": iso(created), "purpose": purpose, "label": label,
        "challenge": ch, "hapticStart": hs,
        "accel": {"t": L(hs + t_a[ia]), "x": L(acc[0, ia]), "y": L(acc[1, ia]), "z": L(acc[2, ia])},
        "gyro": {"t": L(hs + t_g[ig]), "x": L(gyr[0, ig]), "y": L(gyr[1, ig]), "z": L(gyr[2, ig])},
        "motion": {"t": L(hs + t_m[im]),
                   "gravityX": L(grav_m[0, im]), "gravityY": L(grav_m[1, im]), "gravityZ": L(grav_m[2, im]),
                   "userAccelX": L(user[0, im]), "userAccelY": L(user[1, im]), "userAccelZ": L(user[2, im]),
                   "rotationX": L(rot[0, im]), "rotationY": L(rot[1, im]), "rotationZ": L(rot[2, im]),
                   "attitudeW": L(att[0, im]), "attitudeX": L(att[1, im]), "attitudeY": L(att[2, im]),
                   "attitudeZ": L(att[3, im])},
        "device": {"model": phone.model, "system": "sim", "app": "7oda-sim 1"},
    }
    if include_meta:
        grip_kind = "table" if condition == "onTable" else ("twoHanded" if condition == "twoHanded" else hand)
        rec["sim"] = {"version": 1, "seed": str(seed), "condition": condition, "grip": grip_kind,
                      "session": session, "gripInstance": None if grip is None else int(grip),
                      "personSeed": None if person is None else person.seed, "phoneSeed": phone.seed,
                      "imuMode": phone.imu_mode, "latency": float(latency), "f0": float(f0),
                      "gravityAngles": [float(a) for a in tp.gravity_angles], "fidget": motion["fidget"]}
    return rec


def simulate_random(n: int, seed: int, phone: Phone | None = None, people: Sequence[Person] | None = None, *,
                    n_people: int = 1000, imu_mode: str = "none", p_standing: float = 0.5,
                    p_accessory: float = 0.15, p_two_handed: float = 0.0, p_table: float = 0.0,
                    n_sessions: int = 4, batch: int = 512, include_meta: bool = True) -> list[dict]:
    """`n` random labelled recordings (pretraining helper). Each trial draws a person, hand,
    posture, accessory, session and a fresh grip; conditions are handheld, twoHanded (p_two_handed)
    or onTable (p_table). Deterministic in `seed`; phone defaults to make_phone(seed, imu_mode)."""
    r = np.random.Generator(np.random.PCG64(np.random.SeedSequence([0x52414E44, int(seed) & 0xFFFFFFFF])))
    phone = phone if phone is not None else make_phone_cached(int(seed), imu_mode)
    if people is None:
        from .params import make_population
        people = make_population(n_people, seed=int(seed))
    out = []
    for s in range(0, n, batch):
        m = min(batch, n - s)
        u = r.uniform(size=m)
        cond = np.where(u < p_table, "onTable", np.where(u < p_table + p_two_handed, "twoHanded", "handheld"))
        idx = r.integers(0, len(people), m)
        acc_names = np.array(["watch", "ring", "bracelet"])
        acc = np.where(r.uniform(size=m) < p_accessory, acc_names[r.integers(0, 3, m)], "none")
        out += simulate_batch([None if c == "onTable" else people[int(i)] for c, i in zip(cond.tolist(), idx)],
                              np.where(r.uniform(size=m) < 0.5, "right", "left").tolist(),
                              [int(x) for x in r.integers(0, 2 ** 63, m)], phone, r,
                              postures=np.where(r.uniform(size=m) < p_standing, "standing", "sitting").tolist(),
                              accessories=acc.tolist(), sessions=[int(x) for x in r.integers(0, n_sessions, m)],
                              conditions=cond.tolist(), include_meta=include_meta)
    return out


def make_phone_cached(seed: int, imu_mode: str) -> Phone:
    from .params import make_phone
    return make_phone(seed, imu_mode)
