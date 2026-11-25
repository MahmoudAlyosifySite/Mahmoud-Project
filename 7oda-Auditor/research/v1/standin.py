"""Tiny stand-in recording generator, for smoke tests of the v1 pipeline only.

It is NOT the physics simulator (``research/sim``) and is never used to train
or evaluate a reported model. It produces Recording dicts in the SPEC 4
format with a challenge from SPEC 3 (``auditor_ref.challenge.generate`` when
available, otherwise the small port below), a damped-oscillator response per
(person, hand), grip jitter, timestamp jitter, optional duplicates, unsorted
samples and delivery gaps, so the data path, the tensor builder and the
training loops can be exercised before the real simulator exists.
"""
from __future__ import annotations

import math
import uuid

import numpy as np

M64 = 0xFFFFFFFFFFFFFFFF


class _SplitMix64:
    def __init__(self, seed: int):
        self.s = seed & M64

    def next(self) -> int:
        self.s = (self.s + 0x9E3779B97F4A7C15) & M64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)


class _Xoshiro:
    def __init__(self, seed: int):
        sm = _SplitMix64(seed)
        self.s = [sm.next() for _ in range(4)]

    @staticmethod
    def _rotl(x, k):
        return ((x << k) | (x >> (64 - k))) & M64

    def next(self) -> int:
        s = self.s
        result = (self._rotl((s[1] * 5) & M64, 7) * 9) & M64
        t = (s[1] << 17) & M64
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = self._rotl(s[3], 45)
        return result

    def uniform(self, a=None, b=None):
        u = (self.next() >> 11) * 2.0 ** -53
        return u if a is None else a + (b - a) * u

    def int(self, lo, hi):
        return lo + min(int(math.floor(self.uniform() * (hi - lo + 1))), hi - lo)


def _challenge_port(seed: int) -> dict:
    rng = _Xoshiro(seed)
    lead = rng.uniform(0.05, 0.10)
    n = rng.int(6, 8)
    s1 = 1 + min(int(math.floor(rng.uniform() * n)), n - 1)
    s2 = 1 + min(int(math.floor(rng.uniform() * (n - 1))), n - 2)
    if s2 >= s1:
        s2 += 1
    slots = sorted([s1, s2])
    taps = [dict(intensity=rng.uniform(0.4, 1.0), sharpness=rng.uniform(0.0, 1.0),
                 gap=rng.uniform(0.08, 0.15)) for _ in range(n)]
    ramps = []
    for _ in range(2):
        ramps.append(dict(duration=rng.uniform(0.15, 0.25), sharpness=rng.uniform(0.2, 0.8),
                          ascending=rng.uniform() < 0.5, low=rng.uniform(0.25, 0.5),
                          mid=rng.uniform(0.25, 1.0), high=rng.uniform(0.75, 1.0),
                          gap=rng.uniform(0.08, 0.12)))
    t = lead
    events = []
    for i in range(n):
        events.append({"kind": "tap", "time": t, "intensity": taps[i]["intensity"],
                       "sharpness": taps[i]["sharpness"]})
        t += taps[i]["gap"]
        for j in range(2):
            if slots[j] == i + 1:
                r = ramps[j]
                d = r["duration"]
                pts = ([(0.0, r["low"]), (d / 2, r["mid"]), (d, r["high"])] if r["ascending"]
                       else [(0.0, r["high"]), (d / 2, r["mid"]), (d, r["low"])])
                events.append({"kind": "ramp", "time": t, "duration": d, "sharpness": r["sharpness"],
                               "curve": [{"time": a, "intensity": b} for a, b in pts]})
                t += d + r["gap"]
    return {"version": 1, "seed": "0x%016x" % (seed & M64), "duration": t, "events": events}


def make_challenge(seed: int) -> dict:
    try:
        from auditor_ref import challenge as ref_challenge
        return ref_challenge.generate(seed)
    except Exception:  # reference not available yet
        return _challenge_port(seed)


def person_params(rng: np.random.Generator) -> dict:
    """Random 'hand' for the stand-in: oscillator + coupling + grip angle."""
    return {
        "f0": rng.uniform(14.0, 40.0),
        "zeta": rng.uniform(0.08, 0.45),
        "f_alias": rng.uniform(18.0, 45.0),
        "gamma": rng.uniform(0.8, 1.6),
        "acc_gain": rng.uniform(0.003, 0.02) * rng.dirichlet(np.ones(3)) * 3,
        "gyr_gain": rng.uniform(0.01, 0.06) * rng.dirichlet(np.ones(3)) * 3,
        "tilt": rng.uniform(0.15, 0.6),
        "pitch": rng.uniform(0.3, 1.0),
    }


def hand_params(person: dict, hand: str, rng: np.random.Generator) -> dict:
    p = dict(person)
    s = 1.0 if hand == "right" else -1.0
    p["tilt"] = s * person["tilt"]
    p["f0"] = person["f0"] * rng.uniform(0.9, 1.1)
    p["zeta"] = person["zeta"] * rng.uniform(0.85, 1.15)
    p["acc_gain"] = person["acc_gain"] * np.array([1.0, 1.0, 1.0]) * rng.uniform(0.8, 1.2, 3)
    p["gyr_gain"] = person["gyr_gain"] * np.array([1.0, s, s]) * rng.uniform(0.8, 1.2, 3)
    return p


def simulate_recording(hp: dict, seed: int, rng: np.random.Generator, label: dict | None = None,
                       jitter: float = 0.05, messy: bool = False, fs: float = 100.0) -> dict:
    """One Recording dict (SPEC 4) for hand parameters ``hp``."""
    ch = make_challenge(seed)
    pre, post = 0.35, 0.35
    hs = 1000.0 + rng.uniform(0, 100)
    dur = ch["duration"]
    nt = int((pre + dur + post) * fs) + 2
    t = hs - pre + np.arange(nt) / fs + rng.normal(0, 0.0008, nt)
    tau = t - hs
    f0 = hp["f0"] * (1 + jitter * rng.normal())
    zeta = hp["zeta"] * (1 + jitter * rng.normal())
    wd = 2 * math.pi * f0 * math.sqrt(max(1 - zeta ** 2, 0.05))
    resp = np.zeros(nt)
    for ev in ch["events"]:
        if ev["kind"] == "tap":
            dt = tau - ev["time"]
            on = dt >= 0
            amp = ev["intensity"] ** hp["gamma"]
            resp[on] += amp * np.exp(-zeta * 2 * math.pi * f0 * dt[on]) * np.sin(wd * dt[on])
        else:
            d = ev["duration"]
            dt = tau - ev["time"]
            xs = [c["time"] for c in ev["curve"]]
            ys = [c["intensity"] for c in ev["curve"]]
            env = np.where((dt >= 0) & (dt < d), np.interp(dt, xs, ys) ** hp["gamma"], 0.0)
            tail = dt >= d
            env[tail] = (ys[-1] ** hp["gamma"]) * np.exp(-(dt[tail] - d) * zeta * 2 * math.pi * f0)
            resp += 0.6 * env * np.sin(2 * math.pi * hp["f_alias"] * dt)
    tremor = 0.05 * np.sin(2 * math.pi * rng.uniform(8, 12) * tau + rng.uniform(0, 6.28))
    resp = resp + tremor
    tilt = hp["tilt"] + jitter * rng.normal()
    pitch = hp["pitch"] + jitter * rng.normal()
    grav = np.array([math.sin(tilt) * math.cos(pitch), -math.sin(pitch), -math.cos(tilt) * math.cos(pitch)])
    grav = grav / np.linalg.norm(grav)
    ag = hp["acc_gain"] * (1 + jitter * rng.normal(size=3))
    gg = hp["gyr_gain"] * (1 + jitter * rng.normal(size=3))
    acc = grav[:, None] + ag[:, None] * resp[None, :] + rng.normal(0, 0.0015, (3, nt))
    gyr = gg[:, None] * resp[None, :] + rng.normal(0, 0.002, (3, nt))
    tm = t + rng.normal(0, 0.0003, nt)
    grav_s = grav[:, None] + rng.normal(0, 0.001, (3, nt))
    ta, tg = t.copy(), t + rng.normal(0, 0.0004, nt)
    if messy:
        k = rng.integers(10, nt - 10)
        drop = slice(k, k + rng.integers(2, 5))          # delivery gap
        keep = np.ones(nt, bool)
        keep[drop] = False
        ta, acc = ta[keep], acc[:, keep]
        d = rng.integers(5, len(ta) - 5)                 # duplicate a timestamp
        ta = np.insert(ta, d, ta[d])
        acc = np.insert(acc, d, acc[:, d], axis=1)
        perm = np.arange(len(tg))                        # swap two samples (unsorted)
        i = rng.integers(5, len(tg) - 5)
        perm[i], perm[i + 1] = perm[i + 1], perm[i]
        tg, gyr = tg[perm], gyr[:, perm]
    return {
        "schema": "7oda-auditor.recording.v1",
        "id": str(uuid.UUID(int=int(rng.integers(0, 2 ** 62)))),
        "createdAt": "2026-11-24T10:00:00Z",
        "purpose": "check",
        "label": label,
        "challenge": ch,
        "hapticStart": hs,
        "accel": {"t": ta.tolist(), "x": acc[0].tolist(), "y": acc[1].tolist(), "z": acc[2].tolist()},
        "gyro": {"t": tg.tolist(), "x": gyr[0].tolist(), "y": gyr[1].tolist(), "z": gyr[2].tolist()},
        "motion": dict({"t": tm.tolist(), "gravityX": grav_s[0].tolist(), "gravityY": grav_s[1].tolist(),
                        "gravityZ": grav_s[2].tolist()},
                       **{f"userAccel{a}": (acc[i][:0].tolist() + (ag[i] * resp).tolist())
                          for i, a in enumerate("XYZ")},
                       **{f"rotation{a}": (gg[i] * resp).tolist() for i, a in enumerate("XYZ")},
                       attitudeW=[1.0] * nt, attitudeX=[0.0] * nt, attitudeY=[0.0] * nt,
                       attitudeZ=[0.0] * nt),
    }


def generate(n_people: int, per_hand: int, seed: int = 0, sessions: int = 1, messy: bool = False,
             prefix: str = "S") -> list:
    """Recordings for ``n_people`` stand-in people, both hands, ``per_hand`` per session."""
    rng = np.random.default_rng(seed)
    recs = []
    for pi in range(n_people):
        person = person_params(rng)
        hands = {h: hand_params(person, h, rng) for h in ("right", "left")}
        for s in range(sessions):
            drift = {h: dict(hp, f0=hp["f0"] * rng.uniform(0.97, 1.03)) for h, hp in hands.items()}
            for h in ("right", "left"):
                for k in range(per_hand):
                    label = {"personID": f"{prefix}{pi:04d}", "personName": f"{prefix}{pi:04d}",
                             "hand": h, "accessory": "none", "posture": "unspecified",
                             "sessionID": f"{prefix}{pi:04d}-s{s}"}
                    recs.append(simulate_recording(drift[h], int(rng.integers(0, 2 ** 63)), rng,
                                                   label=label, messy=messy))
    return recs
