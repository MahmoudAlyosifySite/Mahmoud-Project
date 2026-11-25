"""Adapter from ``v1.simdata`` plans to the physics simulator (``research/sim``).

* people: ``sim.params.make_person(seed)`` with ``seed = person_seed_base +
  index`` (index parsed from the person id), so train, validation and
  evaluation populations never overlap when their bases differ;
* phones: training draws a phone per person from a pool
  (``phone_pool`` seeds starting at ``phone_seed_base``); protocol sets use a
  single phone (``phone_seed``), as the study uses one iPhone; the IMU mode
  (``imu_mode``: ``none`` = raw aliasing, ``lpf`` = anti-alias low-pass) is
  an option; ``imu_mode=mixed`` alternates per person for training;
* grips: rounds of 5 patterns share a grip (``k // 5`` within a session and
  hand); a check's patterns share one grip; a fresh grip between rounds and
  checks, as in the proposal's enrollment and evaluation protocol;
* two-handed recordings use the simulator's ``twoHanded`` condition (both
  of the person's hands on the phone); their label hand is stored as
  ``twoHanded`` for the hand head and they get no identity class.
"""
from __future__ import annotations

import re

import numpy as np

CHUNK = 256


def _index(pid: str) -> int:
    m = re.search(r"(\d+)$", pid)
    return int(m.group(1)) if m else abs(hash(pid)) % (10 ** 9)


def generate(person_ids, plans, seed: int, options: dict):
    import sim

    rng = np.random.default_rng(seed)
    base = int(options.get("person_seed_base", 0))
    mode = options.get("imu_mode", "none")
    layout_phone = options.get("phone_seed")
    pool = int(options.get("phone_pool", 200))
    pbase = int(options.get("phone_seed_base", 1000))
    phones = {}

    def phone_for(i_person: int):
        if layout_phone is not None:
            key = (int(layout_phone), mode if mode != "mixed" else "none")
        else:
            m = mode if mode != "mixed" else ("none" if i_person % 2 == 0 else "lpf")
            key = (pbase + int(rng.integers(0, pool)), m)
        if key not in phones:
            phones[key] = sim.make_phone(key[0], key[1])
        return phones[key]

    jobs = []   # (phone, person, hand_label, condition, posture, accessory, session, grip, purpose, meta)
    for pid, plan in zip(person_ids, plans):
        idx = _index(pid)
        person = sim.make_person(base + idx, pid, pid)
        phone = phone_for(idx)
        counters = {}
        for (s, hand, posture, acc, check, pattern) in plan:
            key = (s, hand, check >= 0)
            k = counters.get(key, 0)
            counters[key] = k + 1
            if check >= 0:
                grip = 1000 + check
            else:
                grip = k // 5
            if hand == "twoHanded":
                cond, label_hand, grip = "twoHanded", "right", None
            else:
                cond, label_hand = "handheld", hand
            purpose = "enrollment" if check < 0 else "check"
            meta = {"hand": hand, "check": check, "pattern": pattern, "session": s}
            jobs.append((phone, person, label_hand, cond, posture, acc, s, grip, purpose, meta))

    out = []
    # group by phone so each simulate_batch call shares one phone
    by_phone = {}
    for j in jobs:
        by_phone.setdefault(id(j[0]), []).append(j)
    for group in by_phone.values():
        phone = group[0][0]
        for c in range(0, len(group), CHUNK):
            part = group[c:c + CHUNK]
            recs = sim.simulate_batch(
                [j[1] for j in part], [j[2] for j in part],
                [int(v) for v in rng.integers(0, 2 ** 63, len(part))], phone, rng,
                postures=[j[4] for j in part], accessories=[j[5] for j in part],
                sessions=[j[6] for j in part], conditions=[j[3] for j in part],
                grips=[j[7] for j in part], purposes=[j[8] for j in part],
                seeds=[int(v) for v in rng.integers(0, 2 ** 62, len(part))])
            for r, j in zip(recs, part):
                r["_meta"] = j[9]
                out.append(r)
    return out
