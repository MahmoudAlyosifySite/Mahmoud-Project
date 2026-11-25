"""Simulated study: who records what, when (the proposal's evaluation plan on synthetic hands).

20 enrolled people + 5 strangers, one phone. Session 1 calibrates and enrolls (15 patterns
per hand: sitting, standing, fresh-grip rounds of 5), session 5 is an optional second short
sitting (5 per hand), sessions 2-4 are test days with 30 checks per person (alternating
hands, sitting/standing, fresh grip per check). Each check has a stream of up to 12
recordings (3 slots x 4 attempts) from one grip. Accessory quick add in session 2, accessory
checks in sessions 3-4; mimicry (strangers copying the victim's grip angles) in session 3;
replay checks (recorded responses fed to fresh challenges) in sessions 3-4.

Simulation and processing run together in worker processes; only compact items (quality
report + features) travel back.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from multiprocessing import get_context

import numpy as np

from sim import make_phone, make_population, session_clock, simulate_batch
from sim.params import Person

from .processing import build_calibration, item_from, replay

ACCESSORY_CYCLE = ("watch", "ring", "bracelet")


@dataclass
class StudyConfig:
    mode: str = "none"
    seed: int = 2026
    phone_seed: int = 7
    n_people: int = 20
    n_strangers: int = 5
    enroll_session: int = 1
    second_sitting_session: int = 5
    tune_session: int = 2
    test_sessions: tuple = (2, 3, 4)
    checks_per_session: int = 30
    attempts_per_check: int = 12
    patterns_per_round: int = 5
    spare_per_round: int = 2
    calibration_patterns: int = 20
    accessory_add_session: int = 2
    accessory_test_sessions: tuple = (3, 4)
    accessory_checks_per_hand: int = 5
    mimic_session: int = 3
    mimic_tries: int = 20
    replay_checks_per_person: int = 10
    replay_sessions: tuple = (3, 4)
    nogyro: bool = True
    day_scale: float = 1.0
    extra_features: tuple = ()         # extra feature sets per item (evaluation.processing.EXTRA_FEATURES)
    workers: int = 16
    batch: int = 256


def people_of(cfg: StudyConfig):
    enrolled = make_population(cfg.n_people, seed=cfg.seed, prefix="P", day_scale=cfg.day_scale)
    strangers = make_population(cfg.n_strangers, seed=cfg.seed + 1, prefix="X", day_scale=cfg.day_scale)
    return enrolled, strangers


def _hand_posture(j: int):
    return ("right" if j % 2 == 0 else "left"), ("sitting" if (j // 2) % 2 == 0 else "standing")


def plan(cfg: StudyConfig) -> list[dict]:
    """All trials of the study (small dicts; no physics yet)."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([0x53545544, cfg.seed])))
    specs: list[dict] = []
    check_id = [0]

    def add(role, who, idx, hand, posture, session, grip, purpose, check=None, seq=0, accessory="none",
            victim=None, condition="handheld", phys_posture=None, rnd=None):
        specs.append(dict(role=role, who=who, idx=idx, hand=hand, posture=posture, session=session, grip=grip,
                          purpose=purpose, check=check, seq=seq, accessory=accessory, victim=victim,
                          condition=condition, round=rnd, challenge=int(rng.integers(0, 2 ** 63)),
                          seed=int(rng.integers(0, 2 ** 63)), replay_challenge=int(rng.integers(0, 2 ** 63))))

    def new_check():
        check_id[0] += 1
        return check_id[0]

    for k in range(cfg.calibration_patterns):
        add("calibration", None, None, "right", "unspecified", cfg.enroll_session, None, "calibration",
            seq=k, condition="onTable")
    n_round = cfg.patterns_per_round + cfg.spare_per_round
    rounds = (("sitting", 101), ("standing", 102), ("freshGrip", 103))
    for i in range(cfg.n_people):
        for hand in ("right", "left"):
            for r_i, (posture, grip) in enumerate(rounds):
                for k in range(n_round):
                    add("enroll", "P", i, hand, posture, cfg.enroll_session, grip, "enrollment", seq=k, rnd=r_i)
            for k in range(n_round):
                add("enroll2", "P", i, hand, "sitting", cfg.second_sitting_session, 201, "enrollment", seq=k, rnd=0)
            acc = ACCESSORY_CYCLE[i % 3]
            for k in range(n_round):
                add("accessoryAdd", "P", i, hand, "sitting", cfg.accessory_add_session, 301, "accessory", seq=k,
                    accessory=acc, rnd=0)
    for who, n in (("P", cfg.n_people), ("X", cfg.n_strangers)):
        for i in range(n):
            for s in cfg.test_sessions:
                for j in range(cfg.checks_per_session):
                    hand, posture = _hand_posture(j)
                    c = new_check()
                    for a in range(cfg.attempts_per_check):
                        add("test", who, i, hand, posture, s, 1000 + j, "test", check=c, seq=a)
    for i in range(cfg.n_people):
        acc = ACCESSORY_CYCLE[i % 3]
        for s in cfg.accessory_test_sessions:
            for j in range(2 * cfg.accessory_checks_per_hand):
                hand, posture = _hand_posture(j)
                c = new_check()
                for a in range(cfg.attempts_per_check):
                    add("accessoryTest", "P", i, hand, posture, s, 2000 + j, "test", check=c, seq=a, accessory=acc)
    for x in range(cfg.n_strangers):
        for v in range(cfg.n_people):
            for j in range(cfg.mimic_tries):
                hand, posture = _hand_posture(j)
                c = new_check()
                for a in range(cfg.attempts_per_check):
                    add("mimic", "X", x, hand, posture, cfg.mimic_session, 3000 + v * 100 + j, "test", check=c,
                        seq=a, victim=v)
    for v in range(cfg.n_people):
        for j in range(cfg.replay_checks_per_person):
            hand, posture = _hand_posture(j)
            s = cfg.replay_sessions[j % len(cfg.replay_sessions)]
            c = new_check()
            for a in range(cfg.attempts_per_check):
                add("replay", "P", v, hand, posture, s, None, "test", check=c, seq=a)   # each source: its own grip
    return specs


_CACHE = {}


def _people(cfg_d):
    key = (cfg_d["seed"], cfg_d["n_people"], cfg_d["n_strangers"], cfg_d["day_scale"])
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = people_of(StudyConfig(**{k: cfg_d[k] for k in ("seed", "n_people", "n_strangers", "day_scale")}))
    return _CACHE[key]


def _start_time(phone, spec):
    up, _ = session_clock(phone, spec["session"])
    who_off = (0 if spec["who"] is None else (spec["idx"] + (100 if spec["who"] == "X" else 0))) * 900.0
    chk = (spec["check"] or 0) % 1000
    return up + 60.0 + who_off + chk * 14.0 + spec["seq"] * 2.3 + (spec["round"] or 0) * 40.0


def _simulate(cfg_d, specs):
    enrolled, strangers = _people(cfg_d)
    phone = make_phone(cfg_d["phone_seed"], cfg_d["mode"])
    persons, victims = [], []
    for s in specs:
        persons.append(None if s["who"] is None else (enrolled if s["who"] == "P" else strangers)[s["idx"]])
        victims.append(None if s["victim"] is None else enrolled[s["victim"]])
    recs = simulate_batch(persons, [s["hand"] for s in specs], [s["challenge"] for s in specs], phone, None,
                          postures=[s["posture"] for s in specs], accessories=[s["accessory"] for s in specs],
                          sessions=[s["session"] for s in specs], conditions=[s["condition"] for s in specs],
                          grips=[s["grip"] for s in specs], purposes=[s["purpose"] for s in specs],
                          angles_from=victims, seeds=[s["seed"] for s in specs],
                          start_times=[_start_time(phone, s) for s in specs])
    for r, s in zip(recs, specs):
        r["sim"]["study"] = {k: s[k] for k in ("role", "check", "seq", "round", "victim", "who", "idx")}
    return recs


def _work(args):
    cfg_d, specs, calibration = args
    from auditor_ref.challenge import generate
    recs = _simulate(cfg_d, specs)
    out = []
    for r, s in zip(recs, specs):
        if s["role"] == "replay":
            r = replay(r, generate(s["replay_challenge"]), r["hapticStart"] + 5.0)
            r["sim"] = {"study": {k: s[k] for k in ("role", "check", "seq", "round", "victim", "who", "idx")}}
        nog = cfg_d["nogyro"] and s["role"] in ("enroll", "test")
        it, _, _ = item_from(r, calibration, None, nogyro=nog, extra=tuple(cfg_d.get("extra_features") or ()))
        it["study"] = r["sim"]["study"]
        it["victim"] = s["victim"]
        out.append(it)
    return out


def run(cfg: StudyConfig, log=print) -> dict:
    """Simulate and process the whole study. Returns dict(calibration, items, config)."""
    t0 = time.time()
    specs = plan(cfg)
    cfg_d = asdict(cfg)
    cal_specs = [s for s in specs if s["role"] == "calibration"]
    rest = [s for s in specs if s["role"] != "calibration"]
    cal_recs = _simulate(cfg_d, cal_specs)
    calibration, cal_items = build_calibration(cal_recs, device=f"sim-phone-{cfg.phone_seed}",
                                               extra=tuple(cfg.extra_features))
    log(f"[{cfg.mode}] {len(specs)} trials; calibration {'ok' if calibration else 'FAILED'} "
        f"({sum(i['passed'] for i in cal_items)}/{len(cal_items)} passed)")
    # group by (role, check) locality so batches share durations; chunk for the pool
    chunks = [rest[i:i + cfg.batch] for i in range(0, len(rest), cfg.batch)]
    items = list(cal_items)
    for it in cal_items:
        it["study"] = {"role": "calibration"}
        it["victim"] = None
    workers = max(1, min(cfg.workers, len(chunks), os.cpu_count() or 1))
    ctx = get_context("fork")
    with ctx.Pool(workers) as pool:
        for k, res in enumerate(pool.imap(_work, [(cfg_d, c, calibration) for c in chunks], chunksize=1)):
            items.extend(res)
            if (k + 1) % max(1, len(chunks) // 10) == 0:
                log(f"[{cfg.mode}] {k + 1}/{len(chunks)} chunks, {time.time() - t0:.0f}s")
    log(f"[{cfg.mode}] simulated+processed {len(items)} recordings in {time.time() - t0:.0f}s")
    return {"calibration": calibration, "items": items, "config": cfg_d}
