"""Small A/B studies of the simulator's variability settings (evidence for docs/SIMULATION.md section 4).

    cd research && python -m evaluation.sim_variants --mode none --variants base day0 pos_alloc geometry

Each variant runs a reduced study (20 enrolled + 5 strangers, enrollment in session 1, 10 checks of
4 attempts per person in sessions 2 and 3; thresholds tuned on session 2, tested on session 3) and
prints the quality pass rate, forced-choice top-1 (same day and cross-session), the optimistic bound
of the threshold rule and the tuned v0 result. About 10 s per variant with 12 workers.

Variants (applied in this process before the worker pool forks):
- base: the simulator as it is.
- day0: no day-to-day variability (day_scale 0).
- pos_alloc: same total within-person contact-position scatter, moved from the day level to the
  grip level (day 1.2 mm, grip 2.6 mm instead of 2.5 / 1.5 mm).
- geometry: extra between-person grip geometry: a habitual grip height (all contacts that can slide
  along the phone's long axis shift by N(0, 5 mm)) and finger spacing scaled by exp(N(0, 0.06)).
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np

import sim.params as P

from .decision import Thresholds
from .protocol import Scorer, checks_where, closed_set, dataset_from_study, enrollment_samples, separability, variant
from .study import StudyConfig, run


def _geometry():
    orig = P.make_person

    def make_person(seed, pid=None, name=None, dims=(0.0715, 0.147, 0.0078)):
        p = orig(seed, pid, name, dims)
        r = P._rng(0x47454F4D, seed)
        s, dy = float(np.exp(r.normal(0, 0.06))), float(r.normal(0, 0.005))
        fingers = list(P._FINGERS)
        free_y = [i for i, c in enumerate(P._CONTACTS) if c[3] != 1]
        hands = {}
        for side, h in p.hands.items():
            pos = h.pos.copy()
            fy = pos[fingers, 1]
            pos[fingers, 1] = fy.mean() + (fy - fy.mean()) * s
            pos[free_y, 1] += dy
            hands[side] = replace(h, pos=pos)
        return replace(p, hands=hands)

    P.make_person = make_person


def apply(name: str):
    if name == "pos_alloc":
        P.SESSION_JITTER["pos"], P.GRIP_JITTER["pos"] = 0.0012, 0.0026
    elif name == "geometry":
        _geometry()
    elif name not in ("base", "day0"):
        raise ValueError(f"unknown variant {name!r}")


def evaluate(name: str, mode: str, workers: int = 12) -> dict:
    cfg = StudyConfig(mode=mode, checks_per_session=10, attempts_per_check=4, test_sessions=(2, 3), mimic_tries=0,
                      replay_checks_per_person=0, accessory_checks_per_hand=0, nogyro=False, workers=workers,
                      batch=128, day_scale=0.0 if name == "day0" else 1.0)
    data = run(cfg, log=lambda s: None)
    ds = dataset_from_study(data["items"], data["calibration"])
    sc = Scorer(ds, enrollment_samples(ds))
    diag = separability(ds, sc, (3,))
    v = variant(ds, name, checks_where(ds, "test", [2]), checks_where(ds, "test", [3]), Thresholds())[0]
    return {"variant": name, "mode": mode, "pass": float(ds.passed.mean()),
            "same_day": diag.get("same_day_top1_first_pattern"),
            "closed_s2": closed_set(sc, checks_where(ds, "test", [2]))["top1_mean_patterns"],
            "closed_s3": closed_set(sc, checks_where(ds, "test", [3]))["top1_mean_patterns"],
            "oracle": diag.get("oracle_top1_at_stranger_accept", {}),
            "eig": diag.get("discriminant_eigenvalues", [])[:4],
            "tuned_top1": v["test"]["top1_name_acc"], "tuned_stranger": v["test"]["stranger_accept"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="none", choices=("none", "lpf"))
    ap.add_argument("--variants", nargs="+", default=["base", "day0", "pos_alloc", "geometry"])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args(argv)
    if len(args.variants) > 1:   # variants patch module state: run each in a fresh process
        import subprocess
        import sys
        for v in args.variants:
            subprocess.run([sys.executable, "-m", "evaluation.sim_variants", "--mode", args.mode, "--variants", v,
                            "--workers", str(args.workers)], check=True)
        return
    name = args.variants[0]
    apply(name)
    t0 = time.time()
    r = evaluate(name, args.mode, args.workers)
    ob = " ".join(f"@{k} {v:.3f}" for k, v in r["oracle"].items())
    print(f"{name:10s} {args.mode:4s} pass {r['pass']:.3f} | forced choice: same day {r['same_day']:.3f}, "
          f"S2 {r['closed_s2']:.3f}, S3 {r['closed_s3']:.3f} | bound {ob} | tuned v0 top-1 {r['tuned_top1']:.3f} "
          f"stranger {r['tuned_stranger']:.3f} | eig {', '.join(f'{e:.1f}' for e in r['eig'])} | "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
