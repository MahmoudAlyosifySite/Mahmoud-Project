"""Simulated datasets for v1: parallel generation of TensorSet shards.

Backends turn a *plan* (people, hands, sessions, recordings per session)
into Recording dicts (SPEC 4); every recording then goes through the fast
tensor path (``data.tensor_from_recording``). Backends:

* ``sim``: the physics simulator in ``research/sim`` (see ``sim_backend``);
* ``standin``: ``v1.standin``, for smoke tests only.

Two layouts:

* ``train``: many people, ``sessions`` x ``per_hand`` recordings per hand,
  postures and accessories drawn at random (nuisance);
* ``protocol``: the evaluation layout of the proposal: session 0 enrolls
  15 patterns per hand (3 rounds of 5: sitting, standing, fresh grip), later
  sessions hold checks of up to 3 patterns each. Shards also keep the raw
  recordings (gzipped JSON Lines) so v0 can be computed on the same data.

Usage::

    python -m v1.simdata --backend sim --layout train --out DIR --people 20000 \
        --per-hand 12 --sessions 4 --workers 120 --seed 1
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from multiprocessing import get_context

import numpy as np

from . import data as D

POSTURE_ROUNDS = ("sitting", "standing", "freshGrip")


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------

def train_plan(per_hand: int, sessions: int, rng: np.random.Generator, accessory_prob: float = 0.15,
               two_handed_per_session: int = 0):
    """List of (session, hand, posture, accessory, check, pattern) for one person."""
    items = []
    for s in range(sessions):
        for hand in ("right", "left"):
            for k in range(per_hand):
                acc = "none"
                if rng.random() < accessory_prob:
                    acc = str(rng.choice(["watch", "ring", "bracelet"]))
                items.append((s, hand, POSTURE_ROUNDS[k % 3], acc, -1, 0))
        for k in range(two_handed_per_session):
            items.append((s, "twoHanded", POSTURE_ROUNDS[k % 3], "none", -1, 0))
    return items


def protocol_plan(sessions: int, enroll_per_hand: int, checks_per_hand: int, patterns: int,
                  accessory_checks: int = 0):
    """Session 0 enrollment, later sessions checks (check ids are per person)."""
    items = []
    for hand in ("right", "left"):
        for k in range(enroll_per_hand):
            items.append((0, hand, POSTURE_ROUNDS[(k // 5) % 3], "none", -1, 0))
    c = 0
    for s in range(1, sessions):
        for hand in ("right", "left"):
            for k in range(checks_per_hand):
                for p in range(patterns):
                    items.append((s, hand, POSTURE_ROUNDS[k % 3], "none", c, p))
                c += 1
            for k in range(accessory_checks):
                for p in range(patterns):
                    items.append((s, hand, POSTURE_ROUNDS[k % 3], "watch", c, p))
                c += 1
    return items


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def standin_backend(person_ids, plans, seed: int, options: dict):
    from . import standin

    rng = np.random.default_rng(seed)
    out = []
    for pid, plan in zip(person_ids, plans):
        person = standin.person_params(rng)
        hands = {h: standin.hand_params(person, h, rng) for h in ("right", "left")}
        hands["twoHanded"] = dict(hands["right"], tilt=0.0)
        drift = {}
        for (s, hand, posture, acc, check, pattern) in plan:
            if (s, hand) not in drift:
                hp = hands[hand]
                drift[(s, hand)] = dict(hp, f0=hp["f0"] * rng.uniform(0.97, 1.03))
            label = {"personID": pid, "personName": pid, "hand": hand if hand != "twoHanded" else "right",
                     "accessory": acc, "posture": posture, "sessionID": f"{pid}-s{s}"}
            rec = standin.simulate_recording(drift[(s, hand)], int(rng.integers(0, 2 ** 63)), rng, label=label)
            rec["purpose"] = "enrollment" if check < 0 else "check"
            rec["_meta"] = {"hand": hand, "check": check, "pattern": pattern, "session": s}
            out.append(rec)
    return out


def sim_backend(person_ids, plans, seed: int, options: dict):
    """Adapter to ``research/sim`` (filled in once its API is published)."""
    from . import sim_adapter

    return sim_adapter.generate(person_ids, plans, seed, options)


BACKENDS = {"standin": standin_backend, "sim": sim_backend}


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def _v0_features(rec):
    """v0 features and the quality verdict, or None when auditor_ref lacks them."""
    try:
        from auditor_ref import signal as S, features as F, quality as Q
    except Exception:
        return None
    p = S.process(rec)
    rep = Q.evaluate(p, None, "handheld")
    return np.asarray(F.extract_absolute(p), dtype=np.float64), bool(rep["passed"])


def _work(task):
    (shard_path, backend, person_ids, plans, seed, options, keep_raw) = task
    write_jsonl = bool(options.get("raw_jsonl", False))
    t0 = time.time()
    recs = BACKENDS[backend](person_ids, plans, seed, options)
    xs, person, hand, acc, post, sess, nccs, check, pattern, purpose = ([] for _ in range(10))
    v0, qpass = [], []
    pid_index, sid_index = {}, {}
    raw_lines = []
    for rec in recs:
        meta = rec.pop("_meta", {})
        x, fp = D.tensor_from_recording(rec, return_processed=True)
        lab = D.label_fields(rec)
        xs.append(x.astype(np.float16))
        person.append(pid_index.setdefault(lab["personID"], len(pid_index)))
        hand.append(D.HAND_INDEX[meta.get("hand", lab["hand"])])
        acc.append(D.ACCESSORY_INDEX.get(lab["accessory"], 4))
        post.append(D.POSTURE_INDEX.get(lab["posture"], 3))
        sess.append(sid_index.setdefault(lab["sessionID"], len(sid_index)))
        nccs.append(fp.ncc)
        check.append(meta.get("check", -1))
        pattern.append(meta.get("pattern", 0))
        purpose.append(0 if rec.get("purpose") == "enrollment" else 1)
        if keep_raw:
            f = _v0_features(rec)
            v0.append(f[0] if f else np.full(36, np.nan))
            qpass.append(f[1] if f else True)
            if write_jsonl:
                raw_lines.append(json.dumps(dict(rec, _meta=meta)))
    extra = {"check": np.array(check, np.int32), "pattern": np.array(pattern, np.int8),
             "purpose": np.array(purpose, np.int8)}
    if keep_raw:
        extra["v0"] = np.stack(v0)
        extra["quality_ok"] = np.array(qpass, bool)
    ts = D.TensorSet(np.stack(xs), np.array(person, np.int32), np.array(hand, np.int8),
                     np.array(acc, np.int8), np.array(post, np.int8), np.array(sess, np.int32),
                     np.array(nccs, np.float32), list(pid_index), list(sid_index), extra)
    ts.save(shard_path)
    if keep_raw and write_jsonl:
        with gzip.open(shard_path.replace(".npz", ".jsonl.gz"), "wt") as fh:
            fh.write("\n".join(raw_lines))
    return shard_path, len(recs), time.time() - t0


def generate(out_dir: str, backend: str, layout: str, people: int, seed: int, workers: int,
             people_per_shard: int, prefix: str, options: dict, per_hand: int = 12, sessions: int = 4,
             enroll_per_hand: int = 15, checks_per_hand: int = 10, patterns: int = 3,
             accessory_checks: int = 0, accessory_prob: float = 0.15, two_handed: int = 0,
             keep_raw: bool | None = None, log=print):
    os.makedirs(out_dir, exist_ok=True)
    keep_raw = (layout == "protocol") if keep_raw is None else keep_raw
    tasks = []
    for si, start in enumerate(range(0, people, people_per_shard)):
        ids = [f"{prefix}{i:06d}" for i in range(start, min(people, start + people_per_shard))]
        rng = np.random.default_rng([seed, si])
        if layout == "train":
            plans = [train_plan(per_hand, sessions, rng, accessory_prob, two_handed) for _ in ids]
        else:
            plans = [protocol_plan(sessions, enroll_per_hand, checks_per_hand, patterns, accessory_checks)
                     for _ in ids]
        path = os.path.join(out_dir, f"shard_{si:05d}.npz")
        if os.path.exists(path):
            continue
        tasks.append((path, backend, ids, plans, int(rng.integers(0, 2 ** 62)), options, keep_raw))
    t0 = time.time()
    n_rec = 0
    if workers <= 1:
        results = map(_work, tasks)
    else:
        ctx = get_context("spawn")
        pool = ctx.Pool(workers)
        results = pool.imap_unordered(_work, tasks)
    for i, (path, n, dt) in enumerate(results):
        n_rec += n
        if i % max(1, len(tasks) // 20) == 0 or i == len(tasks) - 1:
            log(f"[{time.time() - t0:7.1f}s] shard {i + 1}/{len(tasks)} ({n} rec, {dt:.1f}s) total {n_rec}")
    if workers > 1:
        pool.close()
        pool.join()
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump({"backend": backend, "layout": layout, "people": people, "seed": seed, "prefix": prefix,
                   "options": options, "per_hand": per_hand, "sessions": sessions,
                   "enroll_per_hand": enroll_per_hand, "checks_per_hand": checks_per_hand,
                   "patterns": patterns, "accessory_checks": accessory_checks,
                   "accessory_prob": accessory_prob, "two_handed": two_handed}, fh, indent=1)
    return n_rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="sim", choices=sorted(BACKENDS))
    ap.add_argument("--layout", default="train", choices=["train", "protocol"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--people", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--people-per-shard", type=int, default=50)
    ap.add_argument("--prefix", default="P")
    ap.add_argument("--per-hand", type=int, default=12)
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--enroll-per-hand", type=int, default=15)
    ap.add_argument("--checks-per-hand", type=int, default=10)
    ap.add_argument("--patterns", type=int, default=3)
    ap.add_argument("--accessory-checks", type=int, default=0)
    ap.add_argument("--accessory-prob", type=float, default=0.15)
    ap.add_argument("--two-handed", type=int, default=0, help="two-handed recordings per session (train)")
    ap.add_argument("--option", action="append", default=[], help="backend option key=value (JSON value)")
    args = ap.parse_args(argv)
    opts = {}
    for kv in args.option:
        k, v = kv.split("=", 1)
        try:
            opts[k] = json.loads(v)
        except json.JSONDecodeError:
            opts[k] = v
    generate(args.out, args.backend, args.layout, args.people, args.seed, args.workers, args.people_per_shard,
             args.prefix, opts, per_hand=args.per_hand, sessions=args.sessions,
             enroll_per_hand=args.enroll_per_hand, checks_per_hand=args.checks_per_hand,
             patterns=args.patterns, accessory_checks=args.accessory_checks,
             accessory_prob=args.accessory_prob, two_handed=args.two_handed)


if __name__ == "__main__":
    main()
