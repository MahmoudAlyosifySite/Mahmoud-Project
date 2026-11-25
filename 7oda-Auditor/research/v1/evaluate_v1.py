"""Open-set identification on held-out simulated people: v1 teacher, v1
student and v0 on the same recordings with the same protocol.

Protocol (minimal and compatible with the proposal's evaluation plan; see
docs/V1_MODEL.md):

* the protocol set (``v1.simdata --layout protocol``) holds, per person,
  session 0 = enrollment (15 patterns per hand: sitting, standing, fresh
  grip) and sessions 1.. = checks of up to 3 patterns;
* people are split into groups of 30: 20 enroll, 5 are tuning strangers,
  5 are test strangers;
* thresholds are tuned once on the pooled tuning data (session 1 checks of
  the enrolled people + tuning strangers) to maximize name+hand accuracy
  with stranger accept <= 2%; the test data are sessions 2.. of the enrolled
  people + the test strangers;
* patterns that fail the quality gate (SPEC 6) are skipped, as the app
  retries them;
* v1: cosine prototypes per (person, hand, accessory) and the SPEC 14.3 rule;
  v0: ``auditor_ref`` features + Gaussian model (SPEC 9), the same
  sequential rule, its own thresholds tuned the same way (and the SPEC
  defaults reported too).

Usage::

    python -m v1.evaluate_v1 --protocol '/tmp/.../proto_ideal/*.npz' --tag ideal \
        --model teacher=/tmp/.../T1/best.pt --model student=/tmp/.../S1/best.pt --v0
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from . import data as D
from . import decision as DC
from . import trainlib as TL

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

V1_SLACKS = np.round(np.arange(-0.10, 0.61, 0.02), 3)
V1_NAME = np.round(np.arange(0.0, 0.301, 0.01), 3)
V1_HAND = np.round(np.arange(0.0, 0.301, 0.02), 3)
V0_FACTORS = np.round(np.concatenate([np.arange(0.5, 3.01, 0.1), np.arange(3.5, 10.01, 0.5)]), 3)
V0_NAME = np.array([0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40], dtype=float)
V0_HAND = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10, 15, 20], dtype=float)


# ---------------------------------------------------------------------------
# Protocol structure
# ---------------------------------------------------------------------------

def make_groups(ts: D.TensorSet, seed: int = 0, enrolled: int = 20, strangers: int = 5):
    people = np.arange(len(ts.person_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(people)
    size = enrolled + 2 * strangers
    groups = []
    for g in range(len(people) // size):
        chunk = people[g * size:(g + 1) * size]
        groups.append({"enrolled": chunk[:enrolled], "tune_strangers": chunk[enrolled:enrolled + strangers],
                       "test_strangers": chunk[enrolled + strangers:]})
    return groups


def session_rank(ts: D.TensorSet) -> np.ndarray:
    """Session index within each person (0 = enrollment), from session ids."""
    rank = np.zeros(len(ts), dtype=int)
    for p in np.unique(ts.person):
        idx = np.where(ts.person == p)[0]
        sess = sorted(set(ts.session[idx].tolist()), key=lambda s: str(ts.session_ids[s]))
        order = {s: i for i, s in enumerate(sess)}
        rank[idx] = [order[s] for s in ts.session[idx]]
    return rank


def build_checks(ts: D.TensorSet, people, sessions, quality_ok, max_patterns: int = 3, accessory: bool = False):
    """Checks of ``people`` in ``sessions``: arrays of pattern indices [n, M]
    (-1 = none), true person and true hand (one-handed checks only). Bare-hand
    checks by default; ``accessory=True`` selects the checks with an accessory."""
    check = ts.extra["check"]
    rank = ts.extra["_rank"]
    acc_sel = (ts.accessory != 0) if accessory else (ts.accessory == 0)
    sel = np.isin(ts.person, people) & np.isin(rank, sessions) & (check >= 0) & (ts.hand < 2) & acc_sel
    idx = np.where(sel)[0]
    keyed = {}
    for i in idx:
        keyed.setdefault((int(ts.person[i]), int(check[i])), []).append(i)
    pats, tp, th, n_pat = [], [], [], []
    for (p, c), lst in sorted(keyed.items()):
        lst = sorted(lst, key=lambda i: int(ts.extra["pattern"][i]))
        ok = [i for i in lst if quality_ok[i]][:max_patterns]
        if not ok:
            continue
        row = ok + [-1] * (max_patterns - len(ok))
        pats.append(row)
        tp.append(p)
        th.append(int(ts.hand[lst[0]]))
        n_pat.append(len(ok))
    return (np.array(pats, dtype=int).reshape(-1, max_patterns), np.array(tp, dtype=int),
            np.array(th, dtype=int), np.array(n_pat, dtype=int))


def enrollment_indices(ts: D.TensorSet, people, quality_ok):
    rank = ts.extra["_rank"]
    return np.where(np.isin(ts.person, people) & (rank == 0) & (ts.extra["check"] < 0)
                    & (ts.hand < 2) & quality_ok)[0]


def proto_keys(ts: D.TensorSet, idx):
    return [(ts.person_ids[ts.person[i]], D.HANDS[ts.hand[i]], D.ACCESSORIES[ts.accessory[i]]) for i in idx]


# ---------------------------------------------------------------------------
# Scorers: per group, return SequentialStats + per-check absolute base
# ---------------------------------------------------------------------------

def pooled_hand_model(emb_e: np.ndarray, hand_e: np.ndarray, shrink: float = 0.1):
    """Enrollment-time right-vs-left linear model in embedding space (the analog of the
    pooled Gaussian hand model of SPEC 9.3): LDA direction with a shrunk pooled
    covariance. Returns (w, b) with ``emb @ w + b > 0`` meaning right."""
    er, el = emb_e[hand_e == 0], emb_e[hand_e == 1]
    if len(er) < 2 or len(el) < 2:
        return None
    mu_r, mu_l = er.mean(0), el.mean(0)
    resid = np.concatenate([er - mu_r, el - mu_l])
    cov = resid.T @ resid / len(resid)
    cov = (1 - shrink) * cov + shrink * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    w = np.linalg.solve(cov, mu_r - mu_l)
    return w, -0.5 * float(w @ (mu_r + mu_l))


def v1_group_stats(emb, hand_logits, ts, group, quality_ok, which: str, loo_quantile: float = 0.95,
                   accessory: bool = False, hand_rule: str = "head"):
    """``hand_rule``: ``head`` = the network's hand head (SPEC 14.3), ``pooled`` = a linear
    right/left model fitted on the group's enrollment embeddings, ``none`` = prototype
    hand margin only."""
    e_idx = enrollment_indices(ts, group["enrolled"], quality_ok)
    protos = DC.Prototypes.fit(emb[e_idx], proto_keys(ts, e_idx))
    pid_global = np.array([ts.person_ids.index(p) for p in protos.persons])
    sessions, strangers = ((1,), group["tune_strangers"]) if which == "tune" else (
        tuple(range(2, 99)), group["test_strangers"])
    pe, tpe, the, npe = build_checks(ts, group["enrolled"], sessions, quality_ok, accessory=accessory)
    ps, tps, ths, nps = build_checks(ts, strangers, sessions, quality_ok)
    pats = np.concatenate([pe, ps])
    n_pat = np.concatenate([npe, nps])
    safe = np.where(pats < 0, 0, pats)
    s = np.einsum("nmd,kd->nmk", emb[safe], protos.matrix)
    if hand_rule == "head":
        hs = hand_logits[safe][:, :, 0] - hand_logits[safe][:, :, 1]
    elif hand_rule == "pooled":
        wb = pooled_hand_model(emb[e_idx], ts.hand[e_idx])
        hs = None if wb is None else emb[safe] @ wb[0] + wb[1]
    else:
        hs = None
    st = DC.SequentialStats.build(s, hs, protos.person, protos.hand, n_pat, pid_global)
    base = np.full(len(pats), protos.abs_threshold(loo_quantile, 0.0))
    true_p = np.concatenate([tpe, np.full(len(ps), -1)])
    true_h = np.concatenate([the, ths])
    stranger = np.concatenate([np.zeros(len(pe), bool), np.ones(len(ps), bool)])
    return st, base, true_p, true_h, stranger


def v0_group_stats(v0x, ts, group, quality_ok, which: str, accessory: bool = False):
    from auditor_ref.identifier import Identifier

    e_idx = enrollment_indices(ts, group["enrolled"], quality_ok)
    samples = [{"prototype": k, "features": v0x[i].tolist(), "source": "enrollment"}
               for i, k in zip(e_idx, proto_keys(ts, e_idx))]
    ident = Identifier(samples)
    protos = ident.prototypes
    persons = sorted({p[0] for p in protos})
    pidx = {p: i for i, p in enumerate(persons)}
    proto_person = np.array([pidx[p[0]] for p in protos])
    proto_hand = np.array([DC.HAND_ORDER[p[1]] for p in protos])
    pid_global = np.array([ts.person_ids.index(p) for p in persons])
    sessions, strangers = ((1,), group["tune_strangers"]) if which == "tune" else (
        tuple(range(2, 99)), group["test_strangers"])
    pe, tpe, the, npe = build_checks(ts, group["enrolled"], sessions, quality_ok, accessory=accessory)
    ps, tps, ths, nps = build_checks(ts, strangers, sessions, quality_ok)
    pats = np.concatenate([pe, ps])
    n_pat = np.concatenate([npe, nps])
    uniq = np.unique(pats[pats >= 0])
    dist = {int(i): np.asarray(ident.distances(v0x[i]), dtype=np.float64) for i in uniq}
    hlo = {int(i): float(ident.hand_log_odds(v0x[i])) for i in uniq}
    k = len(protos)
    s = np.zeros(pats.shape + (k,))
    hs = np.zeros(pats.shape)
    for (r, c), i in np.ndenumerate(pats):
        if i >= 0:
            s[r, c] = -dist[int(i)]
            hs[r, c] = hlo[int(i)]
    st = DC.SequentialStats.build(s, hs if ident.hand_model is not None else None, proto_person, proto_hand,
                                  n_pat, pid_global)
    loo = np.asarray(ident.model.loo, dtype=np.float64)
    q = float(np.quantile(loo, ident.model_config.loo_quantile)) if loo.size else 2.0 * v0x.shape[1]
    base = np.full(len(pats), -q)
    true_p = np.concatenate([tpe, np.full(len(ps), -1)])
    true_h = np.concatenate([the, ths])
    stranger = np.concatenate([np.zeros(len(pe), bool), np.ones(len(ps), bool)])
    return st, base, true_p, true_h, stranger


_HAS_ACC = {"value": False}


def ts_has_accessory_checks(groups) -> bool:
    return _HAS_ACC["value"]


def pooled(fn, groups):
    parts = [fn(g) for g in groups]
    st = DC.SequentialStats.concat(p[0] for p in parts)
    gid = np.concatenate([np.full(len(p[1]), i) for i, p in enumerate(parts)])
    return (st, np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts]),
            np.concatenate([p[3] for p in parts]), np.concatenate([p[4] for p in parts]), gid)


EXTRA_TARGETS = (0.05, 0.10)


def evaluate_scorer(name, fn, groups, kind: str, max_stranger_accept: float = 0.02,
                    extra_targets=EXTRA_TARGETS):
    tune = pooled(lambda g: fn(g, "tune"), groups)
    test = pooled(lambda g: fn(g, "test"), groups)
    targets = [max_stranger_accept] + [t for t in extra_targets if t != max_stranger_accept]
    if kind == "v1":
        choice = DC.tune_thresholds(tune[0], tune[1], tune[2], tune[3], tune[4], V1_SLACKS, V1_NAME, V1_HAND,
                                    targets, abs_mode="subtract")
    else:
        choice = DC.tune_thresholds(tune[0], tune[1], tune[2], tune[3], tune[4], V0_FACTORS, V0_NAME, V0_HAND,
                                    targets, abs_mode="scale")
    tuned = choice[max_stranger_accept]
    thr = test[1] - tuned["slack"] if kind == "v1" else test[1] * tuned["slack"]
    rows = []
    out, pp, ph, used = DC.decide(test[0], thr, tuned["name_margin"], tuned["hand_margin"])
    met = DC.check_metrics(out, pp, ph, used, test[2], test[3], test[4])
    per_group = []
    for g in np.unique(test[5]):
        m = test[5] == g
        per_group.append(DC.check_metrics(out[m], pp[m], ph[m], used[m], test[2][m], test[3][m], test[4][m]))
    enr = ~test[4]
    cs = DC.closed_set_top1(DC.SequentialStats(*(getattr(test[0], f)[enr] for f in DC.SequentialStats.__dataclass_fields__)),
                            test[2][enr], test[3][enr], m=1)
    met.update(top1_std_groups=float(np.std([r["top1"] for r in per_group])),
               closed_top1_1pattern=cs[0], closed_name_hand_1pattern=cs[1])
    rows.append(dict(model=name, thresholds="tuned", slack_or_absFactor=tuned["slack"],
                     name_margin=tuned["name_margin"], hand_margin=tuned["hand_margin"],
                     tuning_met_target=tuned["met_target"],
                     tuning_stranger_accept=tuned["tuning_metrics"]["stranger_accept"],
                         tuning_name_and_hand=tuned["tuning_metrics"]["name_and_hand"], **met))
    for t in targets[1:]:
        tt = choice[t]
        thr_t = test[1] - tt["slack"] if kind == "v1" else test[1] * tt["slack"]
        out, pp, ph, used = DC.decide(test[0], thr_t, tt["name_margin"], tt["hand_margin"])
        met = DC.check_metrics(out, pp, ph, used, test[2], test[3], test[4])
        rows.append(dict(model=name, thresholds=f"tuned for stranger accept <= {t:.0%}",
                         slack_or_absFactor=tt["slack"], name_margin=tt["name_margin"],
                         hand_margin=tt["hand_margin"], tuning_met_target=tt["met_target"],
                         tuning_stranger_accept=tt["tuning_metrics"]["stranger_accept"],
                         tuning_name_and_hand=tt["tuning_metrics"]["name_and_hand"], **met))
    if ts_has_accessory_checks(groups):
        acc = pooled(lambda g: fn(g, "test", accessory=True), groups)
        thr_a = acc[1] - tuned["slack"] if kind == "v1" else acc[1] * tuned["slack"]
        out, pp, ph, used = DC.decide(acc[0], thr_a, tuned["name_margin"], tuned["hand_margin"])
        met = DC.check_metrics(out, pp, ph, used, acc[2], acc[3], acc[4])
        rows.append(dict(model=name, thresholds="tuned, accessory on (enrolled bare)",
                         slack_or_absFactor=tuned["slack"], name_margin=tuned["name_margin"],
                         hand_margin=tuned["hand_margin"], tuning_met_target=tuned["met_target"],
                         tuning_stranger_accept=tuned["tuning_metrics"]["stranger_accept"],
                         tuning_name_and_hand=tuned["tuning_metrics"]["name_and_hand"], **met))
    if kind == "v0":
        out, pp, ph, used = DC.decide(test[0], test[1] * 1.5, 6.0, 3.0)
        met = DC.check_metrics(out, pp, ph, used, test[2], test[3], test[4])
        rows.append(dict(model=name, thresholds="spec defaults", slack_or_absFactor=1.5, name_margin=6.0,
                         hand_margin=3.0, tuning_met_target="", tuning_stranger_accept="", tuning_name_and_hand="",
                         **met))
    return rows


def run(protocol: str, models: dict, use_v0: bool, tag: str, seed: int = 0, device: str = "cuda",
        out_csv: str | None = None, quality: bool = True, log=print):
    ts = TL.load_sets(protocol)
    ts.extra["_rank"] = session_rank(ts)
    q_ok = ts.extra.get("quality_ok", np.ones(len(ts), bool)) if quality else np.ones(len(ts), bool)
    groups = make_groups(ts, seed)
    _HAS_ACC["value"] = bool(np.any((ts.accessory != 0) & (ts.extra["check"] >= 0)))
    log(f"[{tag}] recordings {len(ts)}, people {len(ts.person_ids)}, groups {len(groups)}, "
        f"quality pass {float(np.mean(q_ok)):.3f}")
    rows = []
    for name, path in models.items():
        model, _ = TL.load_model(path, device)
        emb, hand = TL.embed(model, ts.x, device, bf16=False)
        for rule, label in (("head", name), ("pooled", f"{name} (pooled hand model)"),
                            ("none", f"{name} (no hand check)")):
            fn = lambda g, which, emb=emb, hand=hand, accessory=False, rule=rule: v1_group_stats(
                emb, hand, ts, g, q_ok, which, accessory=accessory, hand_rule=rule)
            rows += evaluate_scorer(label, fn, groups, "v1")
            main_row = [r for r in rows if r["model"] == label and r["thresholds"] == "tuned"][0]
            log(f"[{tag}] {label}: " + json.dumps({k: round(main_row[k], 4) for k in (
                "top1", "hand_acc", "name_and_hand", "stranger_accept", "false_unknown")}))
    if use_v0:
        v0x = ts.extra.get("v0")
        if v0x is None or not np.all(np.isfinite(v0x[q_ok])):
            log(f"[{tag}] v0 features missing in the protocol shards; skipping v0")
        else:
            fn = lambda g, which, accessory=False: v0_group_stats(v0x, ts, g, q_ok, which, accessory=accessory)
            rows += evaluate_scorer("v0", fn, groups, "v0")
            main_row = [r for r in rows if r["model"] == "v0" and r["thresholds"] == "tuned"][0]
            log(f"[{tag}] v0: " + json.dumps({k: main_row[k] for k in ("top1", "hand_acc", "name_and_hand",
                                                                     "stranger_accept", "false_unknown")}))
    for r in rows:
        r["imu_mode"] = tag
        r["groups"] = len(groups)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        keys = list(rows[0].keys())
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        log(f"wrote {out_csv}")
    return rows


def markdown_table(rows) -> str:
    cols = [("model", "Model"), ("thresholds", "Thresholds"), ("top1", "Top-1"), ("hand_acc", "Hand"),
            ("name_and_hand", "Name+hand"), ("stranger_accept", "Stranger accept"),
            ("false_unknown", "False Unknown"), ("patterns_per_answer", "Patterns/answer"),
            ("closed_top1_1pattern", "Closed-set top-1 (1 pattern)")]
    out = ["| " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for k, _ in cols:
            v = r.get(k, "")
            if isinstance(v, float):
                v = f"{v:.2f}" if k == "patterns_per_answer" else f"{100 * v:.1f}%"
            cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--model", action="append", default=[], help="name=checkpoint")
    ap.add_argument("--v0", action="store_true")
    ap.add_argument("--tag", default="sim")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-quality", action="store_true")
    args = ap.parse_args(argv)
    models = dict(m.split("=", 1) for m in args.model)
    out = args.out or os.path.join(RESULTS, f"v1_eval_{args.tag}.csv")
    rows = run(args.protocol, models, args.v0, args.tag, args.seed, out_csv=out, quality=not args.no_quality)
    print(markdown_table(rows))


if __name__ == "__main__":
    main()
