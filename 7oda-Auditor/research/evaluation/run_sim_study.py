"""Simulated study for both IMU modes -> research/results/sim_study_<mode>.{json,md}.

    cd research && python -m evaluation.run_sim_study --mode both --workers 16

Each mode: simulate + process the study (evaluation.study), run the protocol and the
ablations (evaluation.protocol), then the day-to-day sensitivity runs. The JSON/Markdown
result files are rewritten after every part, so an interrupted run leaves the parts that
finished. Intermediate items are cached under ~/7oda-work/sim (or --cache) so analyses can be
re-run without re-simulating; the cache key is the study configuration without the
runtime-only fields (workers, batch), which do not change any simulated trial.

Memory: each worker holds one batch (about 0.25 MB per trial plus processing); 16 workers
with batch 256 stay well under 10 GB. Do not raise --workers on a shared machine.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import pickle
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from auditor_ref import features as _features
from auditor_ref import quality as _quality

from .protocol import dataset_from_study, full_evaluation, variant, checks_where
from .decision import Thresholds
from .report import render
from .study import StudyConfig, run

RESULTS = Path(__file__).resolve().parent.parent / "results"
RUNTIME_ONLY = ("workers", "batch")      # never change a simulated trial (each trial depends only on its seed)


def _key(cfg_d: dict) -> dict:
    return {k: v for k, v in cfg_d.items() if k not in RUNTIME_ONLY}


def clean(o):
    """JSON-safe copy: numpy scalars to Python, non-finite floats to None, tuples to lists."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write(rep: dict, mode: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"sim_study_{mode}.json").write_text(json.dumps(clean(rep), indent=1))
    (out_dir / f"sim_study_{mode}.md").write_text(render(rep, f"Simulated study, IMU mode `{mode}`"))


def pipeline_stamp() -> dict:
    """What the cached items depend on besides the study configuration (feature version, quality thresholds)."""
    return {"featureVersion": _features.FEATURE_VERSION, "quality": asdict(_quality.DEFAULT_CONFIG)}


def load_or_run(cfg: StudyConfig, cache: Path, tag: str, log):
    path = cache / f"items_{tag}.pkl.gz"
    if path.exists():
        with gzip.open(path, "rb") as f:
            data = pickle.load(f)
        if _key(data.get("config") or {}) == _key(asdict(cfg)) and data.get("pipeline") == pipeline_stamp():
            log(f"[{tag}] loaded cached items ({len(data['items'])})")
            return data
    data = run(cfg, log)
    cache.mkdir(parents=True, exist_ok=True)
    slim = [{k: v for k, v in it.items() if k not in ("createdAt",)} for it in data["items"]]
    with gzip.open(path, "wb", compresslevel=3) as f:
        pickle.dump({**data, "items": slim, "pipeline": pipeline_stamp()}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return data


def study_mode(mode: str, args, log):
    t0 = time.time()
    cfg = StudyConfig(mode=mode, seed=args.seed, phone_seed=args.phone_seed, workers=args.workers)
    if args.small:
        cfg = replace(cfg, n_people=6, n_strangers=3, checks_per_session=8, mimic_tries=2, replay_checks_per_person=2,
                      accessory_checks_per_hand=2, batch=64)
    data = load_or_run(cfg, Path(args.cache), f"{mode}{'_small' if args.small else ''}", log)
    ds = dataset_from_study(data["items"], data["calibration"])
    meta = {"config": asdict(cfg),
            "calibration": {k: data["calibration"][k] for k in ("patternsUsed", "quietGyroRMS", "snr", "ncc")}
            if data["calibration"] else None}

    def part(r):
        write({**r, **meta, "timing_s": {"study": round(time.time() - t0, 1)}}, mode, Path(args.out))

    rep = full_evaluation(ds, cfg.tune_session, [s for s in cfg.test_sessions if s != cfg.tune_session], log=log,
                          on_part=part)
    rep.update(meta)
    rep["timing_s"] = {"study": round(time.time() - t0, 1)}
    write(rep, mode, Path(args.out))
    log(f"[{mode}] wrote results ({time.time() - t0:.0f}s)")
    if args.sensitivity:
        rep["sensitivity"] = {"1": {"test": rep["main"]["test"], "tuned": rep["main"]["tuned"],
                                    "closed_set": rep["main"]["closed_set"]}}
        for scale in args.sensitivity:
            c2 = replace(cfg, day_scale=scale, mimic_tries=0, replay_checks_per_person=0,
                         accessory_checks_per_hand=0, nogyro=False)
            d2 = load_or_run(c2, Path(args.cache), f"{mode}_day{scale}{'_small' if args.small else ''}", log)
            ds2 = dataset_from_study(d2["items"], d2["calibration"])
            tune_ids = checks_where(ds2, "test", [cfg.tune_session])
            test_ids = checks_where(ds2, "test", [s for s in cfg.test_sessions if s != cfg.tune_session])
            v, _, _, _ = variant(ds2, f"day_scale {scale}", tune_ids, test_ids, Thresholds())
            rep["sensitivity"][f"{scale:g}"] = {"test": v["test"], "tuned": v["tuned"], "default": v["default"],
                                                "closed_set": v["closed_set"]}
            rep["timing_s"][f"day{scale:g}"] = round(time.time() - t0, 1)
            write(rep, mode, Path(args.out))
            log(f"[{mode}] sensitivity day_scale {scale}: top1 {v['test']['top1_name_acc']:.3f}, closed-set "
                f"{v['closed_set']['top1_mean_patterns']:.3f}")
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="both", choices=("none", "lpf", "both"))
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--phone-seed", type=int, default=7)
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--cache", default=str(Path.home() / "7oda-work" / "sim"))
    ap.add_argument("--sensitivity", type=float, nargs="*", default=[0.0, 0.5])
    ap.add_argument("--small", action="store_true", help="tiny configuration (smoke test)")
    args = ap.parse_args(argv)
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    for mode in (("none", "lpf") if args.mode == "both" else (args.mode,)):
        study_mode(mode, args, log)


if __name__ == "__main__":
    main()
