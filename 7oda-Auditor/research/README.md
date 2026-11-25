# research/ — reference pipeline, simulator, evaluation

| Folder | Contents |
|---|---|
| `auditor_ref/` | Python reference of the v0 pipeline (SPEC sections 2-9); writes the parity fixtures the Swift tests check |
| `sim/` | Phone + hand vibration simulator ([sim/README.md](sim/README.md), model and ranges in [docs/SIMULATION.md](../docs/SIMULATION.md)) |
| `evaluation/` | Evaluation protocol of the proposal (enrollment, tuned thresholds, sequential checks, metrics, ablations) and the simulated study |
| `results/` | Simulated study results, `sim_study_<mode>.{json,md}` |
| `v1/` | v1 encoder research (GPU) |
| `tests/` | `python -m pytest -q` from this folder |

```bash
cd research
pip install -r requirements.txt
python -m pytest -q
```

## Simulated study

**Simulated numbers validate the pipeline's mechanics (alignment, quality gate, features,
thresholds, sequential checks, Unknown). They are not evidence about real hands** (DECISIONS D9):
the simulator's hand model sets how much identity information reaches the sensors, and that is
exactly what only the pilot can measure.

```bash
cd research
nohup python -m evaluation.run_sim_study --mode both --workers 16 > ~/7oda-work/sim/study.log 2>&1 &
```

- Design (`evaluation/study.py`): 20 enrolled people + 5 strangers, one phone (seed 7), study seed 2026.
  Session 1: 20 on-table calibration patterns and enrollment (sitting, standing, fresh-grip rounds of
  5 passing patterns per hand). Sessions 2-4: 30 checks per person per session (alternating hands,
  sitting/standing, a fresh grip per check, up to 12 recordings per check). Thresholds are tuned on
  session 2 (stranger accept <= 2 %), metrics are reported on sessions 3-4. Also: a second short
  sitting (session 5), accessory quick add (session 2) and accessory checks (sessions 3-4), mimicry
  (strangers copying each victim's grip angles, session 3), replayed responses (sessions 3-4).
- Both IMU modes: `none` (raw point sampling, the Taptic carrier aliases) and `lpf` (2nd-order
  low-pass at 30-50 Hz before decimation). The real phone is somewhere between; the pilot decides.
- Outputs: `results/sim_study_none.{json,md}` and `results/sim_study_lpf.{json,md}`, rewritten after
  each part (main result, operating points and diagnostics; ablations; sensitivity), so an
  interrupted run keeps what finished.
- Cache: processed items go to `~/7oda-work/sim/items_<mode>[_day<scale>].pkl.gz`; a re-run with the
  same configuration only re-evaluates (about 2 minutes per mode). Delete the cache after changing
  the simulator or the reference pipeline.
- Cost on the development server with 16 workers: about 25 s to simulate and process 30 000
  recordings; a full run (both modes, two sensitivity runs each) takes about 12 minutes. Each worker
  holds one batch (about 0.25 MB per trial), so 16 workers stay well under 10 GB. The container's
  memory limit is lower than `free` shows: check `/sys/fs/cgroup/memory.max` before raising
  `--workers`.
- `--small` runs a 6-person smoke configuration in seconds.

What the result files contain: the headline at the tuned operating point and at the SPEC
defaults; forced-choice (closed-set) accuracy without the Unknown option, per session and against
a same-day reference; operating points for stranger targets 1-20 %; where the identity information
is (per-feature variance split into person, day, grip and trial parts, Fisher ratios, discriminant
dimensions, and an optimistic bound for the threshold rule); ablations (patterns per check, feature
groups, accelerometer only, one or two enrollment sittings, accessories, cross-hand, replay,
mimicry); and the day-to-day sensitivity runs (`day_scale` 0, 0.5, 1).

A/B studies of the simulator's variability settings (evidence in docs/SIMULATION.md section 4):

```bash
python -m evaluation.sim_variants --mode none --variants base day0 pos_alloc geometry
```
