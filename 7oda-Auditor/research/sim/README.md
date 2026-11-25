# sim — phone + hand vibration simulator

Generates synthetic `Recording` dicts in exactly the SPEC section 4 form (raw accelerometer with
gravity, raw gyro with bias, device motion with gravity / userAccel / rotation / attitude,
`hapticStart` on the uptime clock, ISO-8601 dates without fractional seconds, label filled in).
Pure numpy, batch-vectorised, deterministic given seeds. Model equations and parameter
sources: `docs/SIMULATION.md`.

Simulated accuracy numbers validate the pipeline's mechanics; they are not evidence about
real hands (DECISIONS D9).

## Quick start (run from `research/`)

```python
import numpy as np
from sim import make_population, make_phone, simulate, simulate_batch, simulate_random

people = make_population(20, seed=1)            # Person objects P01..P20, both hands each
phone = make_phone(seed=3, imu_mode="none")     # "none" = raw aliasing, "lpf" = 30-50 Hz low-pass before decimation
rng = np.random.Generator(np.random.PCG64(0))

rec = simulate(people[0], "right", 1234, phone, rng)         # challenge = seed (auditor_ref.challenge.generate) or a spec dict

recs = simulate_batch(                                       # vectorised over trials; scalars broadcast
    [people[i % 20] for i in range(512)], ["right", "left"] * 256,
    list(range(512)), phone, rng,
    postures="sitting", accessories="none", sessions=2, conditions="handheld", grips=None)

table = simulate_batch(None, "right", list(range(20)), phone, rng, conditions="onTable")  # calibration patterns

pretrain = simulate_random(10_000, seed=7, n_people=5000, imu_mode="none", p_two_handed=0.1)
```

## API

| Function | Meaning |
|---|---|
| `make_population(n_people, seed, prefix="P")` | `n_people` persons, ids `P01`, `P02`, ...; a person is a pure function of its seed |
| `make_person(seed, pid=None, name=None)` | one person |
| `make_phone(seed, imu_mode)` | one phone; physics depend only on `seed`, `imu_mode` picks the anti-alias stage and noise |
| `simulate(person, hand, challenge_spec, phone, rng, posture=, accessory=, session=, condition=, grip=, purpose=, angles_from=, start_time=, include_meta=, fidget=)` | one recording |
| `simulate_batch(persons, hands, challenges, phone, rng, postures=, accessories=, sessions=, conditions=, grips=, purposes=, angles_from=, seeds=, start_times=, include_meta=, sim_rate=2000, fidget=)` | B recordings; every argument is a scalar or a length-B sequence |
| `simulate_random(n, seed, phone=None, people=None, n_people=1000, imu_mode="none", p_standing=, p_accessory=, p_two_handed=, p_table=, n_sessions=4, batch=512)` | random labelled trials for pretraining |
| `session_clock(phone, session)` | (uptime base, UTC datetime) of a session ("day") |

Arguments:

- `hand`: `"right" | "left"` (the label hand).
- `challenge`: a ChallengeSpec dict (SPEC 3.4) or an int seed for `auditor_ref.challenge.generate`.
- `posture`: `sitting | standing | freshGrip | unspecified` (label value; `freshGrip` is physically sitting).
- `accessory`: `none | watch | ring | bracelet | other` (watch/bracelet/other add wrist mass coupled
  through the palm; ring stiffens and adds mass at one finger contact; each person owns their accessories).
- `session`: int "day". Grip state, tissue state, tremor level, gyro bias and the clock change per session.
- `grip`: int grip instance inside a session (the phone picked up again). Use the same int for all
  patterns of one check; `None` draws a fresh grip for every trial.
- `condition`: `handheld` (one hand), `twoHanded` (both hands' contacts, softer; label hand is the one
  passed; `rec["sim"]["grip"] == "twoHanded"`), `onTable` (stiff, lightly damped corner supports, gravity
  `(0, 0, -1)`, no tremor; label `null`, purpose defaults to `calibration`).
- `angles_from`: a Person whose typical grip angles are copied (mimicry proxy).
- `seeds`: per-trial seeds; a trial depends only on its inputs and its seed, never on the rest of
  the batch, so results do not depend on batch size or order.
- `purposes`: SPEC purpose string; default `check` (or `calibration` for onTable).

Extra key: with `include_meta=True` (default) each recording carries `rec["sim"]` with ground truth
(`condition`, `grip` = `right | left | twoHanded | table`, `session`, `gripInstance`, `personSeed`,
`phoneSeed`, `imuMode`, `latency`, `f0`, `gravityAngles`, `fidget`). Decoders ignore unknown keys; pass
`include_meta=False` for strictly-SPEC dicts.

## Parameters (summary; ranges in `params.py`)

Phone: actuator resonance `f0 ~ U(140, 200) Hz`, effective Q 2-4, peak force 1.5-3 N along x with
±0.12 cross-coupling, actuator near the bottom edge, IMU near the centre; tap = 2-4 cycle burst at
`f0 (0.7 + 0.6 sharpness)`, ramp = carrier with a 7-14 ms first-order envelope lag; haptic onset
latency 12-35 ms (trial jitter 2-4 ms, clipped to 10-40 ms); IMU 100 Hz ±0.3 %, timestamp jitter
0.1-0.4 ms, single drops 0.1-0.4 %, short gaps (3-8 samples) in 5 % of recordings, long gaps (30-60
samples) in 0.5 %, duplicated timestamps 0.2 %, fidget (hand movement) in 3 % of handheld recordings.
`imu_mode="lpf"`: 2nd-order Butterworth at 30-50 Hz before decimation (`lpf_order=4` via
`make_phone(..., lpf_order=4)` removes the carrier almost entirely and most recordings then fail
the SNR gate).

Hand (per person, per hand): 7 contacts (palm, thenar, three fingertips, little finger, thumb) with
normal/tangential stiffness, damping, coupled hand mass and a cubic stiffening term; the left hand is
the right hand mirrored in x plus its own random differences; grip angles per posture; physiological
tremor (8-12 Hz peak, person-specific amplitude and axes), slow sway (< 3 Hz), standing sways more.
Variability hierarchy: person/hand -> session (day) -> grip instance -> trial; the day and grip
levels are `SESSION_JITTER` and `GRIP_JITTER` in `params.py`. Ranges, their justification and the
A/B evidence for keeping them: `docs/SIMULATION.md` sections 2-4.

## Speed (one core, this server)

| Batch | `none` | `lpf` |
|---|---|---|
| 256 | ~240 rec/s | ~225 rec/s |
| 512 | ~250 rec/s | ~245 rec/s |

Integration runs at 2 kHz (exact zero-order-hold discretisation of the linear part, explicit cubic
term; agrees with RK4 at 8 kHz to 1-3.5 % RMS). Memory is about 0.25 MB per trial in a batch. For
millions of trials use a process pool with one seed per chunk, e.g.

```python
from multiprocessing import Pool
def chunk(i): return simulate_random(512, seed=1000 + i, people=PEOPLE, phone=PHONES[i % len(PHONES)])
with Pool(16) as pool: ...   # the container's memory limit is far below what `free` shows
```

`auditor_ref.signal.process` costs about 5 ms per recording, i.e. more than the simulation itself.

## Quality gates (SPEC section 6) on simulated data

Typical numbers over 4 phones x 64 handheld recordings: `none` 98 % pass (median SNR ~370),
`lpf` ~89 % pass (median SNR ~6, 10th percentile ~2; failures are mostly `weakSignal`); on-table
calibration patterns pass `onTable` 92-99 %. The full simulated study (phone seed 7) passes 97.8 % of
test recordings in `none` and 80.7 % in `lpf` (`research/results/sim_study_<mode>.md`).
