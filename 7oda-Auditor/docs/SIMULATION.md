# Simulation

`research/sim/` generates recordings in the SPEC section 4 form from a physical model of a phone
held in a hand. It exists to test the pipeline's mechanics (alignment, quality gate, features,
thresholds, sequential checks, Unknown, attacks) before real data exist. **Simulated accuracy is
not evidence about real hands** (DECISIONS D9): how much identity information reaches the sensors
is set by the hand model below, and only the pilot can measure it. API: `research/sim/README.md`.

## 1. Model

**Actuator** (`actuator.py`). The actuator's moving mass follows a prescribed displacement and the
phone feels the reaction force `u = -xi''`, so every event has zero net impulse and zero net
displacement; any low-frequency motion the 100 Hz sensors see comes from the hand's response.
Tap: a 2.5-3.5 cycle `sin^2`-windowed burst at `f0 (0.7 + 0.6 sharpness)` plus a net excursion
of the moving mass (the "click"); ramp: the carrier with the intensity curve passed through a
first-order lag. Force amplitude = peak force x intensity x the actuator's resonance gain.

**Phone + hand** (`dynamics.py`). Rigid phone, 6 degrees of freedom about its resting pose:
`M q'' + C q' + K q + D^T f_nl(D q) = b u(t)`. Each hand contact (palm, thenar, three fingertips,
little finger, thumb) adds a normal and a tangential spring-damper, an effectively coupled hand
mass and a cubic stiffening term at its position on the phone; `b` applies the actuator force at
its position (near the bottom edge). Integration at 2 kHz with an exact zero-order-hold
discretisation of the linear part (checked against RK4 at 8 kHz in the tests).

**Motion and sensors** (`imu.py`, `core.py`). Physiological tremor (band-limited, 8-12 Hz peak),
slow sway (< 3 Hz), occasional fidgets, the grip's gravity direction, then IMU sampling at about
100 Hz. Two anti-alias models bracket the unknown real behaviour: `none` (point sampling: the
140-200 Hz carrier aliases into the 0-50 Hz band) and `lpf` (2nd-order Butterworth at 30-50 Hz
before decimation: the carrier is strongly attenuated). Sensor noise, biases, timestamp jitter,
dropped samples, delivery gaps, duplicated timestamps and Core Motion's device-motion stream
(gravity, user acceleration, rotation rate, attitude) are simulated. Conditions: `handheld`,
`twoHanded`, `onTable` (calibration: four stiff, lightly damped corner supports); accessories
(watch, bracelet, other: extra wrist mass coupled through the palm; ring: a stiffer, heavier
finger contact).

## 2. Variability hierarchy and ranges (`sim/params.py`)

Phone (one per phone seed): 68-78 x 140-160 x 7.5-9 mm, 165-235 g; actuator `f0` 140-200 Hz, Q 2-4,
peak force 1.5-3 N along x (cross-coupling up to 0.12), 45-60 mm below the centre; IMU 0-30 mm above
the centre; haptic latency 12-35 ms (trial sd 2-4 ms); IMU 100 Hz +-0.3 %, timestamp jitter 0.1-0.4 ms;
noise 0.6-1.2 mg and 0.5-1.0 mrad/s per sample (x1.4-1.8 in `none` for aliased broadband noise);
gyro bias sd 6 mrad/s per session; single drops 0.1-0.4 %, short gaps in 5 % and long gaps in 0.5 %
of recordings, duplicated timestamps 0.2 %, fidgets in 3 % of handheld recordings.

Person and hand (between people):

| Parameter | Range |
|---|---|
| grip firmness (common stiffness factor; damping x firmness^0.5) | log-normal, sd 0.25 |
| tissue damping factor | log-normal, sd 0.20 |
| hand size (palm position, finger wrap, coupled masses x size^3) | log-normal, sd 0.08 |
| normal stiffness per contact (log-uniform) | palm 2-8 kN/m, thenar 1-4, fingers 0.3-1.5, thumb 0.1-0.8 |
| normal damping per contact (log-uniform) | palm 4-16 N s/m, thenar 2-8, fingers 0.8-3, thumb 0.3-1.5 |
| coupled hand mass per contact | palm 20-70 g, thenar 5-15 g, each finger and thumb 1-4 g |
| tangential / normal ratio | stiffness 0.3-0.7, damping 0.3-0.8 |
| cubic stiffening | 0-0.5 of the linear force at 5 um |
| contact positions, normals | one-handed grip template + 3 mm in-surface jitter, normal jitter 0.06 |
| grip angles | tilt 35-62 deg, in-plane 4 +- 5 deg, side 6 +- 6 deg; standing: tilt 3-15 deg lower |
| left hand | right hand mirrored, plus 2 mm positions, 15 % stiffness and damping, 10 % masses, 3-4 deg angles |
| handedness | 88 % right; the dominant hand 8 % stiffer and more damped |
| tremor | peak 8-12 Hz, bandwidth 1.5-3.5 Hz, 0.8-4 mg and 2-10 mrad/s rms |
| sway | 0.15-0.6 deg rms per axis, corner 0.4-1.2 Hz, x1.3-2.2 standing |
| consistency (scales day and grip variability) | log-uniform 0.7-1.4 |

Within a person (`SESSION_JITTER`, `GRIP_JITTER`; multiplicative factors as log-normal sd, all
scaled by the person's consistency; the day level also by `day_scale`):

| Level | common stiffness | per-contact stiffness, damping | coupled masses | positions | angles | cubic |
|---|---|---|---|---|---|---|
| day (session) | 12 % | 8 % | 6 % | 2.5 mm | 4 deg | 20 % |
| grip (phone picked up again) | 8 % | 6 % | 4 % | 1.5 mm | 3 deg | 10 % |
| trial (one pattern) | - | 3 % | - | 0.3 mm | 0.7 deg | - |

Each trial also has its own latency jitter, IMU sampling phase, sensor noise, tremor amplitude
(10 %) and its own random challenge. Strangers are drawn from the same population as enrolled people.

## 3. Why these ranges

- **People differ more than days do, in every parameter.** Contact stiffness varies with a
  log-sd of about 0.5-0.65 between people against about 0.18 within a person (day + grip +
  trial); coupled masses about 0.4 against 0.07; grip angles 5-8 deg against about 5 deg.
  Contact positions are the closest call (3 mm between people against 2.9 mm within). This is the
  proposal's premise: hands differ in mass, damping, stiffness and grip.
- **The effect size at the sensor is modest, by physics.** At the 140-200 Hz carrier the phone is
  mass-controlled: its own impedance `2 pi f M` is about 210 N s/m at 170 Hz, while the hand's
  contacts add 10-35 N s/m of damping, and their springs and coupled masses resonate below the
  carrier (the phone-on-hand modes lie at a few tens of Hz). Holding the phone therefore changes
  the size of the carrier response by only a few per cent in this model; the hand shows up more in
  how the vibration energy splits between axes (where and how stiffly it constrains the phone),
  in the weakly excited low-frequency modes, and in grip angle and tremor. The contact values are
  order-of-magnitude engineering estimates for a light one-handed grip, not measurements.
- **Day level:** grip force and tissue state (temperature, hydration, fatigue) change from day
  to day, seats and postures change the holding angle; a fresh grip within a day changes
  placement and force a little less.

## 4. What v0 does on the simulator, and why (2026-11-25)

The first full run reported 6.3 % top-1 for 20 people (chance 5 %). Diagnosis, `none` mode, from
`research/results/sim_study_none.*`:

- **The pipeline's mechanics work.** 97.8 % of test recordings pass the quality gate (issues:
  `moving`, `sensorGaps`); median NCC 0.74; the measured lag matches the simulated haptic latency
  (a trial with 35.5 ms latency aligns at 4 samples; study lags 3-4 samples); the tap windows hold
  the vibration peak (the envelope peaks at the onset sample or the next one).
- **The features identify, the Unknown option rejects.** Forced choice (no Unknown) names the
  right person in 62 % of cross-session checks (3-pattern mean; 56 % from one pattern; 70 % on the
  enrollment day). The 6.3 % is the operating point with stranger accept <= 2 %: 92 % of genuine
  checks end Unknown.
- **The threshold tuning is not the problem.** Thresholds chosen on the test checks themselves
  reach at most 9.7 % top-1 at <= 2 % stranger accept (24 % at <= 10 %). Strangers land as close to
  some enrolled prototype as genuine users do to their own (median best distance 45 vs 35, median
  name margin 5.2 vs 5.6). The tuned value had hit the grid edge (absFactor 0.5); the grid now
  reaches 0.25 and the tuned point (0.35, 8) halves the test stranger accept (2.7 % -> 1.3 %) at the
  same top-1.
- **Day-to-day variability does not dwarf person differences.** For the informative features the
  day component is 1-30 % of the between-person variance. Removing it (`day_scale` 0) raises forced
  choice from 62 % to 84 %, but top-1 at the 2 % target only to 17 %.
- **The cause: the identity information is nearly one-dimensional in `none` mode.** One
  discriminant direction dominates (eigenvalues of W^-1 B: 47, 4.7, 2.6, 2.1, then < 1): the split
  of vibration energy between the x and y axes (`ramp_share`, `tap_share`; cross-session Fisher
  20-44), which encodes how the hand constrains the phone. 18 of 36 features have a cross-session
  Fisher ratio below 0.3: tap and ramp amplitudes, regressions, rise, decay and settle times vary
  more from trial to trial than between people, because 100 Hz point samples of an aliased
  140-200 Hz carrier depend on the sampling phase. With 20 people along about one direction, a
  stranger almost always falls next to someone.
- **`lpf` mode** (no aliasing, weaker signal): 81 % of recordings pass (median SNR 4), forced choice
  83 %, top-1 21 % at 0.7 % stranger accept; six discriminant directions above 1, 10 weak features.

A/B studies of the ranges (`python -m evaluation.sim_variants`, reduced study: 20 + 5 people,
sessions 1-3, 10 checks per session; forced choice on session 3; the open-set bounds rest on 50
stranger checks and are noisy):

| Variant | `none` forced choice | `none` bound at 2 % | `lpf` forced choice | `lpf` bound at 2 % |
|---|---|---|---|---|
| as is | 60.0 % | 7.0 % | 82.6 % | 43.6 % |
| no day-to-day variability (`day_scale` 0) | 80.5 % | 13.0 % | 95.4 % | 28.6 % |
| positions: day 1.2 mm, grip 2.6 mm (same total) | 58.5 % | 6.5 % | 83.6 % | 44.1 % |
| + habitual grip height (5 mm) and finger spacing (6 %) between people | 59.5 % | 8.5 % | 83.1 % | 42.1 % |

**Decision: the ranges are kept.** None is physically implausible enough to justify a change,
and the plausible alternatives do not change the results. The near-chance headline was a
reporting problem (only the strict operating point was shown; the reports now show forced choice,
operating points for 1-20 % stranger targets and the separability diagnostics) plus a real
property of v0's features under aliasing. Whether real phones alias (`none`) or filter (`lpf`),
and how strongly real hands change the response, are exactly the questions the pilot answers.

## 5. Reproducing

```bash
cd research
python -m pytest -q                                                     # tests (simulator included)
nohup python -m evaluation.run_sim_study --mode both --workers 16 > ~/7oda-work/sim/study.log 2>&1 &
python -m evaluation.sim_variants --mode none                           # section 4 A/B table (also --mode lpf)
```

Results: `research/results/sim_study_none.{json,md}` and `sim_study_lpf.{json,md}` (seeds: study
2026, phone 7). A full run takes about 12 minutes with 16 workers (about 25 s per 30 000 recordings
simulated and processed, about one minute of evaluation per mode, two sensitivity runs per mode).
Processed items are cached in `~/7oda-work/sim/items_<mode>[_day<scale>].pkl.gz`; re-running with
the same configuration only re-evaluates. Delete the cache after changing the simulator or
`auditor_ref`. Keep workers at 16 or fewer on the shared server: the container's memory limit
(`/sys/fs/cgroup/memory.max`, 64 GB) is far below what `free` shows.

## 6. What the pilot must calibrate

- Anti-aliasing of the real 100 Hz streams: look at the raw samples of single taps (aliased
  carrier = `none`; smooth envelope = `lpf`).
- How much a hand changes the response: amplitude and decay of the same pattern on a table, in a
  light grip and in a firm grip.
- Day versus grip variability: the same person on different days against re-grips within a day.
- Tremor and sway levels, and the quality-gate pass rates per gate.
