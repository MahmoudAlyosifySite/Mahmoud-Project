# Simulated study, IMU mode `lpf`

IMU mode `lpf`, 20 enrolled + 5 strangers, 30 checks per person per session, phone seed 7, study seed 2026. Enrollment: session 1; thresholds tuned on session 2; test sessions [3, 4].

Simulated numbers check the pipeline's mechanics; they are not evidence about real hands (DECISIONS D9).

How to read: *top-1 name* counts checks that were answered with the right name; Unknown answers count as misses. With thresholds tuned for a low stranger accept rate, a model that ranks the right person first can still answer Unknown most of the time, so the forced-choice (closed-set) accuracy and the operating points below show how much of the gap is the features and how much is the Unknown option.

## Headline (test sessions, sequential checks up to 3 patterns)

| setting | top-1 name | hand (given name) | name+hand | stranger accept | false Unknown | wrong name | hand unclear | quality fail | patterns/answer | s/answer |
|---|---|---|---|---|---|---|---|---|---|---|
| tuned (absFactor 1, nameMargin 20) | 20.8% | 97.8% | 18.8% | 0.7% | 77.8% | 0.1% | 1.5% | 1.4% | 2.67 | 6.21 |
| SPEC defaults (1.5, 6) | 67.8% | 95.2% | 59.6% | 45.0% | 22.9% | 7.8% | 6.1% | 1.4% | 1.89 | 4.40 |

Tuning target stranger accept <= 2% on the tuning session: met. Checks: 1200 enrolled, 300 stranger.

## Forced choice (no Unknown option)

Nearest prototype for enrolled people's checks: first passing pattern, and the mean distance of the first 3 passing patterns. Chance = 5.0%.

| sessions | checks | top-1, 1 pattern | top-1, mean of patterns | name+hand |
|---|---|---|---|---|
| test [3, 4] | 1198 | 73.5% | 83.1% | 74.2% |
| session 2 | 600 | 69.7% | 81.2% | 68.8% |
| session 3 | 599 | 71.6% | 81.0% | 74.1% |
| session 4 | 599 | 75.3% | 85.3% | 74.3% |

Same-day reference (session 1: enroll on the sitting and standing rounds, test the fresh-grip round, 1 pattern): 81.1%.

## Operating points (thresholds chosen on the tuning session for each stranger target)

| target | absFactor | nameMargin | tuning: stranger / top-1 | test: top-1 name | test: stranger accept | test: false Unknown | test: wrong name |
|---|---|---|---|---|---|---|---|
| 1.0% | 1.5 | 30 | 0.7% / 9.3% | 9.3% | 0.0% | 89.2% | 0.0% |
| 2.0% | 1 | 20 | 2.0% / 19.2% | 20.8% | 0.7% | 77.8% | 0.1% |
| 5.0% | 3 | 20 | 4.0% / 22.0% | 22.8% | 0.7% | 75.8% | 0.1% |
| 10.0% | 0.75 | 10 | 10.0% / 35.7% | 43.0% | 6.0% | 54.0% | 1.6% |
| 20.0% | 1 | 10 | 18.0% / 47.2% | 51.7% | 14.0% | 45.1% | 1.8% |

## Where the identity information is (test sessions, enrolled people)

Variance of each feature split into between person-hand, session (day), grip (check) and trial parts; Fisher = between / within (same-session: grip + trial; cross-session: + session).

| feature | between | session | grip | trial | Fisher same-session | Fisher cross-session |
|---|---|---|---|---|---|---|
| 14 tap_log_accel_gyro_ratio | 0.53 | 0.015 | 0.0872 | 0.122 | 2.53 | 2.36 |
| 11 tap_share_gx | 0.0159 | 0.000116 | 0.00229 | 0.00482 | 2.23 | 2.20 |
| 33 gravity_x | 0.013 | 0.000933 | 0.0052 | 0.000126 | 2.43 | 2.07 |
| 24 ramp_share_ay | 0.0032 | 8.51e-05 | 0.000498 | 0.000964 | 2.19 | 2.07 |
| 31 quiet_rms_accel | 0.0681 | 0.006 | 0.0138 | 0.0156 | 2.32 | 1.92 |
| 32 quiet_rms_gyro | 0.109 | 0.012 | 0.0295 | 0.0221 | 2.11 | 1.72 |
| 30 quiet_tremor_gyro | 0.63 | 0.0622 | 0.169 | 0.188 | 1.77 | 1.50 |
| 26 ramp_share_gx | 0.0118 | 0.000191 | 0.0027 | 0.00558 | 1.42 | 1.39 |
| 23 ramp_share_ax | 0.00412 | 0.000132 | 0.00109 | 0.00197 | 1.34 | 1.29 |
| 29 quiet_tremor_accel | 0.413 | 0.0394 | 0.125 | 0.172 | 1.39 | 1.23 |

10 of 36 features have a cross-session Fisher ratio below 0.3 (full table in the JSON file, `diagnostics.features`). Discriminant eigenvalues (W^-1 B): 9.87, 3.61, 2.70, 2.10, 1.76, 1.38, 1.00, 0.90; 6 directions with ratio > 1.

Optimistic bound for the threshold rule (abs and name thresholds chosen on the test checks themselves, single decision on the 3-pattern mean): top-1 name 35.1% at stranger accept <= 2.0%, 51.7% at stranger accept <= 10.0%. Median best distance: genuine 39.5, stranger 49.5; median name margin: genuine 9.7, stranger 3.6.

## Per session (tuned thresholds)

| session | top-1 name | hand (given name) | name+hand | stranger accept | false Unknown | wrong name | hand unclear | quality fail | patterns/answer | s/answer |
|---|---|---|---|---|---|---|---|---|---|---|
| session 2 | 19.2% | 92.0% | 15.3% | 2.0% | 79.2% | 0.2% | 2.5% | 1.5% | 2.67 | 6.40 |
| session 3 | 21.7% | 99.2% | 20.2% | 1.3% | 77.2% | 0.0% | 1.3% | 1.2% | 2.65 | 6.24 |
| session 4 | 19.8% | 96.3% | 17.5% | 0.0% | 78.3% | 0.2% | 1.7% | 1.7% | 2.69 | 6.19 |

## Threshold trade-off on the tuning session (Pareto frontier)

| absFactor | nameMargin | stranger accept | false Unknown | top-1 name | wrong name |
|---|---|---|---|---|---|
| 1 | 30 | 0.0% | 91.0% | 7.5% | 0.0% |
| 1.5 | 30 | 0.7% | 89.2% | 9.3% | 0.0% |
| 0.75 | 20 | 1.3% | 84.0% | 14.5% | 0.0% |
| 1 | 20 | 2.0% | 79.2% | 19.2% | 0.2% |
| 1.5 | 20 | 3.3% | 76.5% | 21.8% | 0.2% |
| 2.5 | 20 | 4.0% | 76.3% | 22.0% | 0.2% |
| 0.75 | 12 | 6.7% | 66.5% | 30.8% | 1.2% |
| 1.25 | 15 | 9.3% | 65.0% | 32.8% | 0.7% |
| 0.75 | 10 | 10.0% | 60.3% | 35.7% | 2.5% |
| 1 | 12 | 12.7% | 56.8% | 39.7% | 2.0% |
| 0.75 | 8 | 14.0% | 52.0% | 42.3% | 4.2% |
| 1 | 10 | 18.0% | 47.5% | 47.2% | 3.8% |
| 0.75 | 6 | 22.0% | 46.8% | 46.2% | 5.5% |
| 1.25 | 10 | 23.3% | 43.7% | 50.7% | 4.2% |
| 1.5 | 10 | 25.3% | 43.5% | 50.8% | 4.2% |
| 2 | 10 | 26.0% | 43.3% | 51.0% | 4.2% |
| 1 | 8 | 27.3% | 37.0% | 55.2% | 6.3% |
| 0.75 | 3 | 34.0% | 33.8% | 52.5% | 12.2% |
| 1.25 | 8 | 35.3% | 32.7% | 58.8% | 7.0% |
| 0.75 | 2 | 38.0% | 30.5% | 52.7% | 15.3% |
| 1 | 6 | 41.3% | 29.7% | 60.5% | 8.3% |
| 0.75 | 1 | 42.7% | 27.2% | 53.8% | 17.5% |
| 0.75 | 0 | 43.3% | 26.2% | 53.0% | 19.3% |
| 1.25 | 6 | 49.3% | 24.8% | 64.7% | 9.0% |
| 1.5 | 6 | 54.7% | 24.0% | 65.5% | 9.0% |
| 2 | 6 | 56.0% | 23.8% | 65.7% | 9.0% |
| 1 | 4 | 57.3% | 18.8% | 65.5% | 14.2% |
| 1 | 3 | 61.3% | 15.0% | 66.7% | 16.8% |
| 1.25 | 4 | 68.0% | 14.3% | 69.3% | 14.8% |
| 1 | 2 | 69.3% | 10.2% | 66.8% | 21.5% |
| 1 | 1 | 74.0% | 6.7% | 67.3% | 24.5% |
| 1 | 0 | 76.7% | 5.5% | 66.0% | 27.0% |
| 1.5 | 2 | 85.3% | 5.0% | 71.2% | 22.3% |
| 1.25 | 1 | 86.7% | 2.3% | 70.7% | 25.5% |
| 1.25 | 0 | 90.0% | 1.0% | 69.3% | 28.2% |
| 1.5 | 0 | 93.3% | 0.2% | 69.7% | 28.7% |
| 2 | 0 | 94.7% | 0.0% | 69.7% | 28.8% |

## Patterns per check (re-tuned each)

| setting | top-1 name | hand (given name) | name+hand | stranger accept | false Unknown | wrong name | hand unclear | quality fail | patterns/answer | s/answer |
|---|---|---|---|---|---|---|---|---|---|---|
| max 1 pattern(s) (absFactor 1, nameMargin 20) | 14.1% | 99.3% | 12.3% | 1.0% | 84.4% | 0.1% | 1.7% | 1.4% | 0.98 | 2.34 |
| max 2 pattern(s) (absFactor 1, nameMargin 20) | 18.4% | 98.5% | 16.6% | 0.7% | 80.1% | 0.1% | 1.6% | 1.4% | 1.85 | 4.33 |
| max 3 pattern(s) (absFactor 1, nameMargin 20) | 20.8% | 97.8% | 18.8% | 0.7% | 77.8% | 0.1% | 1.5% | 1.4% | 2.67 | 6.21 |

## Feature sets (re-tuned each)

| setting | top-1 name | hand (given name) | name+hand | stranger accept | false Unknown | wrong name | hand unclear | quality fail | patterns/answer | s/answer |
|---|---|---|---|---|---|---|---|---|---|---|
| all 36 features (absFactor 1, nameMargin 20) | 20.8% | 97.8% | 18.8% | 0.7% | 77.8% | 0.1% | 1.5% | 1.4% | 2.67 | 6.21 |
| taps+quiet+gravity (no ramps) (absFactor 0.75, nameMargin 15) | 15.8% | 93.2% | 11.5% | 1.3% | 82.8% | 0.1% | 3.4% | 1.4% | 2.77 | 6.46 |
| taps only (absFactor 1.25, nameMargin 12) | 10.8% | 67.6% | 4.0% | 1.3% | 87.8% | 0.1% | 4.8% | 1.4% | 2.84 | 6.64 |
| ramps+quiet+gravity (no taps) (absFactor 1, nameMargin 15) | 11.3% | 98.4% | 10.6% | 1.7% | 87.1% | 0.2% | 0.6% | 1.4% | 2.76 | 6.50 |
| no gyro-derived features (absFactor 0.5, nameMargin 10) | 5.3% | 100.0% | 5.3% | 2.3% | 92.9% | 0.3% | 0.0% | 1.4% | 2.83 | 6.65 |
| no gravity (absFactor 0.75, nameMargin 15) | 15.2% | 94.8% | 12.2% | 0.7% | 83.2% | 0.2% | 2.3% | 1.4% | 2.76 | 6.44 |
| accelerometer only (gyro stream zeroed) (absFactor 0.25, nameMargin 0) | 1.2% | 100.0% | 1.2% | 1.3% | 96.8% | 1.8% | 0.0% | 0.2% | 2.94 | 5.98 |

## Enrollment sittings (re-tuned each)

| setting | top-1 name | hand (given name) | name+hand | stranger accept | false Unknown | wrong name | hand unclear | quality fail | patterns/answer | s/answer |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 sitting (15/hand) (absFactor 1, nameMargin 20) | 20.8% | 97.8% | 18.8% | 0.7% | 77.8% | 0.1% | 1.5% | 1.4% | 2.67 | 6.21 |
| 2 sittings (15 + 5/hand) (absFactor 0.75, nameMargin 20) | 13.0% | 98.7% | 12.2% | 0.3% | 85.6% | 0.0% | 0.6% | 1.4% | 2.76 | 6.45 |

## Accessories (enrolled bare; checks with the person's own accessory)

| setting | top-1 name | hand (given name) | name+hand | stranger accept | false Unknown | wrong name | hand unclear | quality fail | patterns/answer | s/answer |
|---|---|---|---|---|---|---|---|---|---|---|
| before quick add | 24.2% | 96.7% | 22.2% | – | 73.8% | 0.0% | 1.2% | 2.0% | 2.58 | 5.94 |
| after quick add | 23.0% | 95.5% | 21.0% | – | 74.5% | 0.5% | 1.0% | 2.0% | 2.56 | 5.92 |
| bare-hand checks after quick add | 21.5% | 97.1% | 19.7% | 1.7% | 77.1% | 0.0% | 1.2% | 1.4% | 2.67 | 6.19 |

| accessory | top-1 before | top-1 after |
|---|---|---|
| bracelet | 34.2% | 35.8% |
| ring | 17.1% | 15.0% |
| watch | 22.9% | 20.0% |

## Cross-hand test (enroll one hand, check with the other: the right answer is Unknown)

| setting | checks | Unknown | named self | named someone else | quality fail | same-hand control top-1 |
|---|---|---|---|---|---|---|
| enroll right, test left | 600 | 92.2% | 3.0% | 2.7% | 2.2% | 30.0% |
| enroll left, test right | 600 | 91.0% | 7.5% | 0.8% | 0.7% | 17.7% |

## Replay attack (victim's recorded responses fed to fresh challenges, padded to length)

- patterns rejected by the quality gate: 56.2% of 2400 (median NCC 0.30; issues: weakAlignment 1179, weakSignal 459, moving 58, sensorGaps 3)
- checks ending in a quality failure: 10.0% of 200
- checks accepted as the victim: 9.5%; as anyone: 9.5%

## Mimicry proxy (stranger copies the victim's grip angles)

- 2000 checks: accepted as the victim 0.6%, accepted as anyone 2.4%
- baseline (ordinary stranger checks, same session): accepted as anyone 1.3%; chance of naming one given victim 5.0% x accept rate

## Sensitivity to day-to-day variability (main setting, re-tuned)

| day-to-day scale | top-1 name | hand (given name) | name+hand | stranger accept | false Unknown | wrong name | hand unclear | quality fail | patterns/answer | s/answer |
|---|---|---|---|---|---|---|---|---|---|---|
| day_scale 1 | 20.8% | 97.8% | 18.8% | 0.7% | 77.8% | 0.1% | 1.5% | 1.4% | 2.67 | 6.21 |
| day_scale 0 | 26.0% | 99.0% | 24.0% | 0.7% | 72.5% | 0.0% | 1.8% | 1.5% | 2.62 | 6.04 |
| day_scale 0.5 | 31.6% | 99.2% | 29.5% | 2.7% | 66.6% | 0.3% | 1.8% | 1.5% | 2.55 | 5.91 |

Forced choice (no Unknown option, mean of up to 3 patterns): day_scale 1 83.1%, day_scale 0 92.4%, day_scale 0.5 90.2%.

day_scale 0 means every session reproduces the enrollment day's grip and tissue state (only fresh-grip and trial variation remain); 1 is the default simulator.

## Quality gate on simulated recordings

| role | n | pass | median SNR | median NCC | issues |
|---|---|---|---|---|---|
| calibration | 20 | 100.0% | 340.3 | 0.69 |  |
| enroll | 840 | 81.2% | 3.7 | 0.62 | weakSignal 146, weakAlignment 17, moving 13, sensorGaps 4 |
| test | 27000 | 80.7% | 3.7 | 0.62 | weakSignal 4584, weakAlignment 830, moving 491, sensorGaps 114 |
| accessoryTest | 4800 | 81.8% | 3.8 | 0.62 | weakSignal 758, weakAlignment 160, moving 105, sensorGaps 23 |
| mimic | 24000 | 74.7% | 3.0 | 0.60 | weakSignal 5465, weakAlignment 953, moving 497, sensorGaps 95 |

## Most frequent confusions (test sessions, tuned)

stranger -> P02 (2), P09 -> P04 (1)

Full confusion matrix in the JSON file (`confusion`).
