# Simulated study, IMU mode `none`

IMU mode `none`, 20 enrolled + 5 strangers, 30 checks per person per session, phone seed 7, study seed 2026. Enrollment: session 1; thresholds tuned on session 2; test sessions [3, 4].

Simulated numbers check the pipeline's mechanics; they are not evidence about real hands (DECISIONS D9).

How to read: *top-1 name* counts checks that were answered with the right name; Unknown answers count as misses. With thresholds tuned for a low stranger accept rate, a model that ranks the right person first can still answer Unknown most of the time, so the forced-choice (closed-set) accuracy and the operating points below show how much of the gap is the features and how much is the Unknown option.

## Headline (test sessions, sequential checks up to 3 patterns)

| setting | top-1 name | hand (given name) | name+hand | stranger accept | false Unknown | wrong name | hand unclear | quality fail | patterns/answer | s/answer |
|---|---|---|---|---|---|---|---|---|---|---|
| tuned (absFactor 0.35, nameMargin 8) | 6.3% | 100.0% | 6.3% | 1.3% | 92.2% | 1.4% | 0.0% | 0.0% | 2.90 | 5.41 |
| SPEC defaults (1.5, 6) | 45.2% | 100.0% | 45.0% | 61.3% | 37.4% | 17.3% | 0.3% | 0.0% | 1.96 | 3.66 |

Tuning target stranger accept <= 2% on the tuning session: met. Checks: 1200 enrolled, 300 stranger.

## Forced choice (no Unknown option)

Nearest prototype for enrolled people's checks: first passing pattern, and the mean distance of the first 3 passing patterns. Chance = 5.0%.

| sessions | checks | top-1, 1 pattern | top-1, mean of patterns | name+hand |
|---|---|---|---|---|
| test [3, 4] | 1200 | 56.3% | 62.1% | 61.4% |
| session 2 | 600 | 52.0% | 56.2% | 56.2% |
| session 3 | 600 | 58.2% | 65.7% | 65.0% |
| session 4 | 600 | 54.5% | 58.5% | 57.8% |

Same-day reference (session 1: enroll on the sitting and standing rounds, test the fresh-grip round, 1 pattern): 69.9%.

## Operating points (thresholds chosen on the tuning session for each stranger target)

| target | absFactor | nameMargin | tuning: stranger / top-1 | test: top-1 name | test: stranger accept | test: false Unknown | test: wrong name |
|---|---|---|---|---|---|---|---|
| 1.0% | 0.35 | 12 | 0.7% / 2.2% | 3.0% | 1.0% | 96.8% | 0.2% |
| 2.0% | 0.35 | 8 | 2.0% / 5.8% | 6.3% | 1.3% | 92.2% | 1.4% |
| 5.0% | 0.35 | 1 | 4.7% / 11.3% | 11.9% | 5.7% | 81.8% | 6.2% |
| 10.0% | 0.5 | 6 | 10.0% / 21.2% | 24.7% | 13.0% | 67.3% | 8.0% |
| 20.0% | 0.5 | 3 | 16.0% / 28.5% | 31.7% | 21.3% | 51.6% | 16.8% |

## Where the identity information is (test sessions, enrolled people)

Variance of each feature split into between person-hand, session (day), grip (check) and trial parts; Fisher = between / within (same-session: grip + trial; cross-session: + session).

| feature | between | session | grip | trial | Fisher same-session | Fisher cross-session |
|---|---|---|---|---|---|---|
| 24 ramp_share_ay | 0.0038 | 4.55e-05 | 3.53e-05 | 4.75e-06 | 94.87 | 44.37 |
| 23 ramp_share_ax | 0.00405 | 4.61e-05 | 5.01e-05 | 5.15e-06 | 73.31 | 39.95 |
| 9 tap_share_ay | 0.00373 | 5.02e-05 | 5.52e-05 | 4.5e-05 | 37.24 | 24.80 |
| 8 tap_share_ax | 0.00398 | 5.34e-05 | 7.88e-05 | 6.99e-05 | 26.74 | 19.67 |
| 25 ramp_share_az | 6.85e-05 | 5.16e-06 | 1.23e-05 | 5.76e-07 | 5.33 | 3.80 |
| 27 ramp_share_gy | 0.00507 | 0.000539 | 0.000635 | 0.000252 | 5.72 | 3.56 |
| 28 ramp_share_gz | 0.00543 | 0.000582 | 0.000795 | 0.000558 | 4.01 | 2.81 |
| 10 tap_share_az | 8.31e-05 | 5.78e-06 | 1.68e-05 | 1.15e-05 | 2.94 | 2.44 |
| 33 gravity_x | 0.013 | 0.000945 | 0.00522 | 0.00012 | 2.42 | 2.06 |
| 32 quiet_rms_gyro | 0.0898 | 0.0108 | 0.028 | 0.0227 | 1.77 | 1.46 |

18 of 36 features have a cross-session Fisher ratio below 0.3 (full table in the JSON file, `diagnostics.features`). Discriminant eigenvalues (W^-1 B): 47.47, 4.74, 2.65, 2.09, 0.90, 0.62, 0.42, 0.37; 4 directions with ratio > 1.

Optimistic bound for the threshold rule (abs and name thresholds chosen on the test checks themselves, single decision on the 3-pattern mean): top-1 name 9.7% at stranger accept <= 2.0%, 23.9% at stranger accept <= 10.0%. Median best distance: genuine 35.1, stranger 45.0; median name margin: genuine 5.6, stranger 5.2.

## Per session (tuned thresholds)

| session | top-1 name | hand (given name) | name+hand | stranger accept | false Unknown | wrong name | hand unclear | quality fail | patterns/answer | s/answer |
|---|---|---|---|---|---|---|---|---|---|---|
| session 2 | 5.8% | 100.0% | 5.8% | 2.0% | 93.5% | 0.7% | 0.0% | 0.0% | 2.91 | 5.49 |
| session 3 | 6.0% | 100.0% | 6.0% | 1.3% | 92.5% | 1.5% | 0.0% | 0.0% | 2.90 | 5.44 |
| session 4 | 6.7% | 100.0% | 6.7% | 1.3% | 92.0% | 1.3% | 0.0% | 0.0% | 2.89 | 5.38 |

## Threshold trade-off on the tuning session (Pareto frontier)

| absFactor | nameMargin | stranger accept | false Unknown | top-1 name | wrong name |
|---|---|---|---|---|---|
| 0.25 | 0 | 0.0% | 97.7% | 1.7% | 0.7% |
| 2.5 | 30 | 0.7% | 97.2% | 2.0% | 0.8% |
| 0.35 | 10 | 1.3% | 96.2% | 3.5% | 0.3% |
| 0.35 | 8 | 2.0% | 93.5% | 5.8% | 0.7% |
| 0.5 | 12 | 2.7% | 91.7% | 6.5% | 1.8% |
| 0.35 | 3 | 3.3% | 87.7% | 9.5% | 2.8% |
| 0.35 | 1 | 4.7% | 84.8% | 11.3% | 3.8% |
| 0.35 | 0 | 5.3% | 83.7% | 11.5% | 4.8% |
| 0.5 | 8 | 7.3% | 79.5% | 16.2% | 4.3% |
| 0.5 | 6 | 10.0% | 71.5% | 21.2% | 7.3% |
| 0.5 | 4 | 14.0% | 61.7% | 26.5% | 11.8% |
| 0.5 | 3 | 16.0% | 56.7% | 28.5% | 14.8% |
| 0.5 | 2 | 20.7% | 51.3% | 30.7% | 18.0% |
| 0.5 | 1 | 21.3% | 47.8% | 32.0% | 20.2% |
| 0.5 | 0 | 23.3% | 45.2% | 31.8% | 23.0% |
| 0.75 | 4 | 49.3% | 35.3% | 40.7% | 24.0% |
| 0.75 | 3 | 55.3% | 26.2% | 43.3% | 30.5% |
| 0.75 | 2 | 64.0% | 19.8% | 45.3% | 34.8% |
| 0.75 | 1 | 66.7% | 16.2% | 46.2% | 37.7% |
| 0.75 | 0 | 70.7% | 13.8% | 45.5% | 40.7% |
| 1 | 2 | 83.3% | 11.3% | 49.3% | 39.3% |
| 1 | 1 | 88.0% | 7.3% | 50.3% | 42.3% |
| 1 | 0 | 89.3% | 5.0% | 49.8% | 45.2% |
| 1.25 | 1 | 97.3% | 3.7% | 51.5% | 44.8% |
| 1.25 | 0 | 98.7% | 2.0% | 50.3% | 47.7% |
| 1.5 | 0 | 99.3% | 1.3% | 51.0% | 47.7% |
| 6 | 0 | 100.0% | 0.0% | 52.2% | 47.8% |

## Patterns per check (re-tuned each)

| setting | top-1 name | hand (given name) | name+hand | stranger accept | false Unknown | wrong name | hand unclear | quality fail | patterns/answer | s/answer |
|---|---|---|---|---|---|---|---|---|---|---|
| max 1 pattern(s) (absFactor 0.35, nameMargin 6) | 6.2% | 100.0% | 6.2% | 1.7% | 91.9% | 1.8% | 0.0% | 0.0% | 1.00 | 1.86 |
| max 2 pattern(s) (absFactor 0.35, nameMargin 8) | 5.5% | 100.0% | 5.5% | 1.3% | 93.1% | 1.4% | 0.0% | 0.0% | 1.95 | 3.65 |
| max 3 pattern(s) (absFactor 0.35, nameMargin 8) | 6.3% | 100.0% | 6.3% | 1.3% | 92.2% | 1.4% | 0.0% | 0.0% | 2.90 | 5.41 |

## Feature sets (re-tuned each)

| setting | top-1 name | hand (given name) | name+hand | stranger accept | false Unknown | wrong name | hand unclear | quality fail | patterns/answer | s/answer |
|---|---|---|---|---|---|---|---|---|---|---|
| all 36 features (absFactor 0.35, nameMargin 8) | 6.3% | 100.0% | 6.3% | 1.3% | 92.2% | 1.4% | 0.0% | 0.0% | 2.90 | 5.41 |
| taps+quiet+gravity (no ramps) (absFactor 0.35, nameMargin 8) | 7.6% | 100.0% | 7.6% | 2.0% | 91.2% | 1.2% | 0.0% | 0.0% | 2.88 | 5.38 |
| taps only (absFactor 1, nameMargin 10) | 3.2% | 97.4% | 3.2% | 3.0% | 95.1% | 1.7% | 0.0% | 0.0% | 2.92 | 5.46 |
| ramps+quiet+gravity (no taps) (absFactor 0.25, nameMargin 6) | 2.8% | 100.0% | 2.8% | 0.3% | 96.3% | 0.9% | 0.0% | 0.0% | 2.95 | 5.51 |
| no gyro-derived features (absFactor 0.25, nameMargin 2) | 6.0% | 100.0% | 6.0% | 2.7% | 87.8% | 6.2% | 0.0% | 0.0% | 2.82 | 5.27 |
| no gravity (absFactor 0.35, nameMargin 8) | 5.1% | 100.0% | 5.1% | 4.3% | 93.7% | 1.2% | 0.0% | 0.0% | 2.91 | 5.44 |
| accelerometer only (gyro stream zeroed) (absFactor 0.25, nameMargin 3) | 4.9% | 100.0% | 4.9% | 2.0% | 90.2% | 4.8% | 0.0% | 0.0% | 2.85 | 5.33 |

## Enrollment sittings (re-tuned each)

| setting | top-1 name | hand (given name) | name+hand | stranger accept | false Unknown | wrong name | hand unclear | quality fail | patterns/answer | s/answer |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 sitting (15/hand) (absFactor 0.35, nameMargin 8) | 6.3% | 100.0% | 6.3% | 1.3% | 92.2% | 1.4% | 0.0% | 0.0% | 2.90 | 5.41 |
| 2 sittings (15 + 5/hand) (absFactor 1.25, nameMargin 15) | 7.8% | 100.0% | 7.8% | 4.0% | 91.6% | 0.6% | 0.0% | 0.0% | 2.87 | 5.37 |

## Accessories (enrolled bare; checks with the person's own accessory)

| setting | top-1 name | hand (given name) | name+hand | stranger accept | false Unknown | wrong name | hand unclear | quality fail | patterns/answer | s/answer |
|---|---|---|---|---|---|---|---|---|---|---|
| before quick add | 4.5% | 100.0% | 4.5% | – | 95.0% | 0.5% | 0.0% | 0.0% | 2.92 | 5.52 |
| after quick add | 5.0% | 100.0% | 4.8% | – | 93.5% | 1.5% | 0.2% | 0.0% | 2.90 | 5.48 |
| bare-hand checks after quick add | 6.1% | 100.0% | 6.1% | 1.3% | 91.7% | 2.2% | 0.0% | 0.0% | 2.88 | 5.39 |

| accessory | top-1 before | top-1 after |
|---|---|---|
| bracelet | 4.2% | 6.7% |
| ring | 7.1% | 5.7% |
| watch | 2.1% | 2.9% |

## Cross-hand test (enroll one hand, check with the other: the right answer is Unknown)

| setting | checks | Unknown | named self | named someone else | quality fail | same-hand control top-1 |
|---|---|---|---|---|---|---|
| enroll right, test left | 600 | 100.0% | 0.0% | 0.0% | 0.0% | 5.0% |
| enroll left, test right | 600 | 100.0% | 0.0% | 0.0% | 0.0% | 3.3% |

## Replay attack (victim's recorded responses fed to fresh challenges, padded to length)

- patterns rejected by the quality gate: 33.8% of 2400 (median NCC 0.37; issues: weakAlignment 763, moving 58, weakSignal 21, sensorGaps 3)
- checks ending in a quality failure: 0.5% of 200
- checks accepted as the victim: 0.0%; as anyone: 0.0%

## Mimicry proxy (stranger copies the victim's grip angles)

- 2000 checks: accepted as the victim 0.4%, accepted as anyone 2.9%
- baseline (ordinary stranger checks, same session): accepted as anyone 1.3%; chance of naming one given victim 5.0% x accept rate

## Sensitivity to day-to-day variability (main setting, re-tuned)

| day-to-day scale | top-1 name | hand (given name) | name+hand | stranger accept | false Unknown | wrong name | hand unclear | quality fail | patterns/answer | s/answer |
|---|---|---|---|---|---|---|---|---|---|---|
| day_scale 1 | 6.3% | 100.0% | 6.3% | 1.3% | 92.2% | 1.4% | 0.0% | 0.0% | 2.90 | 5.41 |
| day_scale 0 | 17.0% | 100.0% | 17.0% | 2.0% | 82.1% | 0.9% | 0.0% | 0.0% | 2.75 | 5.14 |
| day_scale 0.5 | 14.7% | 99.4% | 14.6% | 2.0% | 83.5% | 1.8% | 0.0% | 0.0% | 2.77 | 5.18 |

Forced choice (no Unknown option, mean of up to 3 patterns): day_scale 1 62.1%, day_scale 0 83.5%, day_scale 0.5 75.7%.

day_scale 0 means every session reproduces the enrollment day's grip and tissue state (only fresh-grip and trial variation remain); 1 is the default simulator.

## Quality gate on simulated recordings

| role | n | pass | median SNR | median NCC | issues |
|---|---|---|---|---|---|
| calibration | 20 | 100.0% | 3298.5 | 0.72 |  |
| enroll | 840 | 98.1% | 380.7 | 0.75 | moving 13, sensorGaps 4, weakAlignment 1 |
| test | 27000 | 97.8% | 378.3 | 0.74 | moving 491, sensorGaps 114, resting 1, weakAlignment 1 |
| accessoryTest | 4800 | 97.4% | 383.0 | 0.74 | moving 105, sensorGaps 23, weakAlignment 1 |
| mimic | 24000 | 97.5% | 315.0 | 0.74 | moving 496, sensorGaps 95, weakAlignment 5 |

## Most frequent confusions (test sessions, tuned)

stranger -> P03 (2), P20 -> P11 (2), P19 -> P14 (2), P10 -> P09 (2), P04 -> P14 (2), P02 -> P13 (2), stranger -> P02 (1), stranger -> P01 (1), P19 -> P09 (1), P18 -> P17 (1), P18 -> P04 (1), P10 -> P16 (1)

Full confusion matrix in the JSON file (`confusion`).
