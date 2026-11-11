# Synthetic pilot (frames_span1)

Seeds: [1, 2, 3, 4, 5]

Config:

```json
{
  "families": [
    "energy",
    "permanence",
    "continuity",
    "stochastic"
  ],
  "pilot": {
    "alpha": null,
    "hidden": [
      128,
      128
    ],
    "lag_high": 90,
    "lag_low": 30,
    "max_invariants": 16,
    "min_correlation": 0.9,
    "n_test": 200,
    "n_train": 300,
    "n_val": 100,
    "noise": 0.0,
    "pca_dim": null,
    "predictor_iters": 150,
    "span": 1,
    "window": 12
  },
  "scenario": {
    "T": 120,
    "ball_counts": [
      2,
      3
    ],
    "energy_factors": [
      0.5,
      1.6
    ],
    "gain_factor": 1.6,
    "improbable_scale": 3.0,
    "rare_sigmas": [
      3.0,
      4.0
    ],
    "rough_jitter_deg": 12.0,
    "settle": 20,
    "t_event_high": 80,
    "t_event_low": 40,
    "teleport_min_dist": 0.3
  }
}
```

## Verdict

**GO**: H2 holds in 2 of 3 conservation families (rule: at least 2).

| Family | Best prediction-error score | Its AUROC | Invariant AUROC | Gain | H1 | H2 |
|---|---|---|---|---|---|---|
| energy | pe_rel | 0.425 | 1.000 | +0.575 | yes | yes |
| permanence | pe_max | 1.000 | 1.000 | +0.000 | no | no |
| stochastic | pe_max | 0.640 | 1.000 | +0.360 | yes | yes |

AUROC here is impossible vs improbable: 0.5 is chance, 1.0 is perfect, below 0.5 means the score ranks lawful rare clips above impossible ones.

## All AUROCs (mean +/- std over seeds)

### energy

| Score | impossible vs possible | impossible vs improbable | improbable vs possible |
|---|---|---|---|
| inv | 1.000 +/- 0.000 | 1.000 +/- 0.001 | 0.921 +/- 0.020 |
| inv_rel | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.912 +/- 0.022 |
| pe_max | 0.564 +/- 0.031 | 0.026 +/- 0.006 | 0.990 +/- 0.003 |
| pe_mean | 0.577 +/- 0.016 | 0.022 +/- 0.010 | 0.990 +/- 0.007 |
| pe_rel | 0.497 +/- 0.054 | 0.425 +/- 0.081 | 0.576 +/- 0.061 |
| oracle_pe | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.500 +/- 0.000 |

### permanence

| Score | impossible vs possible | impossible vs improbable | improbable vs possible |
|---|---|---|---|
| inv | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.922 +/- 0.005 |
| inv_rel | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.913 +/- 0.007 |
| pe_max | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.989 +/- 0.003 |
| pe_mean | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.990 +/- 0.005 |
| pe_rel | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.575 +/- 0.070 |
| oracle_pe | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.500 +/- 0.000 |

### continuity

| Score | impossible vs possible | impossible vs improbable | improbable vs possible |
|---|---|---|---|
| inv | 0.564 +/- 0.004 | 0.100 +/- 0.017 | 0.919 +/- 0.015 |
| inv_rel | 0.564 +/- 0.003 | 0.111 +/- 0.018 | 0.910 +/- 0.015 |
| pe_max | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.986 +/- 0.002 |
| pe_mean | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.988 +/- 0.006 |
| pe_rel | 1.000 +/- 0.000 | 0.999 +/- 0.001 | 0.568 +/- 0.058 |
| oracle_pe | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.500 +/- 0.000 |

### stochastic

| Score | impossible vs possible | impossible vs improbable | improbable vs possible |
|---|---|---|---|
| inv | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.513 +/- 0.027 |
| inv_rel | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.514 +/- 0.027 |
| pe_max | 0.653 +/- 0.020 | 0.640 +/- 0.020 | 0.521 +/- 0.009 |
| pe_mean | 0.634 +/- 0.028 | 0.601 +/- 0.038 | 0.539 +/- 0.020 |
| pe_rel | 0.604 +/- 0.031 | 0.600 +/- 0.026 | 0.504 +/- 0.011 |
| oracle_pe | 0.931 +/- 0.006 | 0.439 +/- 0.035 | 0.926 +/- 0.007 |

## Invariants found

| Seed | World | Kept | R2 vs true energy | R2 vs true count | Validation slowness of the first six candidates |
|---|---|---|---|---|---|
| 1 | specular | 4 | 1.0000 | 1.0000 | 0.000, 0.000, 0.000, 0.000, 0.395, 0.761 |
| 1 | rough | 4 | 1.0000 | 1.0000 | 0.000, 0.000, 0.000, 0.000, 0.926, 1.097 |
| 2 | specular | 4 | 1.0000 | 1.0000 | 0.000, 0.000, 0.000, 0.000, 0.891, 0.921 |
| 2 | rough | 4 | 1.0000 | 1.0000 | 0.000, 0.000, 0.000, 0.000, 0.810, 0.795 |
| 3 | specular | 4 | 1.0000 | 1.0000 | 0.000, 0.000, 0.000, 0.000, 0.698, 1.025 |
| 3 | rough | 4 | 1.0000 | 1.0000 | 0.000, 0.000, 0.000, 0.000, 0.584, 0.767 |
| 4 | specular | 4 | 1.0000 | 1.0000 | 0.000, 0.000, 0.000, 0.000, 0.526, 1.055 |
| 4 | rough | 4 | 1.0000 | 1.0000 | 0.000, 0.000, 0.000, 0.000, 0.672, 0.796 |
| 5 | specular | 4 | 1.0000 | 1.0000 | 0.000, 0.000, 0.000, 0.000, 0.906, 0.770 |
| 5 | rough | 4 | 1.0000 | 1.0000 | 0.000, 0.000, 0.000, 0.000, 0.890, 0.786 |

Slowness is 0 for a perfectly conserved quantity and about 2 for one that is unrelated at long lag. R2 uses the simulator's ground truth and is a diagnostic only; nothing is fitted to it.
