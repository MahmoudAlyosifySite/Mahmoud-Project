# Pixel pilot (dim512_lam1_steps20000_pca16_span2)

Seeds: [1, 2, 3]

Config:

```json
{
  "families": [
    "energy",
    "permanence",
    "continuity",
    "stochastic"
  ],
  "model": {
    "batch": 256,
    "dim": 512,
    "lam": 1.0,
    "lr": 0.0003,
    "size": 48,
    "steps": 20000
  },
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
    "n_train": 400,
    "n_val": 100,
    "noise": 0.0,
    "pca_dim": 16,
    "predictor_iters": 150,
    "span": 2,
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

**NO-GO**: H2 holds in 0 of 3 conservation families (rule: at least 2).

| Family | Best prediction-error score | Its AUROC | Invariant AUROC | Gain | H1 | H2 |
|---|---|---|---|---|---|---|
| energy | pe_rel | 0.637 | 0.500 | -0.137 | yes | no |
| permanence | pe_max | 1.000 | 0.500 | -0.500 | no | no |
| stochastic | pe_mean | 0.728 | 0.500 | -0.228 | no | no |

AUROC here is impossible vs improbable: 0.5 is chance, 1.0 is perfect, below 0.5 means the score ranks lawful rare clips above impossible ones.

## All AUROCs (mean +/- std over seeds)

### energy

| Score | impossible vs possible | impossible vs improbable | improbable vs possible |
|---|---|---|---|
| inv | 0.500 +/- 0.000 | 0.500 +/- 0.000 | 0.500 +/- 0.000 |
| inv_rel | 0.500 +/- 0.000 | 0.500 +/- 0.000 | 0.500 +/- 0.000 |
| pe_max | 0.597 +/- 0.012 | 0.058 +/- 0.011 | 0.956 +/- 0.008 |
| pe_mean | 0.557 +/- 0.038 | 0.008 +/- 0.002 | 1.000 +/- 0.000 |
| pe_rel | 0.559 +/- 0.023 | 0.637 +/- 0.034 | 0.484 +/- 0.036 |
| oracle_pe | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.500 +/- 0.000 |

### permanence

| Score | impossible vs possible | impossible vs improbable | improbable vs possible |
|---|---|---|---|
| inv | 0.500 +/- 0.000 | 0.500 +/- 0.000 | 0.500 +/- 0.000 |
| inv_rel | 0.500 +/- 0.000 | 0.500 +/- 0.000 | 0.500 +/- 0.000 |
| pe_max | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.953 +/- 0.002 |
| pe_mean | 1.000 +/- 0.000 | 0.831 +/- 0.018 | 1.000 +/- 0.000 |
| pe_rel | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.483 +/- 0.030 |
| oracle_pe | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.500 +/- 0.000 |

### continuity

| Score | impossible vs possible | impossible vs improbable | improbable vs possible |
|---|---|---|---|
| inv | 0.500 +/- 0.000 | 0.500 +/- 0.000 | 0.500 +/- 0.000 |
| inv_rel | 0.500 +/- 0.000 | 0.500 +/- 0.000 | 0.500 +/- 0.000 |
| pe_max | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.959 +/- 0.001 |
| pe_mean | 1.000 +/- 0.000 | 0.921 +/- 0.006 | 1.000 +/- 0.000 |
| pe_rel | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.524 +/- 0.010 |
| oracle_pe | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.500 +/- 0.000 |

### stochastic

| Score | impossible vs possible | impossible vs improbable | improbable vs possible |
|---|---|---|---|
| inv | 0.500 +/- 0.000 | 0.500 +/- 0.000 | 0.500 +/- 0.000 |
| inv_rel | 0.500 +/- 0.000 | 0.500 +/- 0.000 | 0.500 +/- 0.000 |
| pe_max | 0.672 +/- 0.031 | 0.557 +/- 0.023 | 0.624 +/- 0.021 |
| pe_mean | 0.760 +/- 0.018 | 0.728 +/- 0.017 | 0.545 +/- 0.003 |
| pe_rel | 0.523 +/- 0.043 | 0.334 +/- 0.020 | 0.630 +/- 0.030 |
| oracle_pe | 0.930 +/- 0.005 | 0.442 +/- 0.044 | 0.923 +/- 0.009 |

## Invariants found

| Seed | World | Kept | R2 vs true energy | R2 vs true count | Validation slowness of the first six candidates |
|---|---|---|---|---|---|
| 1 | specular | 0 | 0.0000 | 0.0000 | 1.249, 1.226, 1.223, 1.501, 1.430, 1.457 |
| 1 | rough | 0 | 0.0000 | 0.0000 | 1.067, 1.754, 1.896, 1.709, 1.578, 1.960 |
| 2 | specular | 0 | 0.0000 | 0.0000 | 1.636, 1.610, 1.270, 1.378, 1.917, 1.788 |
| 2 | rough | 0 | 0.0000 | 0.0000 | 1.314, 1.670, 1.643, 1.747, 1.714, 1.885 |
| 3 | specular | 0 | 0.0000 | 0.0000 | 1.086, 1.157, 1.253, 1.430, 1.383, 1.548 |
| 3 | rough | 0 | 0.0000 | 0.0000 | 1.370, 1.578, 1.772, 1.595, 1.738, 1.872 |

Slowness is 0 for a perfectly conserved quantity and about 2 for one that is unrelated at long lag. R2 uses the simulator's ground truth and is a diagnostic only; nothing is fitted to it.

## Is the encoder any good?

Probe R2 (the Stage C gate, docs/hypotheses.md) is the mean of two linear read-outs from the latent on held-out lawful clips: the number of balls, and a 12 by 12 occupancy map of the frame (1.0 is perfect). If it is below 0.9, fix the model before reading the AUROCs.

- seed 1, specular: probe R2 0.509 (count 0.426, occupancy 0.592; slot-indexed, not gating, 0.080), latent std min/median 0.782/1.002
- seed 1, rough: probe R2 0.501 (count 0.413, occupancy 0.589; slot-indexed, not gating, 0.088), latent std min/median 0.858/1.004
- seed 2, specular: probe R2 0.471 (count 0.355, occupancy 0.587; slot-indexed, not gating, 0.105), latent std min/median 0.835/1.004
- seed 2, rough: probe R2 0.472 (count 0.345, occupancy 0.599; slot-indexed, not gating, 0.111), latent std min/median 0.827/1.004
- seed 3, specular: probe R2 0.526 (count 0.438, occupancy 0.614; slot-indexed, not gating, 0.076), latent std min/median 0.866/1.003
- seed 3, rough: probe R2 0.520 (count 0.422, occupancy 0.617; slot-indexed, not gating, 0.095), latent std min/median 0.837/1.002
