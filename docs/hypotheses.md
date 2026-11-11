# Hypotheses and the go rule

Written 2026-11-11, before any confirmatory run. Changes go at the bottom, dated, with the reason.

## The question

Can a latent world model tell an impossible event from an improbable one?

- **Possible:** an ordinary lawful clip.
- **Improbable:** lawful, but unlike training. One ball three times faster than any in training, or a rough-wall bounce three to four standard deviations off the usual angle.
- **Impossible:** a law is broken at one moment. Four families:

| Family | What breaks | Is a conserved quantity broken? |
|---|---|---|
| energy | a ball changes speed in mid-flight | yes, energy |
| permanence | a ball vanishes | yes, count and energy |
| stochastic | a ball gains speed at a rough wall | yes, energy, hidden among random bounces |
| continuity | a ball teleports | no. Everything conserved stays conserved. |

The number that matters is the AUROC for **impossible vs improbable**. 0.5 is chance. Below 0.5 means the score ranks lawful rare clips above impossible ones.

## Hypotheses

**H1. Prediction error conflates the two.** For the best of the prediction-error scores (`pe_max`, `pe_mean`, `pe_rel`, chosen per family after the fact, in the baseline's favour), AUROC impossible vs improbable is at most 0.70.

**H2. Learned invariants separate them.** The level-shift score `inv` beats that best prediction-error score by at least 0.15 AUROC, as a mean over seeds, and by more than 0 in every seed.

**H3. Invariants stabilise rollouts.** Projecting a model's rollout onto the level sets of its invariants lowers long-horizon error. Not implemented yet. Shown for DreamerV3 on a pendulum in arXiv 2608.23526.

**H4. Real invariants replicate across encoders.** Invariants found in two independently trained encoders of one world agree (canonical correlation at least 0.9), and keeping only those lowers false alarms on improbable clips. Motivated by arXiv 2610.39302, whose theory does not cover action-conditioned predictors. `src/conserve/agreement.py` holds the measure; the filter is not wired in yet.

**Expected failure.** `inv` is at chance in the continuity family, since a teleport breaks no conserved quantity. This is reported, not counted against H2.

## Go rule

H2 must hold in at least two of the three conservation families (energy, permanence, stochastic). `src/conserve/runner.py` applies this and prints GO or NO-GO.

| Stage | Latents | What the gate decides |
|---|---|---|
| A | ideal frame latents | Sanity only. A pass is necessary and proves little: the invariants are low-degree by construction. |
| B | ideal latents, bent and noised | No gate. Maps what a latent must be like for the method to work. |
| C | a pixel JEPA's own latents | **The first real gate.** Read it only if the linear probe R2 is at least 0.9; otherwise fix the encoder first. |
| D | LeWorldModel, V-JEPA 2 | The paper's main result. |

If Stage C fails with a good encoder: try the declared secondary options below, once each, and report all of them. If it still fails, H2 is abandoned. The paper is then the three-way benchmark, H1, and the analysis of why low-degree invariants are absent from learned latents.

## Primary and secondary

- Primary score: `inv`, with `span = 1`.
- Secondary, declared in advance and always reported as secondary: `inv_rel`; `span` in {2, 4, 8}; `pca_dim` in {8, 16, 32}.
- `oracle_pe` is the true simulator's prediction error. It is a ceiling for what prediction error can do, not a competitor.

## Disclosures

- Fixed during exploratory single-seed runs (seed 0) on synthetic latents on 2026-11-11, before any confirmatory run: event threshold 0.25 of the median step, window 12, scale floor 0.05, lag range 30 to 90, and the selection rule (long-lag autocorrelation at least 0.9).
- The selection rule was loosened once during exploration, from slowness 0.05 to autocorrelation 0.9 (slowness 0.2), after a bent latent's energy invariant scored 0.069 and was dropped.
- `inv_rel` and `span` were added after seeing `inv` fail under strong nonlinearity and under noise. That is why they are secondary.
- Scenario parameters (speed scale 3, energy factors 0.5 and 1.6, gain 1.6, rough-wall jitter 12 degrees) were chosen once and not tuned against any score. Sensitivity to them should be reported.

## Changes

**2026-11-11, after the full-text reading (`docs/novelty.md`).** No hypothesis, threshold or gate changed. Three things were added:

- The learner is not presented as new. Bao (arXiv 2608.23526) uses the same criterion. H2 is a claim about what the invariants can be used for.
- H2 carries more risk than first assumed: the identifiability paper's own pixel experiment (arXiv 2605.26379) finds a linear read-out of the state below R2 0.5 on real trajectories.
- Candidate H5, not yet a gated hypothesis: in worlds driven by actions and losing energy, the hidden per-episode constants (mass, drag, stiffness) play the role of conserved quantities. A constant that changes mid-episode is the impossible event; a rare value is the improbable one.
