# World models on one GPU: literature map and experiment proposals

Date: 2026-11-11

**How this was made.** About 45 papers, mostly 2026 arXiv preprints, found by web search and read at abstract level (not full text). Results quoted below are the authors' own claims. "Not found" means not found in the targeted searches run for this pass, not that the work does not exist. Reading the closest papers in full is the first task before any code is written.

## 1. Where the field is (November 2026)

### 1.1 Small JEPA world models: the base everything builds on

| Paper | What it gives |
|---|---|
| LeWorldModel (LeWM), arXiv 2603.19312 | End-to-end JEPA from pixels. Loss = next-embedding prediction + SIGReg (Gaussian regularizer). About 15M parameters, single GPU, a few hours. Plans up to 48x faster than DINO-WM. |
| stable-worldmodel, arXiv 2605.21800 | Open platform: data layer, tested baselines, planners, environments with controllable visual, geometric and physical factors. |
| When Does LeJEPA Learn a World Model?, arXiv 2605.26379 (NeurIPS 2026) | Latents are linearly identifiable if and only if they are Gaussian, for worlds with stationary additive-noise transitions. Includes a non-Gaussian sweep. Circular and discrete latents are not analysed. |
| A Generalization Theory for JEPA-Based World Models, arXiv 2606.27014 | Spectral view of JEPA pretraining; bound linking pretraining error to planning regret. |
| The SIGReg Objective as Variational Free Energy, arXiv 2607.13612 | Theory only; empirical test left open. |

### 1.2 Documented failure modes and hand-designed fixes (crowded)

| Paper | Failure it names | Fix |
|---|---|---|
| Spectral-Target Physical Latent Structuring, arXiv 2609.04264 | "Physical representation laziness": latents avoid collapse but miss key physical properties | Fourier auxiliary head |
| PhyLatent, arXiv 2608.05720 | Three collapse modes | Three training pathways; +12.0 points MPC success on OGBench-Cube |
| PSG-JEPA, arXiv 2608.06799 | Forward prediction does not enforce physical-state identifiability | Grounding in proprioception |
| Temporally Centered SIGReg, arXiv 2607.26924 | Variance biased toward temporally persistent components | SIGReg on centered residuals; LIBERO average success 63.6% to 83.8% |
| AC-MTM ("No Gaussian Required"), arXiv 2608.17542 | Gaussian target imposed regardless of environment | Action-contrastive inverse dynamics; OGBench Visual Scene 80% vs 58% for SIGReg |
| Semigroup-JEPA, arXiv 2609.10464 | Poor out-of-distribution physics | Conditioning on the physics parameter plus rollout training |

Other regularizers seen in passing: lambda-JEPA (2609.35288), UR-JEPA (2606.01443), SD-JEPA (2605.31111).

**Takeaway.** At least six groups each propose a different auxiliary objective, each tested on its own tasks. No like-for-like comparison was found.

### 1.3 Extensions already taken

- **Hierarchy.** H-JEPA, arXiv 2610.06805 (Visual AntMaze success 18% to 73%). Hi-LeWM, arXiv 2607.12547 ("Hierarchy does not automatically improve performance").
- **Stochastic futures.** VJEPA, arXiv 2601.14354. Branch-JEPA, arXiv 2607.05238.
- **Test-time adaptation.** AdaJEPA, arXiv 2606.32026. JEPA-TTT, arXiv 2610.00722 (8 dynamics shifts in 4 environments). ALRA, NeurIPS 2026 workshop (210-parameter action residual). Counterfactual World Models, preprint reported by The Register on 9 Nov 2026.
- **Language side.** NextLat, arXiv 2511.05963. From Tokens to States, arXiv 2606.28127 (position paper).
- **World models for LLM agents.** WMRL, arXiv 2608.12564. MLE-World, arXiv 2610.37175 (confidence-gated execution). Crowded and well resourced.

### 1.4 Conservation and invariants inside learned world models

| Paper | Finding |
|---|---|
| Correcting a learned physical invariant improves world-model rollouts, arXiv 2608.23526 | A frozen DreamerV3 trained on pendulum video holds an energy-like invariant, found without labels. It drifts during rollouts. Projecting back to its level set reduces rollout error in all three conservative models. |
| Conservation Buys Stability and Factoring Buys Counterfactuals, arXiv 2609.19674 | A learned energy with a symplectic integrator keeps rollouts bounded for up to 100x the training horizon (three-body system, also from pixels). |
| When Do Conservation Laws Survive Learned Representations?, arXiv 2606.24945 | "Certified horizons"; a soft learned invariant survives in pixel settings. |
| DreamSAC, arXiv 2603.07545 (CVPR 2026) | Hamiltonian world model with symmetry-driven exploration. |
| Noether Networks, NeurIPS 2021 | Meta-learned conservation losses; modest gains. |

None of these uses an invariant to detect impossible events (checked against each abstract).

### 1.5 Surprise, probing and evaluation

- **When Does Latent Prediction Error Support Physical Surprise Claims?, arXiv 2611.01815.** In Kubric, impossible events can give lower prediction error than possible ones, and a fixed threshold flags many possible events as impossible. Compares possible vs impossible only.
- **Probing Where Surprise Happens, NeurIPS 2026 workshop.** Global and localized violation-of-expectation judgments diverge.
- **LeWM's own test.** Teleporting an object vs changing its colour.
- **What Can Latent World Models Know?, arXiv 2607.27017.** Decodable is not the same as used: drag is weakly readable from latents yet forecasts depend on it.
- **World Model Interpretability, arXiv 2610.25296.** "causal verification, and intervention remains largely unexplored".
- **The Observer Effect, arXiv 2602.12218.** Use low-capacity linear probes on frozen latents; adaptation-based probing misleads.
- **Benchmarks and positions.** ACWM-Phys (2605.08567), SmallWorlds (2511.23465), decision-centric evaluation (2606.15032).
- **Do sequence models learn laws?** Vafa et al., arXiv 2507.06952 (ICML 2025). From Kepler to Newton, arXiv 2602.06923 (ICML 2026).

### 1.6 Many worlds and in-context learning

- **Context and Diversity Matter, arXiv 2509.22353 (ICLR 2026).** In-context learning in world models needs long context and diverse environments. A search summary reports 1 to 4 environments giving little in-context learning and about 8K giving much more; not confirmed from the abstract.
- **Scaling in Games, arXiv 2610.28698.** Scaling law over data, diversity and model size, at 0.8B to 27B parameters.
- **OMNI-EPIC, ICLR 2025.** LLM-written environments for RL agents.

No hit for LLM-written simulators with different laws used to train a latent world model.

### 1.7 Automated research

- **Working systems.** autoresearch (5-minute runs, about 12 per hour on one GPU), AI Scientist-v2 (ICLR 2025 workshop paper, mean score 6.33), AlphaEvolve, MLEvolve (2606.06473), SwarmResearch (2607.02807).
- **Cautions.** The Little Scientist, arXiv 2608.16951: seven of ten agent-loop designs failed to beat simple sampling from the same model, and the full scaffold did worse than blind sampling at 7 to 8x the token cost. Automated Discovery Has No Universally Superior Harness, arXiv 2607.18235. Evolution or Illusion?, arXiv 2609.19799.

No hit for automated discovery of a JEPA or latent world-model objective.

## 2. What looks open

| Gap | Evidence it is open |
|---|---|
| G1. Telling the impossible from the merely improbable | Surprise benchmarks are two-way (possible vs impossible). No three-way design found. |
| G2. Invariants as a detection signal | Invariants are used for rollout stability, never for detecting impossible events. Not tested in JEPA latents. |
| G3. Fair comparison of the 2026 objectives | Each fix is tested on its own tasks. |
| G4. Diversity of laws, not just parameters | Adaptation work uses a handful of hand-made shifts. |
| G5. Machine-discovered world-model objectives | Discovery frameworks exist; none applied here. |

## 3. Proposals

### P1. Conservation as a teacher (recommended)

Prediction learns what changes. Add learning what never changes, and use it to separate the impossible (an invariant breaks) from the improbable (rare, but every invariant holds).

- **H1, the floor.** On a three-way benchmark (possible, improbable but lawful, impossible), prediction-error surprise does not separate improbable from impossible for LeWM, DINO-WM and V-JEPA 2.
- **H2, the innovation.** Linear and quadratic invariant heads, trained without labels on ordinary trajectories of frozen latents, flag impossible events and stay quiet on improbable ones. Rationale: if latents are close to linearly identifiable (2605.26379), physical invariants are low-degree functions of the latent, which extrapolate better than a generic predictor.
- **H3, the bonus.** Projecting LeWM rollouts onto invariant level sets reduces drift and improves planning (shown for DreamerV3 on a pendulum in 2608.23526).
- **Pilot.** Three scenario families, three conditions each, about 200 clips per cell. Compare AUROC for impossible vs improbable under prediction error and under the invariant score. Proposed go rule: the invariant score gains at least 0.15 AUROC in two of three families across three seeds.
- **If H2 fails.** The benchmark and the negative result on prediction-error surprise still stand, extending 2611.01815 with the missing control.
- **Compute.** Frozen encoders and cached embeddings; heads train in minutes.
- **Risks.** Invariants may capture appearance only. Driven, dissipative environments have few exact invariants, so bounds and monotone quantities may be needed. Invariants may not extrapolate to tail states.

### P2. Many universes

Agents write N small simulators with different laws. One context-conditioned latent world model trains on all of them and is tested on held-out laws with no gradient steps.

- **Hypothesis.** At fixed total data, in-context error on held-out laws falls as N grows, and matches gradient-based test-time training on short budgets.
- **Floor.** A validated, open zoo of worlds and the scaling measurement, even if the curve is flat.
- **Novelty risk.** 2509.22353 already shows diversity and context are needed. The delta is law-level diversity from written simulators, a small latent JEPA, and a direct comparison with test-time training. Read it in full first.
- **Compute.** One to two GPU-weeks; heavy agent and engineering work.

### P3. Evolved objectives

An outer loop (the agent swarm, acting as evolution) searches self-supervised loss terms seeded with physical principles: continuity, conservation, time reversal, locality, time composition. An inner loop trains small world models. Selection is on held-out worlds.

- **Floor.** The first like-for-like comparison of six to eight published 2026 objectives on stable-worldmodel (gap G3).
- **Risk.** Highest of the three. Gains may sit below seed noise, and agent scaffolds may not beat simple sampling (2608.16951). Requires a sampling baseline, repeated seeds and held-out worlds.
- **Compute.** Heaviest: every candidate is an end-to-end training run. Two to three GPU-weeks for the floor alone.

**Suggested order.** P1, then the floor of P3 (shared infrastructure), then P2.

## 4. Notes

- ICML 2027: one third-party site projects a full-paper deadline of 22 Jan 2027; not official.
- Two abstracts could not be fetched and rest on search summaries: AD-WM (2609.30264) and MLE-World (2610.37175).
- The Counterfactual World Models preprint was seen only through press coverage.

## 5. Added after the first pass

- **Proving and Improving the Platonic Representation Hypothesis with JEPAs, arXiv 2610.39302 (Balestriero and LeCun, 29 Oct 2026).** Proves that JEPAs trained separately on different modalities learn the same representation up to one orthogonal map, for a finite world with injective renderings, whitened embeddings, a spectral gap and shared positive pairs, at the population optimum. Reports that "good views" need 42x fewer samples than next-token prediction for the same alignment. The analysis does not cover action-conditioned predictors. Code: github.com/galilai-group/prh-proof. Use here: a rotation keeps a low-degree invariant low-degree, so a real conserved quantity should appear in every good encoder of the same world. That gives hypothesis H4 (cross-encoder agreement as a filter for invariants) and a possible route to move invariants between modalities.

## 6. After the full-text reading

See `docs/novelty.md` for the verdict and `docs/reading-list.md` for what each paper says. In short: the three-way test still looks open; the invariant learner has a close precedent (arXiv 2608.23526) and classical roots; using invariants as a detector in a world model's latent was not found.
