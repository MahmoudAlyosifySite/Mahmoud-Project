# Reading list

Answers from the full-text pass of 2026-11-11. Each entry says how much was read word for word. Anything marked "summary" came from an automatic summary of the page and should be checked before it is relied on. The conclusions are in `docs/novelty.md`.

## Read

### 1. When Does Latent Prediction Error Support Physical Surprise Claims in Video World Models? (arXiv 2611.01815)

Coverage: every section and appendix, word for word.

- **Any rare-but-lawful control?** No. Conditions are possible versus impossible, plus appearance changes (hue, brightness, recolouring), static frames, and action changes (zero, reversed, scaled). The authors note that moving lawful targets already have higher error than static ones, and that their interventions are not equally far from the training distribution. They report this as a caveat and do not vary it.
- **Kubric scenes:** a sphere falls onto a platform that is solid or has a gap. Stopping over a gap is unsupported; falling through a solid platform breaks solidity. 256 configurations, 12 clips each, 3,072 clips. The model gives stopping higher error in both settings, so the impossible-minus-possible margin is negative on solid support and positive over the gap.
- **What they propose:** nothing to replace prediction error. Six diagnostic checks.
- **Conservation, invariants, rarity:** none of these words appear in the relevant sense.
- **Models:** V-JEPA 2 (G and L), VideoMAE v2, LeWorldModel, DINO-WM, JEPA-WM.
- **Code:** `github.com/3-ml/jepa-surprise`. Videos and weights are obtained separately.

### 2. Correcting a learned physical invariant improves world-model rollouts (arXiv 2608.23526)

Coverage: every section and appendix, word for word. Equations were garbled in extraction.

- **How the invariant is found:** take the top 12 principal components of a frozen DreamerV3's recurrent state, form degree-4 polynomials (1,819 monomials), and minimise within-trajectory variance over total variance. That is a generalised eigenproblem; the total-variance denominator rules out the constant. A second criterion, alignment with the model's own flow, picks among the top eight candidates. No labels.
- **Used for detection?** No. Only to correct rollouts and to compare models.
- **System:** one pendulum from 64 by 64 video, zero actions, three conservative seeds and three damped ones. Damped models show no comparable invariant.
- **Effect:** projecting onto the level set lowers rollout error by 2.9 to 3.5 percent on the conservative models. Random constraints raise it by a median of 54 percent.
- **Agreement across seeds:** reported as ranges. The representations are not aligned to each other.
- **Open, in the author's words:** contacts, several interacting objects and actions.
- **Code:** `github.com/Zarand3r/world-model-invariants`.

This is the closest method to ours. Our learner is the same criterion with degree 2, long-lag pairs, and event handling.

### 3. Conservation Buys Stability and Factoring Buys Counterfactuals (arXiv 2609.19674)

Coverage: abstract and Sections 1 to 4.3 word for word; the rest by summary.

- The energy is a learned Hamiltonian evolved by a symplectic update, so conservation is built into the architecture. Trained on one-step prediction.
- No use for detection in what was read. Energy drift is a stability diagnostic.
- The authors state that conservation is the wrong prior once a system really loses energy.

### 4. When Do Conservation Laws Survive Learned Representations? (arXiv 2606.24945)

Coverage: abstract and Sections 1 to 5.1 word for word; the rest by summary.

- **Certified horizon:** the largest number of steps for which a rollout provably stays within a tolerance of an invariant's level set. Computed from the tolerance, the invariant's gradient near the level set, the initial embedding error, the per-step drift, and the decoder error.
- **Soft learned invariant:** a small head on the latent, trained with an invariance penalty along the model's rollout, a temporal invariance penalty on real trajectories, and a variance term against the constant solution. Self-supervised.
- No use for detection in what was read. The framing is when a model should stop trusting its own rollout.
- **Kepler boundary:** the invariant stiffens near the closest approach, and the initial embedding error dominates whatever the capacity.

### 5. When Does LeJEPA Learn a World Model? (arXiv 2605.26379)

Coverage: main text and several appendices word for word; Appendix H (detailed numbers) by summary.

- **Theorem:** the embedding equals the world's latent up to an orthogonal map, if and only if the latent is Gaussian. Assumptions: independent latent coordinates, stationarity, a linear Ornstein-Uhlenbeck transition with additive Gaussian noise, the regulariser assumed to have fully succeeded, no predictor, and the population optimum. Nothing is assumed about the observation function.
- **Velocity:** not analysed as a two-frame quantity.
- **Non-Gaussian latents:** linear recovery peaks at the Gaussian. Circular and discrete latents are not analysed and are listed as open.
- **Pixel experiment (two-joint arm):** with samples drawn to match the theory, R2 is 0.95. With real trajectories, one joint reaches 0.71, the other about 0, and the total stays below 0.5. The paper names the violated assumptions: non-Gaussian marginals, different persistence per coordinate, and joint-limit wrapping.
- **Energy, invariants, conserved quantities:** not discussed.
- **Open, per the authors:** identifiability of the action-conditioned transition.
- **Code:** `github.com/klindtlab/lejepa-identifiability`.

### 6. LeWorldModel (arXiv 2603.19312)

Coverage: main text and Appendices A and B word for word; Appendices C to I by summary.

- ViT-tiny encoder (about 5M parameters), 192-dimensional latent, transformer predictor (about 10M). SIGReg weight 0.1 with 1024 projections.
- From the summary of the appendix: 224 by 224 frames, frame skip 5, batch 128, 10 epochs, predictor history 3 frames on PushT and Cube.
- **Its surprise test:** a teleport and a colour change, in three environments. The teleport raises surprise significantly; the colour change does not. No rare-but-lawful control.
- **Linear probes on PushT:** Pearson r 0.97 for agent position, 0.99 for block position, 0.90 for block angle.

### 7. What Can Latent World Models Know? (arXiv 2607.27017)

Coverage: main text and Appendices A to D word for word.

- **PokeWorld:** a custom 2D simulator. Mass, drag and contact stiffness vary per episode. Code and data are promised; no release URL in the text.
- **Readable versus used:** readable is a ridge probe's R2. Used is a slope ratio: how the forecast changes with the parameter, divided by how the true outcome changes with it.
- A JEPA-style model of about five million parameters, trained in about seven minutes on one GPU.
- The words conserved, invariant, anomaly, impossible and surprise do not appear.
- The authors name models with explicit episode-level latent variables as the next test.

### 8. Proving and Improving the Platonic Representation Hypothesis with JEPAs (arXiv 2610.39302)

Coverage: the first third word for word (theory and first experiments); the rest by summary.

- Separately trained JEPAs agree up to one orthogonal map, for a finite world, injective renderings, whitened embeddings, a spectral gap and shared positive pairs, at the population optimum.
- Action-conditioned predictors are outside the analysis. The uniqueness result does not cover SIGReg's full constraint.
- **Code:** `github.com/galilai-group/prh-proof`.

### 9. The Objective Decides: When a Learned Dynamics Model Uses a Conserved Quantity (arXiv 2607.03728)

Liao and Cao, UNSW, v1 4 July 2026. Coverage, 2026-11-11: the HTML word for word from the start through most of Appendix B (about 80 percent); Appendices B (end) to F only through a summary.

- Systems: pendulum, LC circuit, two-body spring, central force, 1-D wave, Kepler; MLP, GRU and Transformer predictors; and Poseidon-B (158M parameters) on Navier-Stokes.
- "Decodable" is a cross-validated ridge probe with R2 near 1. "Used" is a single-step activation interchange: overwrite the probe direction with a donor's value and measure the correlation tau between the change of the read-out and the change of the invariant.
- Main claim (abstract): invariants are "linearly decodable at R2≈1 yet causally inert on next-state prediction", and the same direction becomes load-bearing once the objective rewards the invariant. Pendulum: decode R2 1.00, next-state tau -0.01, invariant-target tau +0.97.
- Not found in what was read: any use of an invariant to detect a violation, an anomaly or a surprise; any rare-but-lawful condition. Their out-of-distribution result predicts generalisation from the gap between objectives; it is not a detector.
- For us: supports wording the claim as "the representation makes lawfulness readable", not "the model uses conservation". Our score reads the latent; it does not test use.

### 10. IntPhys 2 (arXiv 2506.09849)

Bordes, Garrido, Kao, Williams, Rabbat, Dupoux (FAIR at Meta), v1 11 June 2025. Coverage, 2026-11-11: the full HTML text, Sections 1 to 6 and Appendix A, word for word; figures as text only.

- Four videos per scene, two possible and two impossible; the possible video of one scenario is the impossible one of another. Principles: permanence, immutability, spatio-temporal continuity, solidity. 1,416 videos.
- Scoring: surprise from prediction error over windows, paired and single-video protocols. Predictive models are near chance (V-JEPA 2 overall 57.5 percent; humans 96.4).
- Not found in the full text: any lawful but rare or out-of-distribution condition. The Hard split adds visual clutter, which is difficulty, not rarity. Section 3 names "the general prediction difficulty of the video" as a confound for single-video scoring but adds no control for it.
- No mention of conservation, energy, momentum or invariants.

### 11. Kaiser, Kutz and Brunton, Discovering conservation laws from data for control (arXiv 1811.00961)

Coverage, 2026-11-11: abstract and Sections I to V word for word in the HTML version; VI-B onwards through a summary.

- Conserved quantities are the null space of a Koopman-generator regression on a function library (eigenvalue zero), found by SVD; a second method uses a Lie-Poisson bracket. One example, the rigid body (finds angular momentum and energy).
- Used as intrinsic coordinates for control (MPC on the conserved quantity). Not used to detect violations in what was read.
- Same family of learner as ours (a library, a null-space or eigen-problem); ours uses long-lag differences between events and a generalised eigenproblem with a variance constraint.

## Prior art found by search, not read in full

- **Shang et al., AIChE Journal 61:3666 to 3682, 2015.** Slow Feature Analysis for process monitoring, with separate indices for a deviation of the operating point and for an anomaly in the dynamics, so that a normal change of operating point is not flagged as a fault. The same separation as ours, in another field. **Still not read: on 2026-11-11 the publisher's page returned 403 and no open copy was reached.** Describe it in the paper only as above, from the search summary, and say so.
- **Kauf et al., arXiv 2212.01488; Michaelov et al., arXiv 2506.06808; "Shades of Zero", arXiv 2502.20469.** Language models separate impossible from improbable events poorly.

## Cited from memory, still to verify

- Wiskott and Sejnowski, Slow Feature Analysis, Neural Computation 2002.
- Alet et al., Noether Networks, NeurIPS 2021.
- Zhang et al., ConCerNet, ICML 2023.
- Liu and Tegmark, AI Poincare, 2021.
- Greydanus et al., Hamiltonian Neural Networks, NeurIPS 2019.

## Next to read

1. Shang et al. 2015 (needs library access), to state exactly how our score differs from theirs.
2. The later sections of 2609.19674 and 2606.24945 (read only through a summary so far), and Appendices C to F of 2607.03728.
