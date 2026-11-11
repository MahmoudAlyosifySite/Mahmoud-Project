# What is new here and what is not

Written 2026-11-11 after reading the closest papers in full. How much of each paper was read word for word is recorded in `docs/reading-list.md`. "Not found" means not found in what was read and searched.

## Verdict

| Piece | New? | Basis |
|---|---|---|
| A three-way test for physical world models: possible, improbable, impossible | **Looks open** | arXiv 2611.01815, read in full, has no rare-but-lawful condition. LeWorldModel compares a teleport with a colour change. IntPhys 2 pairs possible with impossible. |
| The impossible versus improbable distinction itself | Not new | Developmental psychology. For language models: Kauf et al. (2212.01488), Michaelov et al. (2506.06808), "Shades of Zero" (2502.20469). |
| The invariant learner | **Not new** | Bao (2608.23526) uses the same criterion on a DreamerV3 latent. Wang (2606.24945) trains a "soft learned invariant" with an anti-collapse term. Slow Feature Analysis (2002). Koopman eigenfunctions (Kaiser, Kutz, Brunton, 1811.00961). |
| "An invariant moved, so something is wrong" as a way to detect anomalies | Not new as an idea | Process monitoring: Shang et al. (AIChE Journal, 2015) use slow features to tell a normal change of operating point from a real fault. Cyber-physical security uses invariant violations as alarms. |
| Learned invariants used to detect unlawful events in a world model's latent | **Not found** | Bao, Wang et al. (2609.19674) and Wang (2606.24945) use invariants for rollout correction, stability and certification. None uses them to detect anything. |
| Invariants in a world with several objects and contacts | Named as open by Bao | His study covers one pendulum, and he calls contacts, several interacting objects and actions the main open question. |

## How to pitch the paper

Lead with the benchmark and what it shows about prediction error (H1). Present the invariant score as a simple tool assembled from known parts that passes the test where prediction error fails (H2). Do not call it a new learning method.

Two things can be claimed for the method: it handles events, so it works with contacts and several objects; and it is used as a detector, which the earlier invariant papers do not do.

## What the reading changed

1. **The risk to H2 went up.** The identifiability paper's own pixel experiment (2605.26379, Section 6.2) reports that with real trajectories the linear read-out of the state stays below R2 0.5, because several of the theorem's assumptions fail at once. LeWorldModel's linear probes on PushT are high (Pearson r 0.90 to 0.99) but reported as poor on OGBench-Cube. Whether a learned latent is linear enough depends on the world. Stage C decides.

2. **There is a way through Stage D.** Worlds driven by actions and losing energy to friction have no conserved energy. They do have hidden per-episode constants: mass, drag, stiffness. Those are invariants in our sense. PokeWorld (2607.27017) varies exactly these three and its authors name models with explicit episode-level latents as the next test. "Impossible" becomes a constant that changes mid-episode; "improbable" becomes a rare value of it. Their release is promised and was not found yet.

3. **Word the claim carefully.** A read-out from the latent shows the representation carries the information. It does not show the model uses it. 2607.27017 separates readable from used, and a search summary of 2607.03728 (unread) says a conserved quantity can be decodable yet have no effect on predictions. Say "the representation makes lawfulness readable", not "the model understands conservation".

4. **A second benchmark is available.** 2611.01815 is diagnostic only: it proposes no replacement for prediction error. Its code and scene recipes are public (`github.com/3-ml/jepa-surprise`), including falling-sphere support scenes in Kubric. Its authors note that ordinary difficulty already raises error on lawful clips, and they do not vary rarity. That is the gap the improbable condition fills.

5. **Facts for Stage D.** LeWorldModel: ViT-tiny encoder, 192-dimensional latent, SIGReg weight 0.1 with 1024 projections, and (from a summary of its appendix) a predictor history of 3 frames at 224 by 224. In its own test the teleport is detected and the colour change is not significant.

## Must be cited

- Bao, 2608.23526, as the closest method.
- Wang, 2606.24945, and Wang et al., 2609.19674, for learned invariants in latent world models.
- Wiskott and Sejnowski, 2002; Kaiser, Kutz and Brunton, 2018, as the classical roots.
- Shang et al., 2015, for the same separation in process monitoring.
- Hristov et al., 2611.01815, as the motivation and the nearest benchmark.
- Kauf et al., Michaelov et al., for the distinction in language models.
- Klindt et al., 2605.26379, and Balestriero and LeCun, 2610.39302, for when latents are linear in the state and agree across encoders.

## Still unread

- 2607.03728, "The Objective Decides": known only from a search summary. Read before writing the claims section.
- IntPhys 2 (2506.09849) full text, for any analysis of rarity.
- Shang et al. 2015 and Kaiser et al. 2018 full texts.
- The later sections of 2609.19674 and 2606.24945, read only through a summary.
