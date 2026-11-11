# Stage D plan: published world models

Written 2026-11-11, after Stages A to C. Not started. One page by design; the details that are still open are marked as such.

## Question

Do the latents of published world models carry low-degree invariants that tell an impossible event from an improbable one, where their own prediction error does not? Stage B says what such a latent must be like: close to free of frame-to-frame noise and only mildly nonlinear in the state. Stage C says whether a small JEPA we trained ourselves gets there (see the paper and `results/pilot_pixels/`). Stage D asks the same of models we did not train.

## Models, in order

1. **LeWorldModel** (`github.com/lucas-maes/le-wm`, MIT). ViT-tiny encoder, 192-dim latent, SIGReg (our re-implementation now matches its `SIGReg` class, `experiments/check_sigreg.py`). Pretrained checkpoints on Hugging Face (`quentinll/lewm-pusht`, `-cube`, `-tworooms`, `-reacher`). Downloading them needs the owner's go-ahead (kickoff rule).
2. **V-JEPA 2** (inference on the A40) on our rendered billiards clips and on IntPhys 2. No training.
3. Only if 1 and 2 give a clean answer: a second LeWM training run with a different seed, for H4.

## What has to be built

- **An adapter** per model returning, for one clip, the per-frame embeddings and the model's own per-step prediction errors. `pipeline.fit_latents` and `pipeline.score_latent` then work unchanged.
- **Three-way scenarios in the models' own worlds.** PushT, TwoRoom, Cube and Reacher are driven by actions and lose energy, so energy is not conserved. Candidate conserved quantities, decided on paper before any code:
  - hidden per-episode constants (mass, friction, stiffness): impossible = the constant changes mid-episode; improbable = a rare but fixed value (the PokeWorld idea, arXiv 2607.27017);
  - object count and identity: impossible = an object vanishes or swaps shape; improbable = an unusual but constant object.
  Each needs a test that the lawful clips keep the quantity to numerical precision and the impossible clips change it at exactly one step, as in `tests/test_worlds.py`.
- **Action conditioning.** The invariant learner assumes a passive world. With actions, an invariant must be constant given the actions; the simplest version restricts to quantities the actions cannot change (the constants above). Open question: how to treat steps where an action makes contact.
- **Billiards through V-JEPA 2.** Our clips are 48 by 48 grayscale; render at 224 with the same scenarios.

## Order of work and gates

1. Adapter for LeWM, unit-tested on a short clip. CPU.
2. Linear probe of the LeWM latent on our gate targets (count and occupancy, `src/conserve/probes.py`) on PushT frames. If the probe is below 0.9, report it and stop with that model.
3. The noise test from Stage B on the real latent: the long-lag slowness of the best candidate on lawful clips. A latent with no candidate below the selection threshold cannot pass; report and stop.
4. Only then the three-way AUROCs, seeds 1 to 3, all families, with the go rule of `docs/hypotheses.md` applied as written.

## Compute and data

- LeWM inference and invariant fitting: under an hour per model and world on the A40. Training a LeWM from scratch: several hours per run; needs the owner's go-ahead.
- Data: the HDF5 sets from `quentinll/lewm` and the pretrained checkpoints are external downloads; both need the owner's go-ahead.

## What would change the plan

- If Stage C fails because the learned latent is noisy between frames (as Stage B predicts), Stage D should first measure that noise on LeWM and V-JEPA 2 before building scenarios. That is cheap and may settle the question without the scenario work.
- If a published model passes the probe and keeps candidates below the selection threshold, Stage D is the paper's main table.
