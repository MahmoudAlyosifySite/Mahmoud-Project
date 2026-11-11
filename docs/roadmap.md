# Roadmap

Each stage ends in something that can be written up whatever the next stage shows. Time estimates are guesses until the first run is timed.

## Stage 0. Read before building further

Done on 2026-11-11 for eight papers; answers are in `docs/reading-list.md` and the verdict in `docs/novelty.md`. None of them separates impossible from improbable, and none uses an invariant as a detector. The learner itself has a close precedent (arXiv 2608.23526) and is not claimed as new.

- [ ] Read arXiv 2607.03728 and IntPhys 2 in full.
- [ ] Read Shang et al. 2015 and state how our score differs from theirs.

## Stage A. Ideal latents (CPU, minutes)

```bash
python experiments/pilot_synthetic.py
```

- [ ] Run seeds 1 to 5 and commit the report and the ledger.
- [ ] Check the sanity conditions: `oracle_pe` is 1.0 in the deterministic families, and `inv` is near 0.5 for continuity.
- [ ] Sensitivity: rerun with the scenario parameters moved (speed scale 2 and 4; gain 1.2 and 2.0) and report how the AUROCs move.

Output: the benchmark design and the proof of concept.

## Stage B. Where it breaks (CPU, about an hour)

```bash
python experiments/robustness_sweep.py
```

- [ ] Run the default grid.
- [ ] Find the largest nonlinearity and noise at which an energy invariant is still found (R2 against true energy above 0.9).
- [ ] Decide whether `span` rescues the noisy cases.

Output: a statement of what a latent must be like for conservation to be readable from it. This also gives a label-free probe of any encoder: does a conserved low-degree quantity exist in its latent, yes or no.

## Stage C. A learned encoder (GPU)

```bash
python experiments/pilot_pixels.py --device cuda --steps 20000
```

- [ ] Time one seed on the A40 and record it here.
- [ ] Get the linear probe R2 to at least 0.9. Levers: `--steps`, `--lam`, `--dim`, frame size. If the latent collapses (`latent_std_min` near 0), raise `--lam`.
- [ ] Only then read the verdict. Apply the secondary options from `docs/hypotheses.md` once each if it fails.
- [ ] Train two seeds and measure `agreement.canonical_correlations` between their invariants (H4).
- [ ] Replace or verify the SIGReg re-implementation before quoting any number.

Output: the first evidence on whether the idea survives a learned latent. This is the gate for the rest.

## Stage D. Published world models

Facts below were read from the two READMEs on 2026-11-11. Check them again on the server; the READMEs disagree with each other in places.

**LeWorldModel** (`github.com/lucas-maes/le-wm`, MIT):

```bash
uv venv --python=3.10
```

```bash
source .venv/bin/activate
```

```bash
uv pip install stable-worldmodel[train,env]
```

- Training: `python train.py data=pusht` (Hydra configs under `config/train/`; set the WandB entity and project in `config/train/lewm.yaml` first).
- Data: HDF5 files from the Hugging Face collection `quentinll/lewm`, extracted with `tar --zstd -xvf`, placed under `$STABLEWM_HOME`.
- Pretrained checkpoints: `quentinll/lewm-pusht`, `lewm-cube`, `lewm-tworooms`, `lewm-reacher`. Loading goes through a conversion script in the README, then `swm.policy.AutoCostModel('pusht/lewm')`.
- Evaluation: `python eval.py --config-name=pusht.yaml policy=pusht/lewm`.
- Baseline checkpoints (PLDM, LeJEPA, DINO-WM and others) are in a Google Drive folder linked from the README.
- Unclear from the READMEs: the default `$STABLEWM_HOME` (one says `~/.stable-wm/`, the other `~/.stable_worldmodel/`), and what the `train` and `env` extras contain.

**stable-worldmodel** (`github.com/galilai-group/stable-worldmodel`, MIT): environments `swm/PushT-v1`, `swm/TwoRoom-v1`, `swm/OGBCube-v0`, `swm/ReacherDMControl-v0` and more; `pip install 'stable-worldmodel[all]'`.

What this stage needs from us:

- [ ] An adapter that returns, for one clip, the per-frame embeddings and the model's own prediction errors. With that, `pipeline.fit_latents` and `pipeline.score_latent` work unchanged.
- [ ] Three-way scenarios in those environments, which means editing simulator state mid-episode.
- [ ] An answer to the hard design question: these worlds are driven by actions and lose energy to friction, so energy is not conserved. The best candidate so far is the hidden per-episode constants (mass, drag, stiffness), as varied in PokeWorld (arXiv 2607.27017; release promised, not found yet). Object count and identity are the others. Work this out on paper before coding.
- [ ] Reuse the falling-sphere support scenes from `github.com/3-ml/jepa-surprise` (arXiv 2611.01815) as a second benchmark, and add an improbable condition to them.
- LeWorldModel facts from its paper: 192-dimensional latent, SIGReg weight 0.1 with 1024 projections, and (from a summary of the appendix) predictor history 3 frames at 224 by 224.
- [ ] V-JEPA 2 (inference fits on an A40) on our rendered clips and on IntPhys 2.

Output: the paper's main table.

## Stage E. Beyond detection

- H3: project rollouts onto invariant level sets and measure long-horizon error and planning success.
- A combined detector: invariants for conservation, prediction error for continuity.
- Transfer of invariants across modalities through one rotation (arXiv 2610.39302), for example learned on low-dimensional state and applied to pixels.

## The paper

Working title: *Impossible or Merely Improbable? Conservation Laws as a Test of Physical Understanding in Latent World Models.*

| If... | The paper is... |
|---|---|
| Stage C and D pass | benchmark, method, results on published models |
| Stage C passes, D mixed | benchmark, method, and a map of which models carry invariants |
| Stage C fails | benchmark, H1, and why learned latents hide what is conserved |

ICML 2027: one third-party site projects a full-paper deadline of 22 January 2027. It is not official. A workshop paper from Stages A to C is the realistic first target.
