# Conservation as a teacher

World models learn what changes: they predict the next state. The physical world is defined just as much by what never changes: energy, the number of objects, what an object is. This project teaches a world model's latent space to find those invariants by itself, and uses them to answer a question prediction error cannot: **is this event impossible, or only improbable?**

- **Impossible:** a conserved quantity takes a new value. A ball vanishes, or speeds up with nothing pushing it.
- **Improbable:** rare, but every law still holds. A ball far faster than any seen in training, or a bounce at an unusual angle.

Prediction error is high for both. An invariant moves only for the first.

## Status

Built on 2026-11-11. The code runs and is tested. **No experiment in this repository has produced a result yet.** `results/` is empty and gets filled on the GPU server.

| Part | State |
|---|---|
| Billiards simulator, three-way scenarios, renderer | Tested |
| Invariant learning, level-shift score, metrics, ledger | Tested |
| Stage A and B scripts (synthetic latents, CPU) | Run end to end at toy size |
| Stage C script (pixel JEPA, GPU) | Runs end to end on CPU for 30 steps. Never trained properly. Untuned. |
| SIGReg-style regulariser | Re-implemented from the papers' description. Not checked against the official code. |
| Stage D (LeWorldModel, V-JEPA 2) | Not started. Notes in `docs/roadmap.md`. |
| Reading of the closest papers | Done for eight papers. Verdict in `docs/novelty.md`. |

What exploratory runs showed while the code was written (seed 0, small samples, ideal latents, settings changed since; **rerun before quoting**):

- With an ideal latent the invariants recover energy and ball count without labels (R2 above 0.999).
- They separated impossible from improbable clips perfectly where a conserved quantity was broken (AUROC 1.00), while the best prediction-error score was near or below chance (0.43 in the energy family, 0.62 with random bounces).
- They are blind to a teleport, which breaks continuity and conserves everything. Prediction error catches that one. The two signals are complementary.
- They degrade as the latent gets more nonlinear and they failed outright with frame-to-frame latent noise. Whether a learned encoder is good enough is the open question, and Stage C is the test.

## Layout

```
src/conserve/
  worlds/        billiards simulator, possible / improbable / impossible scenarios, renderer
  latents.py     ideal frame latent, and a mixer that bends it and adds noise
  events.py      which steps are smooth enough to trust
  invariants.py  the learner: a generalised eigenproblem on long-lag differences
  scores.py      level-shift score, prediction-error scores, oracle
  pipeline.py    fit on lawful clips, score any clip, compare
  runner.py      seeds, summary, the go rule
  agreement.py   do two encoders agree on what is conserved?
  jepa/          a tiny pixel JEPA world model (PyTorch)
experiments/     one script per stage
scripts/         server helpers: environment setup, background runs
docs/            hypotheses (read first), roadmap, reading list, literature map
agents/          how a team of AI agents works in this repository
paper/           skeleton of the paper
results/         the ledger and the reports; nothing counts unless it is here
tests/
```

## Run it on the server

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -e ".[pixels,dev]"
```

```bash
pytest -q
```

Then the stages, in order:

```bash
python experiments/pilot_synthetic.py
```

Stage A. Ideal latents, seeds 1 to 5, CPU, a few minutes. It prints a GO or NO-GO against the rule in `docs/hypotheses.md`.

```bash
python experiments/robustness_sweep.py
```

Stage B. Bends the latent and adds noise to find where the method breaks. CPU, roughly an hour.

```bash
python experiments/pilot_pixels.py --device cuda --steps 20000
```

Stage C. Trains a small pixel world model and runs the same comparison on latents it learned by itself. This is the first real test. Read "Is the encoder any good?" at the bottom of its report before reading the AUROCs.

`scripts/` wraps the same steps: `bash scripts/setup_server.sh` builds the environment and runs the tests, and `bash scripts/run_stage_c.sh` starts Stage C in the background so it survives a dropped SSH session.

Every run appends to `results/ledger.jsonl` and writes `report.md` and `summary.json` under `results/<stage>/<settings>/`. Seed 0 is for exploring; report seeds 1 and up.

## How the invariants are learned

For a latent `z`, take `u = (z[t], z[t] - z[t-1])` and all its monomials up to degree 2, `phi(u)`. Find the directions `w` that minimise the change of `w . phi(u)` between two steps of one clip that are far apart, subject to unit variance across clips. That is a generalised eigenproblem with a closed-form solution. At lag 1 it is Slow Feature Analysis. Long lags that span bounces and collisions are what separate a conserved quantity from one that is merely slow.

Steps where the latent moves abruptly are skipped, because a finite difference across a bounce is wrong. Invariants are read between events and compared across them. A clip's score is the largest lasting shift in any invariant.

Only ordinary lawful clips are used for fitting. No labels, no impossible examples, no simulator state.

This learner is not new. Bao (arXiv 2608.23526) finds an energy-like invariant in a DreamerV3 latent with the same criterion, and its roots are Slow Feature Analysis and Koopman theory. What this project adds is the three-way test, the handling of events, and the use of invariants as a detector. `docs/novelty.md` has the full account.

## Before making this public

- Choose a licence. None is included.
- Run `git init` here; the folder ships without Git history.
- Swap the SIGReg re-implementation for the official one, or check it against it.
