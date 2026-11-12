# Conservation as a teacher

World models learn what changes: they predict the next state. The physical world is defined just as much by what never changes: energy, the number of objects, what an object is. This project teaches a world model's latent space to find those invariants by itself, and uses them to answer a question prediction error cannot: **is this event impossible, or only improbable?**

- **Impossible:** a conserved quantity takes a new value. A ball vanishes, or speeds up with nothing pushing it.
- **Improbable:** rare, but every law still holds. A ball far faster than any seen in training, or a bounce at an unusual angle.

Prediction error is high for both. An invariant moves only for the first.

## Status

Updated 2026-11-11, after Stages A to C were run on the A40 server. Every number below is in `results/ledger.jsonl`; the paper draft (`paper/main.tex`) quotes them only through macros generated from the results files by `paper/make_figures.py`. **All results come from one session and are not yet replicated by an independent session**, which `CLAUDE.md` requires before a finding counts.

| Part | State |
|---|---|
| Billiards simulator, three-way scenarios, renderer | Tested |
| Invariant learning, level-shift score, metrics, ledger | Tested |
| Stage A (ideal latents, seeds 1-5, plus four sensitivity runs) | Run. GO in all five. `results/pilot_synthetic/` |
| Stage B (bent and noisy latents, seeds 1-3) | Run. Energy invariant lost at alpha 1 and at any noise tested (0.003). `results/robustness/` |
| Stage C (pixel JEPA, seeds 1-3, five secondary options) | Run. Encoder gate not met (probe about 0.5 against 0.9); no invariant kept; AUROCs reported but not read as a test of H2. `results/pilot_pixels/` |
| H4 (agreement between encoders) | Positive control on ideal latents run; untestable on the Stage C encoders (no invariants). `results/agreement/` |
| Encoder diagnostics | Run. `results/encoder_diagnostics/` |
| SIGReg regulariser | Re-implemented, checked numerically against the official LeWM `SIGReg` and aligned (`experiments/check_sigreg.py`). Not the official file. |
| Stage D (LeWorldModel, V-JEPA 2) | Not started. Plan in `docs/stage-d-plan.md`. |
| Reading of the closest papers | Eleven papers; Shang et al. 2015 still unread (publisher refused access). `docs/novelty.md`, `docs/reading-list.md` |

Headline (details and every seed in the paper draft and the reports under `results/`):

- Prediction error conflates impossible and improbable events: in the energy family every reading ranks lawful fast clips above energy jumps on ideal latents, and the largest and mean errors of the pixel JEPA's own predictor do the same (its best reading stays well short of separating them) (H1).
- The invariant score separates them on ideal latents where a conserved quantity is broken, and is blind to a teleport (expected).
- It breaks with small frame-to-frame noise in the latent, and it finds nothing in the latent of the small pixel JEPA trained here. Whether a better encoder exposes conservation is open; that is Stage D.

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
