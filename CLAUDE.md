# Rules for agents working in this repository

This is a research project. The product is a claim that survives checking, not code that runs. Read `docs/hypotheses.md` before proposing or running anything.

## Evidence

- A number exists only if it is in `results/ledger.jsonl`, written there by an experiment script. Never type a result by hand, never edit the ledger, never quote a number from memory or from a terminal scrollback.
- Seed 0 is for exploring. Claims use seeds 1 and up, and report every seed that was run. Do not drop a seed because it looks bad.
- A finding counts after a second run with new seeds, launched by a different session than the one that found it.
- Report failures and null results in the same detail as successes. A run that crashed or was stopped is logged as such.

## Keeping the test honest

- Fit on ordinary lawful clips only. Improbable and impossible clips are for testing and are never used to fit, tune, select or set a threshold.
- Simulator ground truth (energy, ball count, positions) is for diagnostics. It never enters the fitting of invariants or the choice among them. One declared exception: the linear probe R2 on lawful clips gates Stage C (`docs/hypotheses.md`) and may guide the tuning of the encoder. An AUROC never guides tuning.
- Do not change how a score or metric is computed in the same change as an experiment that uses it.
- The go rule, the thresholds and the list of primary scores live in `docs/hypotheses.md`. To change one, edit that file first with the date and the reason, then the code.
- The prediction-error baseline gets its best reading. If you find a stronger baseline, add it; do not weaken the existing ones.

## Before a new experiment

Copy `agents/proposal-template.md`, fill it in, and include the outcome you expect and what would change your mind. Proposals that only restate a hypothesis already in `docs/hypotheses.md` do not need a new file.

## Claims about novelty

Say "not found in these searches" and list the searches. Do not say "nobody has done this". Read a paper's full text before describing what it did not do.

## Compute

- One GPU job at a time on the A40 unless you have measured that two fit.
- Stage A and B are CPU jobs and can run beside a GPU job.
- Log wall time. Do not start a run longer than a few hours without the owner's go-ahead.

## Code

- `pytest -q` must pass before and after a change.
- Keep modules small and the experiment scripts thin. Shared logic goes in `src/conserve/`.
- The SIGReg in `src/conserve/jepa/sigreg.py` is a re-implementation. Any reported number that depends on it needs the official implementation or a check against it.
