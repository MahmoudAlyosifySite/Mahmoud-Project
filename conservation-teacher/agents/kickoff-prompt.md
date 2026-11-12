# Kickoff

You are picking up a research project on a GPU server. I am the researcher who owns it. I will not be watching, so work through it on your own and stop only at the points listed under "When to stop and ask".

## What this is

This repository tests one idea: a world model's latent space can be taught to find what never changes (energy, the number of objects), and those invariants can tell an impossible event from one that is only improbable, which prediction error cannot do. The code is written and tested. No experiment has been run. Your job is to run the experiments in order on this server, and then write the paper that the results support.

Start by reading, in this order: `README.md`, `CLAUDE.md`, `docs/hypotheses.md`, `docs/novelty.md`, `docs/roadmap.md`. `CLAUDE.md` holds the rules of evidence; follow them throughout. `docs/hypotheses.md` fixes the hypotheses and the go rule before any result exists. Do not change them to fit what you find.

## What done looks like

A complete draft of the paper in `paper/main.tex`, compiled to PDF if LaTeX is installed, in which every number comes from `results/ledger.jsonl` and Appendix A maps each claim to its ledger entries. Plus a short final report to me.

Which paper it is depends on the results. All three outcomes listed under "The paper" in `docs/roadmap.md` are acceptable. A paper that reports the method failing on learned latents, with the benchmark and the reasons, is a good result. A paper containing a number that is not in the ledger is not.

## The work, in order

1. **Set up.** Run `bash scripts/setup_server.sh`. Check `nvidia-smi`: confirm the GPU (an A40 is expected) and that nobody else's job is on it. If this folder is not a Git repository, run `git init` and make a first commit. Commit locally after every stage. Never push.

2. **Stage A.** `python experiments/pilot_synthetic.py`. Check the two sanity conditions in the roadmap. Then do the sensitivity runs listed there; the script has no flags for the scenario parameters yet, so add them.

3. **Stage B.** `python experiments/robustness_sweep.py`. It is a CPU job and can run beside a GPU job.

4. **Stage C, the first real test.** Get the encoder to work first, using seed 0 only: run `experiments/pilot_pixels.py --seeds 0` and adjust `--steps`, `--lam`, `--dim` and `--size` until the linear probe R2 in the report is at least 0.9 and the latent has not collapsed. Tune against the probe and the collapse statistics only, never against an AUROC. Then freeze the settings, record them with the date under Changes in `docs/hypotheses.md`, and run seeds 1 to 3 once. Apply the go rule as written. If it fails, try each declared secondary option once and report all of them.

5. **Agreement between encoders (H4).** Train two seeds and measure how well their invariants agree, with `src/conserve/agreement.py`.

6. **Check the regulariser.** Before any Stage C number goes in the paper, compare `src/conserve/jepa/sigreg.py` with the official implementation in the LeJEPA and le-wm repositories. Either fix it and rerun Stage C, or state the difference in the paper.

7. **Finish the reading.** If you have web access, read the papers listed as unread in `docs/novelty.md` and correct that file if they change the picture. Claim nothing as new beyond what that file supports.

8. **Write the paper.** Follow the positioning in `docs/novelty.md`: lead with the three-way benchmark and what it shows about prediction error, present the invariant score as a simple tool built from known parts, and cite the precedents listed there. Make the figures from the `summary.json` files with a script kept in the repository. Report every seed and every family, including the failures: the continuity family, noise, strong nonlinearity. Write the Limitations section from what actually went wrong.

9. **Stop there.** Do not start Stage D. Write a one-page plan for it in `docs/stage-d-plan.md`.

## What you may decide yourself

- Long runs are approved, which is the go-ahead `CLAUDE.md` asks for: a single job may run up to 12 hours. Use the scripts in `scripts/` or `nohup` so a job survives a dropped connection, and log the wall time.
- Fix bugs you find. If a fix changes how a score or metric is computed, rerun everything that used it and say so in the report.
- Add scenarios, baselines or diagnostics that make the test fairer. Give the prediction-error baseline its strongest reading.

## When to stop and ask

- The tests fail on this machine and you cannot make them pass without changing what they test.
- The GPU is missing, or busy with someone else's work.
- A result looks too good, for example an AUROC near 1.0 on learned latents. Look for a leak first and tell me what you checked.
- You want to change a hypothesis, the go rule, or a primary score.
- A job would need more than 12 hours, or you would need to download a dataset or a checkpoint from outside.

Otherwise keep going. Do not stop between stages to check in.

## The final report

Write it to me in Egyptian Arabic; keep the repository in English. Lead with the verdict for each hypothesis. Then: what ran (commands, seeds, wall times), what failed or was skipped, what is still unverified, and where the paper is. Say plainly if the main idea did not work.
