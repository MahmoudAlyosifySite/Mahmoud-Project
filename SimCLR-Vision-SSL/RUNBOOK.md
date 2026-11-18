# Runbook: second revision of the paper

Paper: *Where Do Scarce Labels Belong? A Compute-Matched Study of Contrastive Pretraining*
(target journal: Neural Networks; fallback: Pattern Recognition; arXiv version in the same `elsarticle` template).

## Where everything is

| what | where | survives a server restart? |
|------|-------|----------------------------|
| code, job files, per-run results (`result.json`, `preds.npz`), paper sources | this folder (`/home/jovyan/Mahmoud-Project/SimCLR-Vision-SSL`), pushed to GitHub `MahmoudAlyosifySite/Mahmoud-Project` | yes |
| checkpoints (`<run>/model.pt`), datasets (CIFAR, STL-10, Tiny-ImageNet archives), queue logs | `/home/jovyan/simclr_store/{ckpt,data,logs}` (persistent volume) | yes |
| running processes (queue, watchdog, the job in progress) | memory of the server | **no** |

## Check the GPU queue

```bash
cd /home/jovyan/Mahmoud-Project/SimCLR-Vision-SSL
bash scripts/queue_status.sh          # running?, jobs on the GPU, finished runs per phase, failed jobs
```

The queue runs the job files listed in `jobs/phases.txt`, in order. A run counts as finished when
`results/runs/<name>/result.json` exists, so the queue can be stopped and restarted at any time; only the run in
progress is lost and starts again from the beginning.

## If something goes wrong

| situation | what to do |
|-----------|------------|
| the server / JupyterHub pod restarted | `bash scripts/resume_queue.sh` (starts the queue and the watchdog; finished runs are skipped) |
| the queue process died but the server is up | nothing: the watchdog restarts it within 5 minutes (`/home/jovyan/simclr_store/logs/watchdog.log`); or run `bash scripts/resume_queue.sh` |
| `queue_status.sh` lists a failed job | read `results/runs/_logs/<name>.log`; to run it again, put its line from `jobs/rev_*.txt` into a new file `jobs/rev_retry_<n>.txt`, append that file to `jobs/phases.txt`, and run `bash scripts/resume_queue.sh` |
| out-of-memory failures | the GPU is shared; check `nvidia-smi` for other users' processes, then requeue as above |
| disk full | `/home/jovyan` (49 GB) holds the checkpoints; the remaining runs need about 8 GB. Delete `/home/jovyan/simclr_store/data/*_hf` caches or old checkpoints of pilot runs first |
| GitHub push rejected | `git pull --rebase origin main` and push again (other projects share the repository) |

Starting the queue by hand (equivalent to `resume_queue.sh`):
```bash
export DLT_DATA=/home/jovyan/simclr_store/data
setsid nohup scripts/run_phases.sh jobs/phases.txt /home/jovyan/simclr_store/ckpt /home/jovyan/simclr_store/logs 2 > /dev/null 2>&1 &
```

## After new runs finish: update the paper

```bash
export DLT_DATA=/home/jovyan/simclr_store/data
python scripts/reeval.py --ckpt_dir /home/jovyan/simclr_store/ckpt      # label-limited probes and CDNV split (skips done runs)
python scripts/diversity.py --ref_run simclr_all_e100_s0 --ckpt_dir /home/jovyan/simclr_store/ckpt --extra_refs supervised=ce_f1.0_e100_s0   # after the coverage runs
python scripts/make_assets.py                                            # all numbers, tables and figures
cd paper && bash build.sh && bash make_arxiv_package.sh
```

Text that depends on unfinished runs is wrapped in `\pending{...}` (gray italics in the PDF; `??` marks a number
whose runs are missing). Once the runs behind a `\pending` sentence are finished, rewrite the sentence from the real
numbers (the conclusion may change!) and remove the wrapper. A version with any `\pending` left is a draft.
Writing rule (from the authors): every negative result is followed by a one- or two-sentence scientific reason
(the mechanism, from the loss, the data or our measurements; say "likely" if untested). No long discussions.

## State of the revision (keep this list current)

Integrated in the paper: compute-matched 2x2 at 970-19,400 steps (5 seeds at 9,700 and 19,400); protocol table;
fine-tuning recipe search with the label budget (fails for label-using encoders: their validation images are already
separated) and oracle grid (3 seeds: about half of the fine-tuning gap is the recipe; SimCLR stays ~3 points ahead);
two-stage fine-tuning; SimCLR v2-style self-distillation; BYOL fine-tuning;
38,800-step runs (3 seeds: SupCon flat at ~84, SimCLR 89.9); colour jitter with 3 seeds incl. SupCon+U; CIFAR-100
at 9,700 steps with 3 seeds;
STL-10 at 20,500 steps (3 seeds, incl. FixMatch from scratch); FixMatch from every encoder (3 seeds) and the
equal-total-compute comparison with FixMatch from scratch; three-way CDNV split with a cross-entropy control; CDNV
ratio during training; ResNet-50 at 19,400 steps (1 seed); 97,000 steps for all four arms with 2 of 3 seeds
(linear probes only; the sentence keeps a `\pending` note until seed 2 and the fine-tuning runs p11 finish); compute table; statistics (hierarchical bootstrap, Holm,
TOST); new related work, audit of budget reporting, journal statements, elsarticle template.

Waiting for runs (phase in `jobs/phases.txt`): STL-10 at 41,000 steps (p08c-p08e; 96-px phases run one job at a time); 97,000 steps (seed 2 of p10, then p11); coverage intervention (p09); label fraction x budget grid (p12/p13); SupCon
regularisation and ResNet-50 seeds (p14); CIFAR-100 at 19,400 steps (p15/p16); Tiny-ImageNet (p17-p20).

Before submission (authors' decisions, see `paper/SUBMISSION.md`): CRediT roles, funding, the journal's
generative-AI declaration, JCR quartile check, Zenodo DOI for the code, the journal's Guide for Authors.

## Continuing with a new Claude session

Start the session in this folder and give it this message:

> Read RUNBOOK.md and paper/SUBMISSION.md in /home/jovyan/Mahmoud-Project/SimCLR-Vision-SSL. Check the queue with
> scripts/queue_status.sh (restart it with scripts/resume_queue.sh if needed), then continue the second revision:
> integrate every finished run into the paper, rewrite the \pending sentences from the real numbers, rebuild, and push.
> Reply in Arabic. Do not add yourself as co-author or add any AI acknowledgement.
