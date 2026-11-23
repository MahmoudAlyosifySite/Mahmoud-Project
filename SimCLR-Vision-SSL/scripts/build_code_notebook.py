"""Build notebooks/SimCLR_Vision_SSL_full_code.ipynb: every source file of the project, organised by role.

    python scripts/build_code_notebook.py .      # from the repository root; re-run after code changes

Each source file becomes one `%%writefile <path>` cell with the file's exact content, so the notebook can be
read top to bottom or run to recreate the code tree in an empty folder. No cell trains, downloads or evaluates
anything; the commands that do are shown in Markdown blocks. The notebook is written without outputs.
"""
import json
import os
import sys

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = sys.argv[1]
OUT = os.path.join(REPO, "notebooks", "SimCLR_Vision_SSL_full_code.ipynb")
os.chdir(REPO)

cells = []
written = []          # every path written by a %%writefile cell, for the folder-creation cell and the check



def md(text):
    cells.append(new_markdown_cell(text.strip("\n")))


def src(path, text=None):
    """One file: a short Markdown description, then a %%writefile cell with the exact content."""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    nlines = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
    head = f"**`{path}`** ({nlines} lines)"
    md(head + (" — " + text if text else ""))
    target = '"' + path + '"' if " " in path else path
    cells.append(new_code_cell(f"%%writefile {target}\n" + content))
    written.append(path)


def job_count(path):
    n = 0
    for line in open(path, encoding="utf-8"):
        if line.split("#")[0].strip():
            n += 1
    return n


# ------------------------------------------------------------------------------------------------------------
md("""
# SimCLR-Vision-SSL: the complete code

Code behind the paper *Where Do Scarce Labels Belong? A Compute-Matched Study of Contrastive Pretraining*
(Mahmoud Alyosify, Mirna Imbabi, Natalie Nashed, Hazem M. Abbas; Queen's University), together with the earlier
course project it grew out of.

**What this notebook is.** Every source file of the project, grouped by its role in the pipeline, each with a
one-line description. A code cell holds one file, unchanged, behind a `%%writefile <path>` line.

**How to use it.**
* *To read:* go through the sections in order; the table of contents follows the order in which the pipeline runs.
* *To rebuild the code tree:* run all cells. The first code cell creates the folder `SimCLR-Vision-SSL-code/`
  next to the notebook and moves into it; every other code cell only writes one file there. Nothing is trained,
  downloaded or evaluated. The commands that run the experiments are in Markdown blocks (Section 9), so they never
  run by accident.

The notebook was generated from the repository by a script and saved without outputs; it has not been run.
""")

md("""
## Contents

0. [Set-up: target folder](#0.-Set-up)
1. [Environment](#1.-Environment)
2. [The `dlt` package: data, augmentations, models, loss, training, evaluation](#2.-The-dlt-package)
3. [Tests](#3.-Tests)
4. [Datasets and labeled pools](#4.-Datasets-and-labeled-pools)
5. [The GPU job queue](#5.-The-GPU-job-queue)
6. [Experiment definitions (job files)](#6.-Experiment-definitions)
7. [Analysis: re-evaluation, statistics, tables and figures](#7.-Analysis)
8. [Building the paper](#8.-Building-the-paper)
9. [Running the whole study (commands, not executed)](#9.-Running-the-whole-study)
10. [Appendix A: the original course project](#Appendix-A:-the-original-course-project)
11. [Appendix B: the visual-search demo](#Appendix-B:-the-visual-search-demo)
12. [Appendix C: Weights & Biases sync and the first asset generator](#Appendix-C:-W&B-sync-and-the-first-asset-generator)
13. [Appendix D: code cells of the earlier notebooks](#Appendix-D:-code-cells-of-the-earlier-notebooks)

### How the pieces fit together

```
datasets (CIFAR-10/100 auto-download; STL-10, Tiny-ImageNet via scripts/prepare_*.py)
   │
   ▼
jobs/*.txt ──► scripts/run_queue.py / run_phases.sh ──► python -m dlt.train | dlt.finetune | dlt.semisup | dlt.distill | scripts.ft_search
                                                           │   (one run = results/runs/<name>/result.json + preds.npz; checkpoint model.pt)
                                                           ▼
                                      scripts/reeval.py (label-limited probes, CDNV split) · scripts/diversity.py (pool statistics)
                                                           │
                                                           ▼
                                      scripts/make_assets.py (+ scripts/stats.py) ──► paper/generated/*.tex, paper/figures/*.pdf
                                                           │
                                                           ▼
                                                  paper/build.sh ──► arxiv.pdf, submission.pdf
```

| question in the paper | code | experiments |
|---|---|---|
| SimCLR vs. SupCon vs. SupCon+U at equal steps | `dlt/losses.py`, `dlt/train.py` | `jobs/01_main.txt`, `jobs/05_revision_main.txt`, `jobs/rev_p01_short.txt`, `rev_p06`, `rev_p10` |
| evaluation protocols (probes, kNN, fine-tuning) | `dlt/evaluate.py`, `dlt/finetune.py`, `scripts/ft_search.py` | `jobs/03_finetune.txt`, `rev_p02`–`rev_p05`, `rev_p07`, `rev_p11` |
| semi-supervised training and distillation | `dlt/semisup.py`, `dlt/distill.py` | `jobs/02_semisup.txt`, `rev_p04_fixmatch_from`, `rev_p05b/c` |
| representation geometry (CDNV) | `dlt/evaluate.py` (`geometry`), `scripts/reeval.py` | all encoders |
| pool coverage | `scripts/make_pools.py`, `scripts/diversity.py` | `rev_p09_coverage` |
| label fraction × budget grid | `dlt/train.py`, `dlt/finetune.py` | `rev_p12`, `rev_p13`, `rev_p14b` |
| SupCon regularisation, ResNet-50 | `dlt/train.py`, `dlt/models.py` | `rev_p14` |
| other datasets | `scripts/prepare_stl10.py`, `scripts/prepare_tinyimagenet.py` | `rev_p08*` (STL-10), `rev_p15/16` (CIFAR-100), `rev_p17`–`rev_p20` (Tiny-ImageNet) |
| statistics, every number, table and figure | `scripts/stats.py`, `scripts/make_assets.py` | — |
""")

# placeholder for the set-up cell; filled after all paths are known
SETUP_INDEX = len(cells)
md("## 0. Set-up")
cells.append(None)

# ------------------------------------------------------------------------------------------------------------
md("""
## 1. Environment

Python 3.11 with PyTorch 2.x on one CUDA GPU (all reported numbers: one NVIDIA A40). `requirements.txt` and
`environment.yml` list the packages; the `Dockerfile` builds the same environment.
""")
src("requirements.txt", "Python packages.")
src("environment.yml", "Conda environment of the paper (`conda env create -f environment.yml`).")
src("Dockerfile", "Container with the same environment.")

# ------------------------------------------------------------------------------------------------------------
md("""
## 2. The `dlt` package

Everything that trains or evaluates a network. Datasets are kept on the GPU as uint8 tensors and augmented there
in batches. Budgets are gradient steps: `--epochs E` means `E * (N // 512)` steps, where `N` is the size of the
full training set (97 steps per unit on CIFAR).
""")
src("dlt/__init__.py", "package description; sets the CUDA allocator option that reduces out-of-memory failures when two runs share the GPU.")
src("dlt/data.py", "CIFAR-10/100, STL-10 and Tiny-ImageNet as uint8 GPU tensors; stratified, nested label subsets (the 1% subset is inside the 5% subset, and so on).")
src("dlt/gpu_aug.py", "batched GPU augmentations (crop, flip, colour jitter, grayscale, blur, erasing) with per-image random parameters, checked against torchvision in `tests/`; named pipelines (`simclr`, `simclr_strong`, ...).")
src("dlt/models.py", "ResNet-18/50 encoders with the CIFAR stem (and `_mp` variants with max-pooling for 64–96 px images), projection heads, BYOL/SimSiam models, classifier.")
src("dlt/losses.py", "one grouped contrastive loss: SimCLR (no labels), SupCon (all labels) and SupCon+U (some labels).")
src("dlt/train.py", "pretraining of one run (SimCLR, SupCon, SupCon+U, BYOL, SimSiam, cross-entropy) and its evaluation; writes `result.json`, `preds.npz` and the checkpoint.")
src("dlt/evaluate.py", "frozen-encoder evaluation: linear probes, weighted kNN, CDNV and simplex-ETF geometry, colour-reliance probes.")
src("dlt/finetune.py", "fine-tuning on the labeled subset only (linear-probe initialisation, then the whole network); default recipe and the recipes found by the search.")
src("dlt/semisup.py", "FixMatch and MixMatch on the same labeled split, from scratch or from a pretrained encoder.")
src("dlt/distill.py", "SimCLR v2-style pipeline: fine-tune a teacher, then self-distil into a student on all images.")

md("## 3. Tests")
src("tests/test_gpu_aug.py", "checks the GPU augmentations against torchvision and the loss against reference NT-Xent / SupCon code (`python -m tests.test_gpu_aug`).")

# ------------------------------------------------------------------------------------------------------------
md("""
## 4. Datasets and labeled pools

CIFAR-10/100 are downloaded on first use by `dlt/data.py`. STL-10 and Tiny-ImageNet are converted once from their
Hugging Face parquet exports into a single uint8 archive. `make_pools.py` builds the labeled pools of the coverage
experiment.
""")
src("scripts/__init__.py", "makes `scripts` importable (`python -m scripts.ft_search`).")
src("scripts/prepare_stl10.py", "STL-10 parquet export → `stl10_uint8.npz` (5,000 labeled, 100,000 unlabeled, 8,000 test images).")
src("scripts/prepare_tinyimagenet.py", "Tiny-ImageNet parquet export → `tinyimagenet_uint8.npz` (100,000 training and 10,000 validation images, 64 px).")
src("scripts/make_pools.py", "labeled pools of fixed size with controlled diversity: redundant groups of near-duplicates, greedy k-center selection, and mixtures with random images.")

# ------------------------------------------------------------------------------------------------------------
md("""
## 5. The GPU job queue

A job file lists one run per line (the arguments of `python -m <module>`, default `dlt.train`). A run counts as
finished when `results/runs/<name>/result.json` exists, so the queue can be stopped and restarted at any time.
`run_phases.sh` works through the job files of `jobs/phases.txt` in order; the watchdog restarts it if it dies.
""")
src("scripts/run_queue.py", "runs the jobs of one file on one GPU, a few at a time, skipping finished runs and retrying failures.")
src("scripts/run_phases.sh", "runs the job files of a phases file one after another; re-reads the phases file after every phase.")
src("scripts/resume_queue.sh", "starts the queue and its watchdog (safe to run again; nothing starts twice).")
src("scripts/queue_watchdog.sh", "every 5 minutes: restarts the queue if it died while phases are left and copies the logs to the persistent store.")
src("scripts/queue_status.sh", "one-screen status: is the queue running, which jobs are on the GPU, finished runs per phase, failed jobs.")

# ------------------------------------------------------------------------------------------------------------
md("""
## 6. Experiment definitions

Every run of the study, as job files. `00`–`04` are the first study, `05`–`06` the first revision, and
`jobs/phases.txt` lists the phases of the second revision (`rev_p*`) in the order the queue runs them.
""")
first = ["jobs/00_pilot.txt", "jobs/01_main.txt", "jobs/02_semisup.txt", "jobs/03_finetune.txt",
         "jobs/04_extra_finetune.txt", "jobs/05_revision_main.txt", "jobs/06_revision_followup.txt"]
for p in first:
    src(p, f"{job_count(p)} runs.")
src("jobs/phases.txt", "order of the second-revision phases.")
for p in [l.strip() for l in open("jobs/phases.txt") if l.strip() and not l.startswith("#")]:
    src(p, f"{job_count(p)} runs.")

# ------------------------------------------------------------------------------------------------------------
md("""
## 7. Analysis

After the runs: label-limited probes and the CDNV split for every saved encoder (`reeval.py`), statistics of the
pools (`diversity.py`), the fine-tuning recipe search (`ft_search.py`, launched from the job files), the
hierarchical bootstrap (`stats.py`), and `make_assets.py`, which writes every number, table and figure of the paper.
""")
src("scripts/reeval.py", "linear probes and kNN with 1–50% of the labels, probe fit on its own images, CDNV on labeled / other / test images; writes `probes.json`.")
src("scripts/diversity.py", "pool statistics in a fixed reference space: spread, Vendi score, coverage radius.")
src("scripts/ft_search.py", "fine-tuning recipe search on a split of the labeled images (72 recipes), plus the oracle grid scored on the test set.")
src("scripts/stats.py", "hierarchical bootstrap over seeds and test images, Holm correction, TOST equivalence test.")
src("scripts/summarize.py", "collects every `result.json` into one table with group means.")
src("scripts/hist_probe.py", "trivial colour baseline: a linear probe on the RGB histogram of the raw images.")
src("scripts/ood_resolution.py", "resolution shift: ImageNet dogs (Imagewoof) shrunk to 32×32, probe recall and retrieval precision.")
src("scripts/make_assets.py", "every number (`\\R{...}` macros), table and figure of the paper from `results/runs/`.")

# ------------------------------------------------------------------------------------------------------------
md("""
## 8. Building the paper

The LaTeX sources live in `paper/` (`arxiv.tex`, `submission.tex`, `highlights.tex`, the `sec_*.tex` files);
they are text, not code, and are not repeated here. The two scripts below build the PDFs and the arXiv package.
""")
src("paper/build.sh", "builds `arxiv.pdf`, `submission.pdf` and `highlights.pdf` and reports errors and undefined references.")
src("paper/make_arxiv_package.sh", "packs the sources, the `.bbl` file and the figures for arXiv and checks that the package compiles on its own.")

# ------------------------------------------------------------------------------------------------------------
md("""
## 9. Running the whole study

These commands are **not** executed by this notebook. Run them in a terminal from the repository root.

**Set-up and checks**
```bash
conda env create -f environment.yml && conda activate spend-labels-last   # or: docker build -t eic .
export DLT_DATA=./data                       # CIFAR-10/100 are downloaded here on first use
python -m tests.test_gpu_aug                 # GPU augmentations and loss against reference code

# STL-10 and Tiny-ImageNet: download the parquet exports, then build one uint8 archive each
#   huggingface-cli download jxie/stl10 --repo-type dataset --local-dir $DLT_DATA/stl10_hf
python scripts/prepare_stl10.py --src $DLT_DATA/stl10_hf --out $DLT_DATA
#   huggingface-cli download zh-plus/tiny-imagenet --repo-type dataset --local-dir $DLT_DATA/tin_hf
python scripts/prepare_tinyimagenet.py --src $DLT_DATA/tin_hf --out $DLT_DATA
```

**1. First study**
```bash
for f in jobs/00_pilot.txt jobs/01_main.txt jobs/02_semisup.txt jobs/03_finetune.txt jobs/04_extra_finetune.txt; do
  python scripts/run_queue.py $f --out results/runs; done
python scripts/make_pools.py --ref_run simclr_all_e100_s0 --ckpt_dir results/runs --mix 0.25 0.5 0.75
```

**2. Everything else, in the order of `jobs/phases.txt`** (finished runs are skipped; safe to stop and restart)
```bash
scripts/run_phases.sh jobs/phases.txt results/runs results/_queue_logs 1
# on the shared server: checkpoints and logs on the persistent volume, two runs at a time, with a watchdog
bash scripts/resume_queue.sh
bash scripts/queue_status.sh
```

**3. Analysis and paper**
```bash
python scripts/reeval.py    --ckpt_dir results/runs      # label-limited probes, CDNV on labeled / other / test images
python scripts/diversity.py --ref_run simclr_all_e100_s0 --ckpt_dir results/runs --extra_refs supervised=ce_f1.0_e100_s0
python scripts/make_assets.py                # every table, figure and number -> paper/generated, paper/figures
python scripts/stats.py simclr_all_e200 supcon_lab_f0.1_e200 --key linear   # any pairwise hierarchical-bootstrap test
cd paper && bash build.sh && bash make_arxiv_package.sh
```

**Single runs by hand** (CIFAR-10, 10% labels, 9,700 steps)
```bash
python -m dlt.train --name simclr     --method simclr --pool all     --epochs 100                          # SimCLR, all images
python -m dlt.train --name simclr_sub --method simclr --pool labeled --label_fraction 0.1 --epochs 100     # SimCLR, labeled subset
python -m dlt.train --name supcon     --method supcon --pool labeled --label_fraction 0.1 --epochs 100     # SupCon, labeled subset
python -m dlt.train --name supcon_u   --method supcon --pool all     --label_fraction 0.1 --epochs 100 \\
       --temperature_unlabeled 0.5 --labeled_share 0.25                                                     # SupCon+U, all images
python -m dlt.finetune  --name ft_simclr --from_run simclr --label_fraction 0.1                           # FT-5k (default recipe)
python -m scripts.ft_search --name fts_simclr --from_run simclr --ckpt_dir results/runs --label_fraction 0.1  # recipe search
python -m dlt.semisup   --name fm_simclr --method fixmatch --label_fraction 0.1 --init_run simclr          # FixMatch from the encoder
python -m dlt.train     --name two_stage --method supcon --pool labeled --label_fraction 0.1 --epochs 5 --lr 0.02 --init_run simclr
python -m dlt.distill   --name distill   --from_run simclr --label_fraction 0.1                           # fine-tune + self-distil
```
""")

# ------------------------------------------------------------------------------------------------------------
md("""
## Appendix A: the original course project

CISC 867 (Group 20, Queen's University): SimCLR and SupCon on CIFAR-10 with augmentation ablations, before the
compute-matched study. Two snapshots, `src/Midterm Work` and `src/Final Work`, plus the ablation launcher at the
repository root. The scripts read `configs/augmentation.yaml` relative to their own folder.
""")
src("configs/augmentation.yaml", "augmentation settings of the course project (repository root).")
md("### Midterm version")
src("src/Midterm Work/configs/augmentation.yaml", "augmentation settings.")
src("src/Midterm Work/augmentations.py", "augmentation pipelines built from the YAML settings.")
src("src/Midterm Work/dataset.py", "dataset and augmentation registry, SimCLR and SupCon loaders.")
src("src/Midterm Work/loss.py", "NT-Xent and SupCon losses.")
src("src/Midterm Work/model.py", "ResNet encoder with projection head.")
src("src/Midterm Work/train_master.py", "SimCLR training script with logging and checkpoints.")
md("### Final version")
src("src/Final Work/configs/augmentation.yaml", "augmentation settings.")
src("src/Final Work/run_config.json", "configuration of the final runs.")
src("src/Final Work/augmentations.py", "augmentation pipelines built from the YAML settings.")
src("src/Final Work/dataset.py", "dataset and augmentation registry, SimCLR and SupCon loaders.")
src("src/Final Work/loss.py", "NT-Xent and SupCon losses.")
src("src/Final Work/model.py", "ResNet encoder with projection head.")
src("src/Final Work/train_master.py", "SimCLR training script (final phase).")
src("src/Final Work/supcon/dataset_subset.py", "stratified 10% CIFAR-10 subset for SupCon.")
src("src/Final Work/supcon/loss_supcon.py", "SupCon loss (Khosla et al., 2020).")
src("src/Final Work/supcon/train_supcon.py", "SupCon training on the 10% subset.")
src("src/Final Work/run_final_ablations_18_may.py", "launches the final augmentation ablations, one folder per experiment.")
src("src/Final Work/run_beast_experiments.py", "launches the 200-epoch experiments.")
src("src/Final Work/run_top_5_production.py", "runs five selected augmentation configurations for 200 epochs.")
src("src/Final Work/evaluate_all_probes.py", "linear-probe evaluation of every trained encoder.")
src("src/Final Work/evaluate_and_plot_beast.py", "evaluation and plots of the 200-epoch experiments.")
src("src/Final Work/plot_ablation.py", "figures of the ablation study.")
src("run_ablations.py", "first ablation launcher (repository root).")

# ------------------------------------------------------------------------------------------------------------
md("""
## Appendix B: the visual-search demo

Streamlit app: ONNX-exported SimCLR encoder and a FAISS index over the CIFAR-10 test set (the model files and
the index are binary and are not part of this notebook).

""")
src("deployment/app.py", "Streamlit visual-search app: upload or pick an image, nearest CIFAR-10 test images by cosine similarity.")
src("deployment/build_faiss.py", "builds the FAISS index and metadata for the CIFAR-10 test set.")
src("deployment/export_onnx.py", "exports the SimCLR encoder to ONNX.")

md("## Appendix C: W&B sync and the first asset generator")
src("wandb/sync_to_wandb.py", "uploads the logs of the course-project ablations (experiments 38–42) to Weights & Biases.")
src("scripts/make_assets_v1.py", "asset generator of the first version of the paper (TMLR template, `paper/old_tmlr/`); replaced by `scripts/make_assets.py`.")

# ------------------------------------------------------------------------------------------------------------
md("""
## Appendix D: code cells of the earlier notebooks

The two exploratory notebooks of the course project, code cells only (Markdown cells and outputs omitted). They
are shown as text so that running this notebook does not run them; open the original notebooks to execute them.
""")
for nbpath in ["notebooks/visualize-pairs.ipynb",
               "notebooks/Linear evaluation notebook/supervised-baseline-and-linear-probe-evaluation.ipynb"]:
    nb = json.load(open(nbpath, encoding="utf-8"))
    code = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for c in nb["cells"] if c["cell_type"] == "code"]
    code = [c for c in code if c.strip()]
    md(f"### `{nbpath}` ({len(code)} code cells)")
    for i, c in enumerate(code, 1):
        fence = "````" if "```" in c else "```"
        md(f"*Cell {i}*\n\n{fence}python\n{c.rstrip()}\n{fence}")

# ------------------------------------------------------------------------------------------------------------
dirs = sorted({os.path.dirname(p) for p in written if os.path.dirname(p)})
setup = ('import os\n\n'
         '# Every %%writefile cell below writes one file of the project into this folder.\n'
         'TARGET = "SimCLR-Vision-SSL-code"\n'
         'DIRS = ' + json.dumps(dirs, indent=4) + '\n\n'
         'os.makedirs(TARGET, exist_ok=True)\n'
         'os.chdir(TARGET)\n'
         'for d in DIRS:\n'
         '    os.makedirs(d, exist_ok=True)\n'
         'print("writing the code tree into", os.getcwd())\n')
cells[SETUP_INDEX + 1] = new_code_cell(setup)

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
nbformat.validate(nb)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
nbformat.write(nb, OUT)
json.dump({"written": written}, open(os.environ.get("NB_MANIFEST", "/dev/null"), "w"))
print(f"{OUT}: {len(cells)} cells, {len(written)} files")
