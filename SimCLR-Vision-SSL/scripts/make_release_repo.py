"""Build the stand-alone paper repository (github.com/MahmoudAlyosify1/scarce-labels-contrastive) from this folder.

Only the part of SimCLR-Vision-SSL that the paper needs is copied, from the files git tracks here (so nothing
untracked or ignored slips in): the code, the job files, the per-run results, the paper sources and the built
PDFs. Left out: the course project (src/, deployment/, wandb/, outputs/, figures/, the course report and its
notebooks), the first version of the paper, build by-products, and the authors' internal notes (submission plans,
reviews and responses to them).

    python scripts/make_release_repo.py <target_dir>     # the target is emptied first, except for its .git
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEEP_TOP = ("dlt/", "scripts/", "jobs/", "tests/", "configs/", "results/", "paper/",
            "notebooks/SimCLR_Vision_SSL_full_code.ipynb", "LICENSE", "requirements.txt", "environment.yml",
            "Dockerfile")
DROP = ("paper/old_tmlr/", "paper/tmlr_submission.pdf", "paper/overleaf_project.zip", "paper/arxiv_submission.tar.gz",
        "paper/cover_letter", "paper/SUBMISSION.md", "paper/JOURNAL_NOTES.md", "paper/RELATED_WORK_NOTES.md",
        "paper/ARXIV_METADATA.md", "paper/RESPONSE_", "paper/make_overleaf_package.sh", "scripts/make_assets_v1.py",
        "scripts/zenodo_upload.py", "scripts/queue_watchdog.sh", "scripts/resume_queue.sh", "scripts/resume_rev3.sh")
BUILD_EXT = (".aux", ".log", ".out", ".blg", ".spl", ".synctex.gz", ".fls", ".fdb_latexmk")
# built PDFs are copied even if git does not track them
EXTRA = ("paper/arxiv.pdf", "paper/submission.pdf", "paper/highlights.pdf")

README = """# Where Do Scarce Labels Belong?

Code, job files, per-run results and paper sources for

> **Where Do Scarce Labels Belong? Evaluating Contrastive Pretraining under Equal Compute and Realistic Label Budgets**
> Mahmoud Alyosify, Mirna Imbabi, Natalie Nashed, Hazem M. Abbas. School of Computing, Queen's University.

The paper (preprint layout) is [`paper/arxiv.pdf`](paper/arxiv.pdf). The checkpoints of all pretraining runs are
archived with the code and results on Zenodo: [doi:10.5281/zenodo.24494137](https://doi.org/10.5281/zenodo.24494137).

## What is here

| folder | content |
|--------|---------|
| `dlt/` | the library: data and labeled subsets, GPU augmentation, the one contrastive loss behind SimCLR, SupCon and SupCon+U, BYOL, SimSiam, cross-entropy, FixMatch, MixMatch, fine-tuning, self-distillation, evaluation (linear probes, kNN, CDNV) |
| `scripts/` | the queue that runs the job files, the fine-tuning recipe search, label-limited re-evaluation, coverage pools, statistics (hierarchical bootstrap, Holm, TOST), and `make_assets.py`, which writes every number, table and figure of the paper |
| `jobs/` | one line per run: the exact arguments of every run in the paper (`phases.txt` and `phases_rev3_*.txt` list them in order) |
| `results/runs/` | per run: `result.json` (arguments, metrics, curves) and the test-set predictions (`preds.npz`); `_logs/` has the training logs |
| `paper/` | LaTeX sources (Elsevier `elsarticle`), `generated/` (numbers and tables written by `make_assets.py`), figures, built PDFs |
| `notebooks/` | the whole code in one notebook, organised by pipeline stage (generated, not executed) |

## Reproducing

```bash
pip install -r requirements.txt                 # Python 3.12, CUDA GPU (all results: one NVIDIA A40)
export DLT_DATA=/path/to/data                   # CIFAR-10/100 are downloaded; STL-10 and Tiny-ImageNet: scripts/prepare_*.py
python -m pytest tests                          # GPU augmentation against torchvision

# one run (any line of jobs/*.txt; lines without "-m" are arguments of dlt.train)
python -m dlt.train --name simclr_all_e100_s0 --method simclr --pool all --epochs 100 --seed 0 --ckpt_dir ckpt

# every run of the paper, in order (finished runs are skipped, so the queue can be stopped and restarted)
for ph in jobs/phases.txt jobs/phases_rev3_grid.txt jobs/phases_rev3_pre.txt jobs/phases_rev3_r50.txt; do
  bash scripts/run_phases.sh $ph ckpt logs 2
done
python scripts/reeval.py --ckpt_dir ckpt        # label-limited probes and the CDNV split
python scripts/make_assets.py                   # all numbers, tables and figures -> paper/generated, paper/figures
cd paper && bash build.sh                       # arxiv.pdf and submission.pdf
```

A 9,700-step ResNet-18 contrastive run on CIFAR-10 takes about 12 minutes on an A40; all runs together took
several hundred GPU-hours. `make_assets.py` needs only `results/`, so every number of the paper can be checked
without a GPU.

## License

MIT (see `LICENSE`). CIFAR, STL-10 and Tiny-ImageNet keep their own licenses.
"""


def tracked():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    return [p for p in out.decode().split("\0") if p]


def keep(path):
    if not path.startswith(KEEP_TOP) or path.startswith(DROP) or path.endswith(BUILD_EXT):
        return False
    return "__pycache__" not in path and ".ipynb_checkpoints" not in path


def main():
    target = os.path.abspath(sys.argv[1])
    os.makedirs(target, exist_ok=True)
    for name in os.listdir(target):           # empty the target, keep its git history
        if name != ".git":
            p = os.path.join(target, name)
            shutil.rmtree(p) if os.path.isdir(p) and not os.path.islink(p) else os.remove(p)
    files = sorted({p for p in tracked() if keep(p)} | {p for p in EXTRA if os.path.exists(os.path.join(ROOT, p))})
    for p in files:
        src = os.path.join(ROOT, p)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(target, p)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    open(os.path.join(target, "README.md"), "w").write(README)
    open(os.path.join(target, ".gitignore"), "w").write(
        "__pycache__/\n*.pyc\n.ipynb_checkpoints/\nckpt/\nlogs/\ndata/\n*.pt\n"
        "paper/*.aux\npaper/*.log\npaper/*.out\npaper/*.blg\npaper/*.spl\npaper/*.synctex.gz\n")
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(target) if "/.git" not in d for f in fs)
    print(f"{len(files)} files, {size / 1e6:.1f} MB -> {target}")


if __name__ == "__main__":
    main()
