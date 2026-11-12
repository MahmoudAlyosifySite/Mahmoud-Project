"""Build the arXiv submission from paper/.

    python paper/make_figures.py      # first: tables, figures and numbers from results/
    python paper/make_arxiv.py [--out DIR]

Steps:
1. Copies main.tex, generated/*.tex and figures/*.pdf into a clean folder.
2. Runs pdflatex and bibtex there once to produce main.bbl, which arXiv needs
   (arXiv does not run BibTeX), then removes everything arXiv should not get.
3. Compiles a second clean copy the way arXiv does (pdflatex only, with the .bbl)
   and fails on any LaTeX error, undefined reference or undefined citation.
4. Writes arxiv-<date>.tar.gz and arxiv-metadata.txt: title, authors, a plain-text
   abstract with every number expanded from generated/numbers.tex, suggested
   categories and the comments line.

The bundle is named ...-DRAFT.tar.gz while the author footnote still says
"to be added". Output goes to --out (default: a temporary folder), not to the repository.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

PAPER = Path(__file__).resolve().parent
PLACEHOLDER = "to be added"
ABSTRACT_LIMIT = 1920  # characters, arXiv's limit for the abstract field


def run(cmd: list[str], cwd: Path) -> str:
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300)
    return out.stdout + out.stderr


def macros() -> dict[str, str]:
    text = (PAPER / "generated" / "numbers.tex").read_text(encoding="utf-8")
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", text))


def expand(tex: str, table: dict[str, str]) -> str:
    """LaTeX abstract -> plain text for the arXiv form."""
    cp = "CDFiveOneTwolOnesTwoZeroZeroZeroZeropOneSixspOne"
    tex = re.sub(r"\\Cp\{([A-Za-z]+)\}", lambda m: table[cp + m.group(1)], tex)
    tex = re.sub(r"\\([A-Za-z]+)\{\}", lambda m: table.get(m.group(1), m.group(0)), tex)
    tex = re.sub(r"\\url\{([^}]*)\}", r"\1", tex)
    tex = tex.replace("~", " ").replace("``", '"').replace("''", '"')
    if "\\" in tex:
        raise ValueError(f"unexpanded LaTeX left in the abstract: {tex[tex.index(chr(92)):][:60]}")
    return " ".join(tex.split())


def stage(dst: Path) -> None:
    (dst / "generated").mkdir(parents=True)
    (dst / "figures").mkdir()
    shutil.copy(PAPER / "main.tex", dst / "main.tex")
    for f in sorted((PAPER / "generated").glob("*.tex")):
        shutil.copy(f, dst / "generated" / f.name)
    for f in sorted((PAPER / "figures").glob("*.pdf")):
        shutil.copy(f, dst / "figures" / f.name)


def check_log(log: str) -> list[str]:
    bad = [line for line in log.splitlines() if line.startswith("!")]
    bad += [line for line in log.splitlines() if re.search(r"(Reference|Citation) .* undefined", line)]
    return bad


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    out = args.out or Path(tempfile.mkdtemp(prefix="arxiv-"))
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="arxiv-build-"))

    # 1-2: bundle with a .bbl
    bundle = work / "bundle"
    stage(bundle)
    shutil.copy(PAPER / "refs.bib", bundle / "refs.bib")
    run(["pdflatex", "-interaction=nonstopmode", "main.tex"], bundle)
    bib = run(["bibtex", "main"], bundle)
    if not (bundle / "main.bbl").exists():
        raise SystemExit("bibtex produced no main.bbl:\n" + bib)
    for f in list(bundle.iterdir()):
        if f.is_file() and f.name not in ("main.tex", "main.bbl"):
            f.unlink()

    # 3: compile as arXiv would
    test = work / "test"
    shutil.copytree(bundle, test)
    for _ in range(3):
        log = run(["pdflatex", "-interaction=nonstopmode", "main.tex"], test)
    problems = check_log(log)
    if problems or not (test / "main.pdf").exists():
        raise SystemExit("arXiv-style build failed:\n" + "\n".join(problems[:20]))
    overfull = len(re.findall(r"Overfull \\hbox \((\d+\.\d+)pt", log))
    pages = re.search(r"Output written on main.pdf \((\d+) pages?", log)

    # 4: archive and metadata
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    draft = PLACEHOLDER in tex
    stamp = time.strftime("%Y%m%d")
    name = f"arxiv-{stamp}{'-DRAFT' if draft else ''}.tar.gz"
    with tarfile.open(out / name, "w:gz") as tar:
        for f in sorted(bundle.rglob("*")):
            if f.is_file():
                tar.add(f, arcname=str(f.relative_to(bundle)))
    shutil.copy(test / "main.pdf", out / "main.pdf")

    title = re.search(r"\\title\{(.*?)\}\n", tex, re.S).group(1).replace("\\\\\n", " ").replace("\\\\", " ")
    abstract_tex = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
    abstract = expand(abstract_tex, macros())
    n_fig = len(list((bundle / "figures").glob("*.pdf")))
    meta = [
        f"Title: {' '.join(title.split())}",
        "Authors: Mahmoud Alyosify",
        "Suggested primary category: cs.LG (Machine Learning)",
        "Suggested cross-lists: cs.AI, cs.CV",
        f"Comments: {pages.group(1) if pages else '?'} pages, {n_fig} figures. "
        "Code, pre-registered protocol and results ledger: https://github.com/MahmoudAlyosifySite/Mahmoud-Project",
        f"Abstract ({len(abstract)} of {ABSTRACT_LIMIT} characters allowed):",
        abstract,
    ]
    (out / "arxiv-metadata.txt").write_text("\n".join(meta) + "\n", encoding="utf-8")

    print(f"bundle:   {out / name}")
    print(f"pdf:      {out / 'main.pdf'} ({pages.group(1) if pages else '?'} pages, {overfull} overfull boxes)")
    print(f"metadata: {out / 'arxiv-metadata.txt'}")
    if len(abstract) > ABSTRACT_LIMIT:
        print(f"WARNING: abstract is {len(abstract)} characters; arXiv allows {ABSTRACT_LIMIT}")
    if draft:
        print(f"WARNING: the author footnote still says '{PLACEHOLDER}'. Fill it in paper/main.tex before submitting.")
    shutil.rmtree(work)


if __name__ == "__main__":
    main()
