"""Builds the arXiv submission of the BA paper and checks it compiles the way arXiv compiles it.

Run (after make_assets.py and build.sh):   python3 BA/paper/scripts/make_arxiv.py

Writes BA/paper/arxiv/ (git-ignored):
  src/                 the exact files uploaded (main.tex with \\pdfoutput=1, preamble, sections, generated numbers,
                       tables and Arabic PDFs, figures, elsarticle class and style, main.bbl; no .bib needed)
  ba-arxiv-src.tar.gz  the upload
  metadata.txt         title, author, abstract as plain text (checked against arXiv's 1920-character limit),
                       categories, comments, licence
  check.pdf            the PDF compiled from src/ with pdflatex only (no bibtex), as arXiv does
Fails if the clean build has errors, undefined references, missing numbers (??), a \\pending note, or an abstract
over the limit.
"""
import glob, os, re, shutil, subprocess, sys, tarfile

PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PAPER, "arxiv")
SRC = os.path.join(OUT, "src")
ABSTRACT_LIMIT = 1920


def fail(msg):
    sys.exit(f"make_arxiv: {msg}")


def numbers():
    R = {}
    for line in open(os.path.join(PAPER, "generated", "numbers.tex"), encoding="utf-8"):
        m = re.match(r"\\expandafter\\def\\csname R@(.+?)\\endcsname\{(.*)\}$", line.strip())
        if m:
            R[m.group(1)] = m.group(2)
    return R


def plain(tex, R):
    t = re.sub(r"\\R\{([^}]*)\}", lambda m: R[m.group(1)] if m.group(1) in R else fail(f"abstract key {m.group(1)} missing"), tex)
    for a, b in [(r"\vox", "VoxCPM2"), (r"\cbthree", "Chatterbox V3"), (r"\BA", "BA"), (r"\ensuremath{-}", "-"),
                 (r"\%", "%"), ("$>$", ">"), ("$<$", "<"), ("~", " ")]:
        t = t.replace(a, b)
    t = re.sub(r"\\emph\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\\[a-zA-Z]+\s?", "", t)
    t = t.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", t).strip()


def main():
    for f in ["main.pdf", "main.bbl", os.path.join("generated", "numbers.tex")]:
        if not os.path.exists(os.path.join(PAPER, f)):
            fail(f"{f} missing: run make_assets.py and build.sh first")
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(SRC)
    main_tex = open(os.path.join(PAPER, "main.tex"), encoding="utf-8").read()
    if "\\pdfoutput=1" not in main_tex:
        main_tex = "\\pdfoutput=1\n" + main_tex
    open(os.path.join(SRC, "main.tex"), "w", encoding="utf-8").write(main_tex)
    files = ["preamble.tex", "main.bbl", "elsarticle.cls", "elsarticle-num-names.bst"]
    files += sorted(glob.glob("sections/*.tex", root_dir=PAPER))
    files += ["generated/numbers.tex"] + sorted(glob.glob("generated/tab_*.tex", root_dir=PAPER))
    files += sorted(glob.glob("generated/ar/*.pdf", root_dir=PAPER)) + sorted(glob.glob("figures/*.pdf", root_dir=PAPER))
    for f in files:
        os.makedirs(os.path.dirname(os.path.join(SRC, f)) or SRC, exist_ok=True)
        shutil.copy2(os.path.join(PAPER, f), os.path.join(SRC, f))
    for f in glob.glob("sections/*.tex", root_dir=SRC):
        if re.search(r"\\pending\{", open(os.path.join(SRC, f), encoding="utf-8").read()):
            fail(f"{f} still contains a \\pending note")

    # compile as arXiv does: pdflatex with the shipped .bbl, no bibtex
    for _ in range(3):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=SRC,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log = open(os.path.join(SRC, "main.log"), encoding="latin-1").read()
    errors = re.findall(r"^! .*", log, re.M)
    undefined = re.findall(r"(?:Reference|Citation) `[^']+' on page \d+ undefined", log)
    missing = sorted(set(re.findall(r"MISSING-NUMBER: (\S+)", log)))
    if errors or undefined or missing or not os.path.exists(os.path.join(SRC, "main.pdf")):
        fail(f"clean build failed: errors={errors[:3]} undefined={undefined[:3]} missing numbers={missing[:5]}")
    pages = re.search(r"Output written on main\.pdf \((\d+) pages", log).group(1)
    shutil.copy2(os.path.join(SRC, "main.pdf"), os.path.join(OUT, "check.pdf"))
    for f in os.listdir(SRC):
        if f.startswith("main.") and f not in ("main.tex", "main.bbl"):
            os.remove(os.path.join(SRC, f))

    with tarfile.open(os.path.join(OUT, "ba-arxiv-src.tar.gz"), "w:gz") as tar:
        for root, _, fs in os.walk(SRC):
            for f in sorted(fs):
                p = os.path.join(root, f)
                tar.add(p, arcname=os.path.relpath(p, SRC))

    R = numbers()
    abstract = plain(open(os.path.join(PAPER, "sections", "abstract.tex"), encoding="utf-8").read(), R)
    if len(abstract) > ABSTRACT_LIMIT:
        fail(f"abstract is {len(abstract)} characters, arXiv allows {ABSTRACT_LIMIT}")
    pre = open(os.path.join(PAPER, "preamble.tex"), encoding="utf-8").read()
    title = re.search(r"pdftitle=\{([^}]*)\}", pre).group(1)
    n_tables = len(re.findall(r"\\begin\{table\}", "".join(open(os.path.join(SRC, f), encoding="utf-8").read()
                                                          for f in glob.glob("sections/*.tex", root_dir=SRC))))
    n_figs = len(glob.glob("figures/*.pdf", root_dir=SRC))
    meta = f"""Title:
{title}

Authors:
Mahmoud Alyosify

Abstract ({len(abstract)} of {ABSTRACT_LIMIT} characters):
{abstract}

Comments:
{pages} pages, {n_figs} figure, {n_tables} tables. Code, evaluation texts, per-utterance scores and the scripts behind every number: https://github.com/MahmoudAlyosifySite/Mahmoud-Project/tree/main/BA

Primary category: cs.CL (Computation and Language)
Cross-lists: eess.AS (Audio and Speech Processing), cs.SD (Sound)
Licence (recommended): CC BY 4.0
MSC/ACM classes: none
Journal reference: none (preprint)
"""
    open(os.path.join(OUT, "metadata.txt"), "w", encoding="utf-8").write(meta)
    print(meta)
    print(f"upload: {os.path.join(OUT, 'ba-arxiv-src.tar.gz')}  ({len(files) + 1} files; clean build {pages} pages, 0 errors)")


if __name__ == "__main__":
    main()
