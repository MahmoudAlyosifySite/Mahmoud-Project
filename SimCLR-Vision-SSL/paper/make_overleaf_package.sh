#!/bin/bash
# Creates overleaf_project.zip: one Overleaf project with both versions of the paper (Overleaf: New Project ->
# Upload Project, or upload into an existing project). Main document: arxiv.tex (preprint) or submission.tex
# (journal review layout); choose it in Menu -> Main document. highlights.tex and cover_letter.tex compile too.
set -e
cd "$(dirname "$0")"
bash build.sh > /dev/null
rm -rf _overleaf && mkdir -p _overleaf/generated _overleaf/figures
cp arxiv.tex submission.tex highlights.tex cover_letter.tex preamble.tex frontmatter.tex abstract.tex body.tex \
   sec_*.tex statements.tex appendix.tex elsarticle.cls elsarticle-num-names.bst references.bib _overleaf/
cp generated/numbers.tex generated/tab_*.tex _overleaf/generated/
cp figures/fig_budget.pdf figures/fig_wins.pdf figures/fig_geometry.pdf figures/fig_coverage.pdf _overleaf/figures/
cat > _overleaf/README.txt <<'TXT'
Where Do Scarce Labels Belong? A Compute-Matched Study of Supervised and Self-Supervised Contrastive Pretraining

arxiv.tex         preprint version (compact single column)            -> set as main document for the arXiv PDF
submission.tex    journal version for Neural Networks (review layout)  -> set as main document for the journal PDF
highlights.tex    Elsevier highlights file
cover_letter.tex  cover letter to the editor

Both versions share all text files (sec_*.tex, abstract.tex, appendix.tex, statements.tex). The numbers, tables
and figures in generated/ and figures/ are produced by scripts/make_assets.py in the code repository; edit the
text files, not generated/. Code: https://github.com/MahmoudAlyosifySite/Mahmoud-Project/tree/main/SimCLR-Vision-SSL
TXT
# check that both versions build on their own, as Overleaf will build them
(cd _overleaf && for f in arxiv submission; do pdflatex -interaction=nonstopmode $f.tex > /dev/null; bibtex $f > /dev/null;
   pdflatex -interaction=nonstopmode $f.tex > /dev/null; pdflatex -interaction=nonstopmode $f.tex > /dev/null;
   echo "$f standalone: $(grep -c '^!' $f.log) errors, $(grep -c 'undefined' $f.log) undefined"; done;
   rm -f *.aux *.log *.out *.pdf *.spl *.bbl *.blg)
rm -f overleaf_project.zip && python3 -c "import shutil; shutil.make_archive('overleaf_project', 'zip', '_overleaf')"
rm -rf _overleaf
echo "wrote paper/overleaf_project.zip ($(du -h overleaf_project.zip | cut -f1))"
