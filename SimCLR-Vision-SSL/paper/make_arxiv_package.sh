#!/bin/bash
# Creates arxiv_submission.tar.gz, ready to upload to arXiv (main file: arxiv.tex).
set -e
cd "$(dirname "$0")"
bash build.sh > /dev/null
rm -rf _arxiv && mkdir -p _arxiv/generated _arxiv/figures
cp arxiv.tex preamble.tex frontmatter.tex abstract.tex body.tex sec_*.tex statements.tex appendix.tex \
   elsarticle.cls elsarticle-num-names.bst references.bib arxiv.bbl _arxiv/
cp generated/numbers.tex generated/tab_*.tex _arxiv/generated/
cp figures/fig_budget.pdf figures/fig_wins.pdf figures/fig_geometry.pdf figures/fig_coverage.pdf _arxiv/figures/
# arXiv compiles with its own TeX Live; check that the package builds on its own
(cd _arxiv && pdflatex -interaction=nonstopmode arxiv.tex > /dev/null; pdflatex -interaction=nonstopmode arxiv.tex > /dev/null; \
   echo "standalone build: $(grep -c '^!' arxiv.log) errors"; rm -f arxiv.aux arxiv.log arxiv.out arxiv.pdf arxiv.spl)
tar -czf arxiv_submission.tar.gz -C _arxiv .
rm -rf _arxiv
echo "wrote paper/arxiv_submission.tar.gz ($(du -h arxiv_submission.tar.gz | cut -f1))"
