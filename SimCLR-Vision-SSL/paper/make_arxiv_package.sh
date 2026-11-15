#!/bin/bash
# Creates arxiv_submission.tar.gz, ready to upload to arXiv (main file: arxiv.tex).
set -e
cd "$(dirname "$0")"
bash build.sh > /dev/null
rm -rf _arxiv && mkdir -p _arxiv/generated _arxiv/figures
cp arxiv.tex preamble.tex title.tex body.tex sec_*.tex appendix.tex tmlr.sty fancyhdr.sty tmlr.bst \
   math_commands.tex references.bib arxiv.bbl _arxiv/
cp generated/*.tex _arxiv/generated/
cp figures/*.pdf _arxiv/figures/
# arXiv compiles with its own TeX Live; check that the package builds on its own
(cd _arxiv && pdflatex -interaction=nonstopmode arxiv.tex > /dev/null && pdflatex -interaction=nonstopmode arxiv.tex > /dev/null \
   && grep -c "^!" arxiv.log | xargs -I{} echo "standalone build: {} errors" && rm -f arxiv.aux arxiv.log arxiv.out arxiv.pdf)
tar -czf arxiv_submission.tar.gz -C _arxiv .
rm -rf _arxiv
echo "wrote paper/arxiv_submission.tar.gz ($(du -h arxiv_submission.tar.gz | cut -f1))"
