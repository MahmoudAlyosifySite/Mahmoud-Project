#!/bin/bash
# Builds arxiv.pdf and tmlr_submission.pdf.  Usage: bash build.sh
set -e
cd "$(dirname "$0")"
for f in arxiv tmlr_submission; do
  pdflatex -interaction=nonstopmode $f.tex > /dev/null || true
  bibtex $f > /dev/null || true
  pdflatex -interaction=nonstopmode $f.tex > /dev/null || true
  pdflatex -interaction=nonstopmode $f.tex > /dev/null || true
  echo "$f: $(grep -c '^!' $f.log) errors, $(grep -c 'undefined' $f.log) undefined refs/cites"
done
