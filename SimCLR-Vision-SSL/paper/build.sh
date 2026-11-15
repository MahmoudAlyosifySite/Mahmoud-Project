#!/bin/bash
# Builds arxiv.pdf (preprint), submission.pdf (journal review layout) and highlights.pdf.   Usage: bash build.sh
set -e
cd "$(dirname "$0")"
for f in arxiv submission; do
  pdflatex -interaction=nonstopmode $f.tex > /dev/null || true
  bibtex $f > /dev/null || true
  pdflatex -interaction=nonstopmode $f.tex > /dev/null || true
  pdflatex -interaction=nonstopmode $f.tex > /dev/null || true
  echo "$f: $(grep -c '^!' $f.log) errors, $(grep -c 'undefined' $f.log) undefined refs/cites, $(grep -c '??' $f.log || true) log '??', pages $(grep -o 'Output written on .*' $f.log | grep -o '([0-9]* pages' | tr -d '(')"
done
pdflatex -interaction=nonstopmode highlights.tex > /dev/null || true
