#!/bin/bash
# Builds main.pdf. Usage: bash build.sh   (run scripts/make_assets.py first after any results change)
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main.tex > /dev/null || true
bibtex main > /dev/null || true
pdflatex -interaction=nonstopmode main.tex > /dev/null || true
pdflatex -interaction=nonstopmode main.tex > /dev/null || true
echo "main: $(grep -c '^!' main.log) errors, $(grep -c 'undefined' main.log) undefined, pages $(grep -o 'Output written on .*' main.log | grep -o '([0-9]* pages' | tr -d '(')"
grep -o 'R@[^ ]*' main.log | sort -u | head || true
