#!/bin/bash
# Self-contained source package of the VarDial 2027 submission (for Overleaf or the submission system):
#   bash make_package.sh   ->   vardial2027_source.zip and package/ (builds it to check)
set -e
cd "$(dirname "$0")"
rm -rf package && mkdir -p package/generated/ar package/figures
cp acl.sty acl_natbib.bst package/
cp ../refs.bib package/
cp ../generated/numbers.tex ../generated/tab_{bakeoff_short,selection,listening,seeds,ablation,calibration,features}.tex package/generated/
cp ../generated/ar/*.pdf package/generated/ar/
cp ../figures/dissociation.pdf package/figures/
sed -e 's#\\graphicspath{{../}}#\\graphicspath{{./}}#' -e 's#\.\./generated/#generated/#g' -e 's#\\bibliography{\.\./refs}#\\bibliography{refs}#' \
    main.tex > package/main.tex
(cd package && { pdflatex -interaction=nonstopmode main.tex > /dev/null; bibtex main > /dev/null; \
 pdflatex -interaction=nonstopmode main.tex > /dev/null; pdflatex -interaction=nonstopmode main.tex > /dev/null; } || true; \
 echo "package: $(grep -c '^!' main.log) errors, pages $(grep -o 'Output written on .*' main.log | grep -o '([0-9]* pages' | tr -d '(')")
rm -f vardial2027_source.zip
python3 -c "import shutil; shutil.make_archive('vardial2027_source', 'zip', 'package', '.')"
echo "wrote vardial2027_source.zip ($(du -h vardial2027_source.zip | cut -f1))"
