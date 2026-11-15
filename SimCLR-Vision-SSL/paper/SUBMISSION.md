# Submitting the paper

Both versions are built from the same source (`body.tex` and the `sec_*.tex` files):

| file | purpose | authors shown |
|------|---------|---------------|
| `arxiv.tex` -> `arxiv.pdf` | arXiv preprint (TMLR style with the `preprint` option, no TMLR header) | yes, with ORCID links |
| `tmlr_submission.tex` -> `tmlr_submission.pdf` | anonymous submission to TMLR (official `tmlr.sty`) | no |

Build both with `bash build.sh` (needs a TeX Live with `lmodern`, `orcidlink`, `cleveref`).
All numbers, tables and figures come from `scripts/make_assets.py`; never edit `generated/` by hand.

## arXiv

1. `bash make_arxiv_package.sh` creates `arxiv_submission.tar.gz` (sources + `.bbl` + figures). It also
   compiles the package on its own to check that nothing is missing.
2. On arxiv.org: *Submit* -> upload the tar.gz -> main file `arxiv.tex`.
3. Primary category **cs.LG**, cross-list **cs.CV**.
4. Licence: CC BY 4.0 is the most common choice for ML preprints (allows later publication in TMLR).
5. Title and abstract: copy them from `title.tex` and the abstract in `body.tex` (arXiv's form takes plain text,
   so replace macros such as `\supcon` by "SupCon" and numbers by their values in `arxiv.pdf`).
6. Comments field, for example: "18 pages, 7 figures. Code: https://github.com/MirnaEmbaby/SimCLR-Vision-SSL".

## Recommended venue: TMLR (Transactions on Machine Learning Research)

Why TMLR fits this paper:
* It judges whether the claims are correct and supported by evidence, not whether they are surprising or
  state of the art. A careful empirical study that corrects a negative result is exactly in scope.
* Rolling submission (no deadline), no page limit, open reviews on OpenReview; decisions in about 2-4 months.
* High acceptance rate among papers that pass desk review (about 69% in TMLR's 2024 report).
* Posting the paper on arXiv first is allowed.

How to submit:
1. Create OpenReview accounts for all three authors (use the queensu.ca e-mails).
2. Submit `tmlr_submission.pdf` at https://openreview.net/group?id=TMLR (anonymous; the repository link in the
   anonymous version points to anonymous.4open.science - create an anonymised copy of the repo there and
   replace the URL in `preamble.tex` (`\repo`) before submitting).
3. After acceptance, switch to `\usepackage[accepted]{tmlr}` and fill `\month`, `\year`, `\openreview`.

Other options (in order of fit):
* **A workshop at ICLR 2027** (contributions due around 1 Feb 2027; workshops on 29-30 April 2027 in San
  Francisco). Workshop papers are usually non-archival, so they can be presented while the TMLR review runs.
  Check each workshop's call: look for workshops on representation learning, data-centric ML or evaluation.
* **BMVC 2027** (deadline expected around May 2027; acceptance rate historically 30-38%). Needs reformatting to
  the BMVC template and a 9-page limit.
* **CRV 2027** (Canadian Conference on Robots and Vision, deadline usually in February) - a good Canadian venue
  for a Queen's University paper; check computerrobotvision.org for the 2027 call.
