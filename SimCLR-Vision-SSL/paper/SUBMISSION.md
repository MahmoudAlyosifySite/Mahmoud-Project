# Submitting the paper

Title: **Where Do Scarce Labels Belong? A Compute-Matched Study of Supervised and Self-Supervised Contrastive Pretraining**

All versions are built from the same source (`frontmatter.tex`, `abstract.tex`, `body.tex`, the `sec_*.tex` files,
`statements.tex`, `appendix.tex`) with the Elsevier `elsarticle` class.

| file | purpose |
|------|---------|
| `arxiv.tex` -> `arxiv.pdf` | preprint: compact single-column layout, authors with ORCID links |
| `submission.tex` -> `submission.pdf` | journal submission: Elsevier review layout (double spacing, line numbers) |
| `highlights.tex` -> `highlights.pdf` | the separate "Highlights" file that Elsevier journals ask for |
| `cover_letter.tex` -> `cover_letter.pdf` | cover letter to the editor of Neural Networks (build: `pdflatex cover_letter.tex`) |
| `ARXIV_METADATA.md` | title, abstract (plain text), comments and categories for the arXiv form |
| `old_tmlr/` | the first version of the paper (TMLR template), kept for reference |

Build with `bash build.sh`. All numbers, tables and figures come from `scripts/make_assets.py` (`\R{...}` macros in
`generated/numbers.tex`); never edit `generated/` by hand. Text in gray italics in a draft build depends on runs
that are still in the queue (`\pending{...}`); none may remain in a submitted version.

## Before submitting: things only the authors can decide

1. **Author contributions (CRediT).** `statements.tex` contains a proposed split of roles. Check and correct it.
2. **Funding and competing interests.** `statements.tex` uses the standard "no specific grant" and "no competing
   interests" sentences. Replace them if they are not accurate.
3. **Declaration of generative AI.** Added at the end of `statements.tex`, immediately before the references, with
   the section title and the standard wording of Elsevier's policy. Check that the stated use matches how the
   tool was used, and adapt it if the chosen journal (or Springer, which wants it in the Methods section) asks
   for something else.
4. **Anonymity.** If the journal is double-anonymised, set `\anontrue` in `submission.tex` (this hides the authors,
   the CRediT section and the repository link) and prepare an anonymised copy of the repository.
5. **Code archive.** Zenodo record 24494138 (DOI 10.5281/zenodo.24494138, reserved; the draft holds the code, results,
   paper and all checkpoints). Publish it on zenodo.org once the authors agree; the DOI is already in `statements.tex`.

## Journal

Current choice: **Neural Networks** (Elsevier), with **Pattern Recognition** as the second choice; both use the same
`elsarticle` class, so switching is one line (`\journal{...}` in `submission.tex`). Neural Networks fits a study of
learning objectives, representation geometry (neural collapse) and evaluation protocols, and has no page range that this
paper violates as far as we could check. Pattern Recognition (the supervisor's example) requires 20-35 single-column,
double-spaced pages; the review layout of this paper has about 30 pages of text and 9 of references before the
appendix, so it would need cutting, with the appendix moved to supplementary material.
Official acceptance rates are on each journal's ScienceDirect "Insights" page (open it in a browser).

Quartile (checked 2026-11-15 from public sources only): both journals are listed as **Q1** in Web of Science's
"Computer Science, Artificial Intelligence" category (JCR 2023, as reported by Peeref) and as Q1 in Scopus-based
rankings; impact factors on the Elsevier pages are 7.2 (Neural Networks) and 9.1 (Pattern Recognition). The JCR
itself needs an institutional login: Queen's Library -> Databases -> Journal Citation Reports -> search the ISSN
(Neural Networks 0893-6080, Pattern Recognition 0031-3203) -> "Rank by JIF" shows the category quartile.


The supervisor asked for a journal template and advised against TPAMI, IJCV and JMLR. The manuscript is prepared for
an Elsevier journal (`\journal{Neural Networks}` in `submission.tex`; the same class serves Neurocomputing,
Computer Vision and Image Understanding and Image and Vision Computing). Requirements that we could
verify on official pages, and those that still need a check in a browser, are listed in `JOURNAL_NOTES.md` in this
folder. Checked against the Neural Networks Guide for authors (2026-11-24, via search results): abstract at most
250 words (ours: 238), 1-6 keywords (ours: 6), highlights optional, 3-5 bullets of at most 85 characters (ours: 5,
at most 80). Still to confirm in the submission system: review model (single- or double-anonymised), length
limit, reference style.
*Machine Learning* (Springer) is a good alternative for an empirical-methodology paper; it uses the Springer Nature
`sn-jnl` class, 150-250-word abstracts and a "Statements and Declarations" section. Switching class only requires a new
wrapper file next to `submission.tex`.

## arXiv

1. `bash make_arxiv_package.sh` creates `arxiv_submission.tar.gz` (sources, `.bbl`, figures) and checks that the
   package compiles on its own.
2. On arxiv.org: *Submit* -> upload the tar.gz -> main file `arxiv.tex`.
3. Primary category **cs.LG**, cross-list **cs.CV**.
4. Title and abstract: copy them from `arxiv.pdf` (arXiv's form takes plain text).
5. Keep the arXiv version identical to the submitted version, or update it when the submission changes.

## Bibliography

Entries added in the second revision were checked against the publishers' pages (PMLR, NeurIPS proceedings, CVF open
access, iclr.cc, arXiv). Not checked against a primary source: pages of ICLR papers (omitted), Holm (1979) and
Schuirmann (1987) (consistent secondary sources), Le & Yang (2015), the venue of Inoue & Goto (APSIPA ASC 2020), and
the published venues of the three arXiv entries cited only in the budget audit.
