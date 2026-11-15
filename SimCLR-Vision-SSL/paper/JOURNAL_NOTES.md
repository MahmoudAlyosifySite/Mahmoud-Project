# Journal requirements for the compute-matched contrastive-pretraining paper

Status: PARTIAL (updated 2026-11-15). Elsevier Guide-for-Authors pages on sciencedirect.com return HTTP 403 (Cloudflare) for automated requests, and the cached pr_gfa.html is only a Cloudflare error page. So every Elsevier guide-level item is UNVERIFIED. Verified: Elsevier shop pages (journal scope, page limit for PR, metrics), Elsevier generic AI policy and preprint policy, and the Springer Machine Learning pages.

Scope: five journals: Pattern Recognition, Neural Networks, Neurocomputing,
Computer Vision and Image Understanding (all Elsevier) and Machine Learning (Springer).
IEEE TNNLS and TMLR were dropped from the scope and are NOT covered.

Rule used: a requirement is only stated if it was seen on an official page (URL given). Anything else is marked UNVERIFIED.

## 0. LaTeX templates downloaded

Templates were downloaded from the sources below; `elsarticle.cls` and the `.bst` files used by the paper are in this folder.

| Template | Source (HTTP 200) | Key files |
|---|---|---|
| elsarticle | https://mirrors.ctan.org/macros/latex/contrib/elsarticle.zip | `elsarticle/elsarticle/elsarticle.cls` (NOT in the zip; generated with `latex elsarticle.ins`), `elsarticle-num.bst`, `elsarticle-num-names.bst`, `elsarticle-harv.bst`, `elsarticle-template-num.tex`, `elsarticle-template-num-names.tex`, `elsarticle-template-harv.tex`, `doc/elsdoc.pdf` |
| IEEEtran | https://mirrors.ctan.org/macros/latex/contrib/IEEEtran.zip | `IEEEtran/IEEEtran/IEEEtran.cls`, `bibtex/IEEEtran.bst`, `bare_jrnl.tex`, `IEEEtran_HOWTO.pdf` |
| Springer Nature (sn-jnl) | CTAN URL https://mirrors.ctan.org/macros/latex/contrib/sn-jnl.zip returned HTTP 404 (not on CTAN). Downloaded instead from the official Springer Nature link (December 2024 version) https://resource-cms.springernature.com/springer-cms/rest/v1/content/18782940/data/v12 , which is linked from https://www.springernature.com/gp/authors/campaigns/latex-author-support | `sn-jnl/sn-article-template/sn-jnl.cls`, `sn-article.tex`, `bst/sn-basic.bst`, `bst/sn-mathphys-num.bst`, `bst/sn-mathphys-ay.bst` (and 6 other .bst), `user-manual.pdf` |

## 1. Pattern Recognition (Elsevier)
- Official page seen: https://shop.elsevier.com/journals/pattern-recognition/0031-3203 (Elsevier shop; journal description + metrics). Guide URL (not readable here): https://www.sciencedirect.com/journal/pattern-recognition/publish/guide-for-authors
- Metrics as shown on the Elsevier shop page (retrieved 2026-11-15; year of the figures not stated on the page): Impact factor 9.1; 5-year impact factor 8.8. CiteScore: UNVERIFIED (not shown on that page).
- Scope text (verbatim essentials): "original contributions to the theory, methodology and application of pattern recognition in any area, provided that the context of the work is both clearly explained and grounded in the pattern recognition literature. Papers whose primary concern falls outside the pattern recognition domain and which report routine applications of it using existing or well known methods, should be directed elsewhere."
- Length (VERIFIED on the shop page): "Submitted papers should be single column, double spaced, no less than 20 and no more than 35 (40 for a review) pages long, with numbered pages." A 16-page paper is therefore BELOW the stated 20-page minimum; either expand (single column, double spaced counts more pages) or ask the editor. In double-spaced single-column format a 16-page two-column paper typically exceeds 20 pages (inference, not stated).
- Fit warning: scope says routine application of existing methods is redirected elsewhere; an empirical-study paper must be framed as methodology/evaluation insight.
- Review model (single vs double anonymized): UNVERIFIED. Elsevier journals are single-anonymized by default, but this journal's guide must be checked; do not assume.
- Abstract limit, keywords, highlights, graphical abstract, figure/reference limits: UNVERIFIED. (Elsevier general convention is highlights + keywords, but not confirmed for this journal.)
- Required statements (data availability, CRediT, competing interests, funding): UNVERIFIED for this journal. AI declaration: see the Elsevier generic policy in section 7 (verified).
- LaTeX/reference style: UNVERIFIED for the journal; generic elsarticle class is downloaded (section 0): elsarticle-num.bst (numbered), elsarticle-num-names.bst, elsarticle-harv.bst (author-year).
- Preprint: Elsevier generic policy (section 7) allows sharing preprints; journal-specific exceptions UNVERIFIED.
- APC / no-fee route: UNVERIFIED for the exact figure; Elsevier hybrid journals normally offer a free subscription route (UNVERIFIED here). Time to first decision: UNVERIFIED.

## 2. Neural Networks (Elsevier)
- Official page seen: https://shop.elsevier.com/journals/neural-networks/0893-6080 (Elsevier shop; journal description + metrics). Guide URL (not readable here): https://www.sciencedirect.com/journal/neural-networks/publish/guide-for-authors
- Metrics as shown on the Elsevier shop page (retrieved 2026-11-15; year of the figures not stated on the page): Impact factor 7.2; 5-year impact factor 7.4. CiteScore: UNVERIFIED (not shown on that page).
- Scope text (verbatim essentials): "all aspects of neural networks, including deep learning and related approaches to artificial intelligence and machine learning"; five sections including "learning systems"; archival journal of INNS, APNNS, JNNS.
- Review model (single vs double anonymized): UNVERIFIED. Elsevier journals are single-anonymized by default, but this journal's guide must be checked; do not assume.
- Abstract limit, keywords, highlights, graphical abstract, figure/reference limits: UNVERIFIED. (Elsevier general convention is highlights + keywords, but not confirmed for this journal.)
- Required statements (data availability, CRediT, competing interests, funding): UNVERIFIED for this journal. AI declaration: see the Elsevier generic policy in section 7 (verified).
- LaTeX/reference style: UNVERIFIED for the journal; generic elsarticle class is downloaded (section 0): elsarticle-num.bst (numbered), elsarticle-num-names.bst, elsarticle-harv.bst (author-year).
- Preprint: Elsevier generic policy (section 7) allows sharing preprints; journal-specific exceptions UNVERIFIED.
- APC / no-fee route: UNVERIFIED for the exact figure; Elsevier hybrid journals normally offer a free subscription route (UNVERIFIED here). Time to first decision: UNVERIFIED.

## 3. Neurocomputing (Elsevier)
- Official page seen: https://shop.elsevier.com/journals/neurocomputing/0925-2312 (Elsevier shop; journal description + metrics). Guide URL (not readable here): https://www.sciencedirect.com/journal/neurocomputing/publish/guide-for-authors
- Metrics as shown on the Elsevier shop page (retrieved 2026-11-15; year of the figures not stated on the page): Impact factor 6.7; 5-year impact factor 6.4. CiteScore: UNVERIFIED (not shown on that page).
- Scope text (verbatim essentials): "theoretical contributions aimed at winning further understanding of neural networks and learning systems ... applications in ... computer vision"; also a Software Track.
- Review model (single vs double anonymized): UNVERIFIED. Elsevier journals are single-anonymized by default, but this journal's guide must be checked; do not assume.
- Abstract limit, keywords, highlights, graphical abstract, figure/reference limits: UNVERIFIED. (Elsevier general convention is highlights + keywords, but not confirmed for this journal.)
- Required statements (data availability, CRediT, competing interests, funding): UNVERIFIED for this journal. AI declaration: see the Elsevier generic policy in section 7 (verified).
- LaTeX/reference style: UNVERIFIED for the journal; generic elsarticle class is downloaded (section 0): elsarticle-num.bst (numbered), elsarticle-num-names.bst, elsarticle-harv.bst (author-year).
- Preprint: Elsevier generic policy (section 7) allows sharing preprints; journal-specific exceptions UNVERIFIED.
- APC / no-fee route: UNVERIFIED for the exact figure; Elsevier hybrid journals normally offer a free subscription route (UNVERIFIED here). Time to first decision: UNVERIFIED.

## 4. Computer Vision and Image Understanding (Elsevier)
- Official page seen: https://shop.elsevier.com/journals/computer-vision-and-image-understanding/1077-3142 (Elsevier shop; journal description + metrics). Guide URL (not readable here): https://www.sciencedirect.com/journal/computer-vision-and-image-understanding/publish/guide-for-authors
- Metrics as shown on the Elsevier shop page (retrieved 2026-11-15; year of the figures not stated on the page): Impact factor 3.6; 5-year impact factor 4.3. CiteScore: UNVERIFIED (not shown on that page).
- Scope text (verbatim essentials): "computer analysis of pictorial information ... from low-level ... to high-level ... recognition and interpretation".
- The shop page says submissions "should be sent using one of these templates latex/work" (link to the guide, section #writing-and-formatting), i.e. a LaTeX template exists (VERIFIED wording; template identity UNVERIFIED).
- Review model (single vs double anonymized): UNVERIFIED. Elsevier journals are single-anonymized by default, but this journal's guide must be checked; do not assume.
- Abstract limit, keywords, highlights, graphical abstract, figure/reference limits: UNVERIFIED. (Elsevier general convention is highlights + keywords, but not confirmed for this journal.)
- Required statements (data availability, CRediT, competing interests, funding): UNVERIFIED for this journal. AI declaration: see the Elsevier generic policy in section 7 (verified).
- LaTeX/reference style: UNVERIFIED for the journal; generic elsarticle class is downloaded (section 0): elsarticle-num.bst (numbered), elsarticle-num-names.bst, elsarticle-harv.bst (author-year).
- Preprint: Elsevier generic policy (section 7) allows sharing preprints; journal-specific exceptions UNVERIFIED.
- APC / no-fee route: UNVERIFIED for the exact figure; Elsevier hybrid journals normally offer a free subscription route (UNVERIFIED here). Time to first decision: UNVERIFIED.

## 5. Pattern Recognition Letters (Elsevier)
- Official page seen: https://shop.elsevier.com/journals/pattern-recognition-letters/0167-8655 (Elsevier shop; journal description + metrics). Guide URL (not readable here): https://www.sciencedirect.com/journal/pattern-recognition-letters/publish/guide-for-authors
- Metrics as shown on the Elsevier shop page (retrieved 2026-11-15; year of the figures not stated on the page): Impact factor 3.5; 5-year impact factor 3.7. CiteScore: UNVERIFIED (not shown on that page).
- Scope text (verbatim essentials): "rapid publication of concise articles"; research reports must be concise (so a 16-page study is a poor fit; length limit UNVERIFIED).
- Review model (single vs double anonymized): UNVERIFIED. Elsevier journals are single-anonymized by default, but this journal's guide must be checked; do not assume.
- Abstract limit, keywords, highlights, graphical abstract, figure/reference limits: UNVERIFIED. (Elsevier general convention is highlights + keywords, but not confirmed for this journal.)
- Required statements (data availability, CRediT, competing interests, funding): UNVERIFIED for this journal. AI declaration: see the Elsevier generic policy in section 7 (verified).
- LaTeX/reference style: UNVERIFIED for the journal; generic elsarticle class is downloaded (section 0): elsarticle-num.bst (numbered), elsarticle-num-names.bst, elsarticle-harv.bst (author-year).
- Preprint: Elsevier generic policy (section 7) allows sharing preprints; journal-specific exceptions UNVERIFIED.
- APC / no-fee route: UNVERIFIED for the exact figure; Elsevier hybrid journals normally offer a free subscription route (UNVERIFIED here). Time to first decision: UNVERIFIED.

## 6. Image and Vision Computing (Elsevier)
- Official page seen: https://shop.elsevier.com/journals/image-and-vision-computing/0262-8856 (Elsevier shop; journal description + metrics). Guide URL (not readable here): https://www.sciencedirect.com/journal/image-and-vision-computing/publish/guide-for-authors
- Metrics as shown on the Elsevier shop page (retrieved 2026-11-15; year of the figures not stated on the page): Impact factor 5; 5-year impact factor 4.6. CiteScore: UNVERIFIED (not shown on that page).
- Scope text (verbatim essentials): "encouraging the quantitative comparison and performance evaluation of the proposed methodology" (good match for an evaluation study).
- Review model (single vs double anonymized): UNVERIFIED. Elsevier journals are single-anonymized by default, but this journal's guide must be checked; do not assume.
- Abstract limit, keywords, highlights, graphical abstract, figure/reference limits: UNVERIFIED. (Elsevier general convention is highlights + keywords, but not confirmed for this journal.)
- Required statements (data availability, CRediT, competing interests, funding): UNVERIFIED for this journal. AI declaration: see the Elsevier generic policy in section 7 (verified).
- LaTeX/reference style: UNVERIFIED for the journal; generic elsarticle class is downloaded (section 0): elsarticle-num.bst (numbered), elsarticle-num-names.bst, elsarticle-harv.bst (author-year).
- Preprint: Elsevier generic policy (section 7) allows sharing preprints; journal-specific exceptions UNVERIFIED.
- APC / no-fee route: UNVERIFIED for the exact figure; Elsevier hybrid journals normally offer a free subscription route (UNVERIFIED here). Time to first decision: UNVERIFIED.

## 7. Machine Learning (Springer) - mostly VERIFIED
Sources (official, retrieved 2026-11-15, cached copies in pages/): https://link.springer.com/journal/10994 , /submission-guidelines , /how-to-publish-with-us , /aims-and-scope.
- Metrics (journal home page): Journal Impact Factor 4.9 (2025); 5-year JIF 8.0 (2025); "Submission to first decision (median) 5 days"; CiteScore not shown.
- Time to first decision (guidelines): "In most cases, the decision is made in less than three months." Online publication 3-5 weeks after acceptance.
- Scope: "Research methodology papers improve how machine learning research is conducted"; papers making "claims about ... relative performance of alternative algorithms" must give "solid support via empirical studies". Strong fit for an evaluation-methodology paper.
- Abstract: "150 to 250 words". Keywords: "4 to 6 keywords". Headings: "decimal system of headings with no more than three levels". No page/word limit found in the guidelines text (UNVERIFIED that none exists).
- Required: section "Statements and Declarations" (competing interests; submissions without relevant declarations "will be returned as incomplete"); Data Availability Statement is mandatory for original research ("All original research must include a data availability statement"). CRediT: not seen in cached text (UNVERIFIED). Funding: covered under "Competing Interests and Funding" (details not read).
- AI use (Springer ML guidelines, quoted): "Use of an LLM should be properly documented in the Methods section (and if a Methods section is not available, in a suitable alternative part) of the manuscript. The use of an LLM (or other AI-tool) for 'AI assisted copy editing' purposes does not need to be declared." LLMs cannot be authors.
- Template: Springer Nature LaTeX template ("We recommend using Springer Nature's LaTeX template"); submit LaTeX source plus compiled PDF, no subfolders. Template downloaded (section 0). References: "Cite references in the text by name and year in parentheses", APA 7 style, alphabetised, DOIs as full links -> use sn-apa.bst-type author-year style from the sn-jnl package (sn-mathphys-ay.bst is the downloaded author-year option; the exact bst Springer wants for ML is UNVERIFIED, the class's own "sn-apacite"/APA option is the safest choice).
- Review model: the guidelines contain only a conditional sentence about "If the journal ... uses double-anonymous peer review"; the journal's own model is NOT stated on the pages read -> UNVERIFIED. Must be checked in the Editorial Manager submission form before submitting (it decides anonymisation).
- Preprint: the Machine Learning page does not state it; Springer Nature general policy (https://www.springernature.com/gp/policies/editorial-policies) "encourages posting preprints" but does not mention arXiv timing. ML guidelines require the work "is not under consideration for publication anywhere else" (a preprint is not peer-review consideration, but confirm with the editor).
- Overlap rule: extending a conference paper is acceptable with a significant contribution beyond it, described at the start.
- Fees (VERIFIED): hybrid. Open access APC "GBP 2490 / USD 3390 / EUR 2790" (plus VAT); "Authors can also choose to publish under the subscription publishing model (no APC charges apply)"; choice is offered after acceptance. OA licence CC BY or CC BY-NC-ND.

## 8. Generic Elsevier policies (VERIFIED)
- AI declaration, https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals : "Authors should disclose the use of AI tools for manuscript preparation in a separate AI declaration statement". Section title: "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process". Statement: "During the preparation of this work, the author(s) used [NAME OF TOOL / SERVICE] in order to [REASON]. After using this tool/service, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the published article." Placed "at the end of your manuscript, immediately before the references". "Basic checks of grammar, spelling and punctuation do not need a declaration statement." AI cannot be listed as author.
- Preprints, https://www.elsevier.com/about/policies-and-standards/sharing : "Authors can share their preprint anywhere at any time." Caveat on the page: some society-owned and double-anonymized journals have different rules; check the journal's guide.

## 9. Not researched
IEEE TNNLS, TMLR, Neural Processing Letters, Machine Vision and Applications: not fetched (no time before the limit reset); nothing is claimed about them. Background knowledge only (UNVERIFIED): TMLR is double-blind OpenReview with no page limit, free, arXiv allowed; TNNLS is IEEEtran, high bar.

## 10. Ranking (top 3), judgement based on the verified material plus UNVERIFIED review models
1. Machine Learning (Springer). Best scope fit: its aims explicitly include research-methodology and empirical-comparison papers; the 16-page length fits (no minimum seen); abstract 150-250 words; free subscription route (no APC); median first decision shown as 5 days (<3 months stated); JIF 4.9 / 5-year 8.0. Risks: review model unverified (may be double-anonymous, then anonymise and be careful with the arXiv link), author-year references, bar for novelty is real.
2. Neural Networks (Elsevier). Scope explicitly covers deep-learning learning systems; IF 7.2; but selective, and its Guide for Authors is unverified here, so check review model, limits and statements first.
3. Image and Vision Computing (Elsevier). Its scope text stresses "quantitative comparison and performance evaluation"; IF 5; more realistic acceptance than Pattern Recognition for a CIFAR/STL study. Pattern Recognition (the supervisor's example, IF 9.1) is a harder target: it requires 20-35 double-spaced pages and warns against routine applications of known methods, so it is a stretch unless the paper is framed with a methodological contribution.
Cheap, fast fallbacks: Neurocomputing, Pattern Recognition Letters (too short-form for 16 pages), TMLR (unverified).
Next step for the user: open each ScienceDirect guide in a browser (blocked here) and fill the UNVERIFIED review-model/abstract/limit items before choosing between Machine Learning and the Elsevier journals.
