# BA paper — internal review (2026-11-24)

A reviewer-style reading of the draft after step A finished, what was changed because of it, and what still stands
between this paper and acceptance at a strong venue. Scores are this review's judgement, on the scale a programme
committee uses (1 = reject, 5 = strong accept), not a promise.

## 1. Verdict

| Version | Soundness | Novelty | Clarity | Reproducibility | Overall |
|---|---|---|---|---|---|
| Draft of 13:55 (first full draft) | 2.5 | 3.5 | 3.5 | 4.5 | 3.0, weak reject: the headline overstated the CER gap, one calibration set was mislabelled |
| arXiv v1 (this revision) | 3.5 | 3.5 | 4.0 | 4.5 | 3.5, borderline/weak accept at a speech venue; a solid preprint |
| v2 with the listening test and volunteer voices | 4.0–4.5 | 4.0 | 4.0 | 4.5 | 4.0+, competitive at Interspeech / SSW / a journal |

The finding is real and useful, the protocol and the open, script-generated numbers are well above the field's norm,
and the claims now match the evidence. What keeps v1 from a strong accept is human evidence: one listener (the author),
six sentences per system, and FLEURS read-MSA references. No amount of automatic analysis replaces native listeners
for a paper whose subject is how a dialect sounds.

## 2. Problems found, and what was done

| # | Problem (as a reviewer would write it) | Severity | Fix in v1 |
|---|---|---|---|
| 1 | NileTTS is called "real colloquial Egyptian speech". Its paper states the audio was generated with NotebookLM. | Factual error | Called synthetic everywhere; real-speech reference is lahgtna-v2 only; its undocumented provenance is stated; the G3 bar was already lahgtna-only. |
| 2 | "A model with almost twice the CER" is presented as a difference. Over three seeds VoxCPM2 − Chatterbox V3 CER = +0.014 [−0.015, +0.044]; the bake-off intervals overlap. | Overclaim | Reframed: the standard metrics cannot tell the two apart (CER, ECAPA) or prefer the non-Egyptian one (DNSMOS), while the dialect share differs by 58 points (Holm p < 0.001). |
| 3 | Spearman ρ ≈ 0 over nine systems is read as "CER tells nothing about dialect". With n = 9 the bootstrap interval is [−0.62, 0.33]. | Statistical overreach | Reported with its interval and explicitly not relied on. New direct test (`selection.py`): over 10,000 two-way bootstrap resamples, selecting among the eligible bases by CER, ECAPA or DNSMOS picks a base below the dialect bar in 93%, 95% and >99% of resamples (classifier 1); the amended rule picks VoxCPM2 in >99%. |
| 4 | Is the dissociation an artefact of read-MSA FLEURS references? | Confound | New control: Egyptian share split by the reference clip's own dialect label. Classifier 1: unchanged (Chatterbox V3 12% / 17%, VoxCPM2 81% / 85%). Classifier 2: the gap narrows with Egyptian references (62% vs 84%). Reported as is; two voices per group. |
| 5 | The listening check is counted as agreeing evidence, but the rater is the author and knew the hypothesis. | Weak evidence presented as support | Labelled "informal", the conflict stated, explicitly not counted as evidence; claims rest on automatic measures. |
| 6 | "Fixed in advance" is asserted, not shown. | Credibility | The gates' commit (`1fb38a0`, 12 Nov 2026) and the results + amendment commit (`f6aab1c`, seven hours later) are cited; classifier 2 and all bootstrap analyses are stated to post-date the amendment. |
| 7 | Long-form: one BA passage has CER 0.49. | Possible system failure | Diagnosed: Whisper's sequential long-form decoding looped ("ايوه" ×100) on the first 25 s; transcribing in pieces returns the text. All 96 long clips of both systems re-scored with ~20 s windows (`long_form_windows.py`): that clip is repaired, but windowed decoding breaks another passage for both systems (Whisper rewrites the colloquial opening into MSA, then loops). Both modes reported; no conclusion drawn on long-form Egyptian CER. Also: neither system failed any passage, so the chunk checks' benefit is not shown, and the paper says so. |
| 8 | Watermark robustness was a placeholder. | Missing result | 24 processing conditions, 60 watermarked + 270 clean clips: 0 false positives in 6,750 tests; survives codecs, telephone band, gain; removed by ±5% speed/tempo (message bits still decode after tempo changes). Full table in the appendix. |
| 9 | DNSMOS favouring Chatterbox V3 was waved away. | Missed point | Now part of the finding: the third standard metric is also blind to the dialect, and prefers the least Egyptian system. |
| 10 | Abstract over arXiv's limit; pending placeholders; appendix empty. | Presentation | Abstract rewritten (checked ≤ 1920 characters by `make_arxiv.py`), no placeholders, appendix holds the full bake-off and watermark tables; missing numbers now fail the build. |
| 11 | Figure 1: overlapping labels, no bar line in panel 2, off-scale systems unmentioned, marker size encodes an informal one-rater score. | Presentation | Redrawn: colour = model family, filled = licence-eligible base, every point labelled, bar and real-speech lines in both panels, off-scale systems named in the caption, no size encoding. |
| 12 | `synth_paper.py` crashed on Arabic text in non-UTF-8 locales (lost the cbv3 seeds run). | Reproducibility | `encoding="utf-8"` on every write. |

## 3. What a strong-venue reviewer will still ask for (in order of value)

1. **Multi-listener blind test** (12–20 native Egyptian listeners; protocol already pre-specified in
   `human_eval/listening_test_protocol.md`): Egyptian authenticity, naturalness, similarity, correctness; real speech as
   hidden anchor; BA, VoxCPM2 plain, Chatterbox V3, NAMAA. Report listener agreement and correlate both classifiers
   with listener authenticity at the clip and system level. This single item moves the paper the most.
2. **Colloquial reference voices from consenting volunteers** (6–10, balanced gender; consent form and sentences in
   `human_eval/`). Re-run the four main systems with them; this answers the reference-dialect question of item 4 above
   and narrows every interval (more voices).
3. **Listener test of the number normaliser and loanword policy** (a small A/B: same sentence with and without), since
   CER cannot see them; the paper currently makes no claim about them, and this would turn a limitation into a result.
4. **A commercial reference point** (optional; only if terms allow publishing results). NileTTS shows a commercial
   generator reaches 98–100% Egyptian under both classifiers; a reviewer may ask how far open systems are from it.
5. **More systems** as they appear (e.g. a new Arabic-capable open model), to show the selection failure is not
   specific to this pool. The paper already says the failure depends on the pool.

## 4. Venue plan

- **arXiv v1 now** (cs.CL, cross-list eess.AS and cs.SD), with the honest one-rater wording. Package:
  `python3 BA/paper/scripts/make_arxiv.py` → `BA/paper/arxiv/ba-arxiv-src.tar.gz` + `metadata.txt`.
- **v2 after items 1–2**, then submit to one of (check each call for current dates, page limits and anonymity rules
  before formatting):
  - Interspeech 2027 (4 pages + references; anonymous; the strongest fit for an evaluation finding),
  - the ISCA Speech Synthesis Workshop (SSW) 2027 (synthesis evaluation is its core topic),
  - ACL Rolling Review (dialectal NLP angle; cs.CL audience),
  - a journal with no page pressure: Computer Speech & Language (Elsevier; this `elsarticle` source fits as is) or
    IEEE/ACM TASLP.
- Conference versions must be anonymised (no author block, no repository URL, anonymous mirror for code), and the
  preprint must be allowed by the venue's policy at the time; both are routine but must be checked per venue.

## 5. Rules that stay

Every number from `BA/eval/results/` through `make_assets.py`; no third-party voices or clones published; every
citation checked against its source; gates and the amendment reported as they happened; the human evidence described
exactly as it was collected.
