# BA paper — plan

Written 2026-11-15. Target: an arXiv preprint (free), primary category cs.CL, cross-listed to eess.AS and cs.SD. Single author:

```latex
\author[a]{Mahmoud Alyosify\,\orcidlink{0009-0005-1336-2726}\corref{cor1}}
\cortext[cor1]{Corresponding author: mahmoud.alyosify@queensu.ca}
\affiliation[a]{organization={Queen's University}, city={Kingston}, state={Ontario}, country={Canada}}
```

Class: `elsarticle` (`preprint,12pt` or `5p` two-column), matching the `\author[a]...\corref` syntax. It is not installed on this server; it is on arXiv's TeX Live, and its CTAN files (LPPL) can be placed next to `main.tex` for local builds.

## 0. Status and revised order (2026-11-24)

The A40 is free again (the SimCLR runs finished). `/tmp/ba` was lost in a container restart, so the venvs are rebuilt
first; generated audio in `/home/jovyan/ba-work/` survived.

| Item | State |
|---|---|
| Second dialect classifier on the bake-off, BA and calibration speech | **Done** (`dialect_classifier2.md`): classifiers agree on ranking (Spearman 0.85), kappa 0.64 on real speech |
| Uncertainty analysis with two-way bootstrap, paired tests, Holm | **Done for seed 0**; seeds 1-2 not generated yet (38 of 180 BA files exist) |
| Seeds 1-2 for BA, raw VoxCPM2, Chatterbox V3, NAMAA | To run (GPU, about 1.5 h) |
| Ablations: no normaliser, no loanword policy, no transcript, raw reference | To run (GPU, about 40 min), then `ablations.py` |
| Long-form set: 24 new passages (8 per language) x 2 voices, BA and raw VoxCPM2 | To run (GPU, about 2-3 h) |
| Watermark under 25 attacks, 60 watermarked + 290 clean clips | To run (CPU/GPU, about 1 h) |
| Literature and novelty check | Half done (`novelty.md`, 45 verified entries in `refs.bib`) |
| Multi-rater listening test, volunteer voices | Needs the owner (section 7) |

Order from here: (1) all automated runs in one queue (`run_paper.sh`), (2) literature check finished and the
related-work section written meanwhile, (3) full draft with every number from results files, (4) listening test
with volunteer voices, (5) final paper and arXiv package. A v1 on arXiv after step 3 is possible, with the one-rater
check reported honestly; the listening test then becomes v2.

**The name, as the owner wants it stated** (introduction, one short paragraph, with an Egyptological citation): in
ancient Egyptian belief, the *ba* is the part of the soul that carries a person's true personality and
individuality; a voice is the most personal sound a person makes, so a system that speaks in someone's own voice
is named after it. The Arabic letter ب is read *ba*, and the mark draws it with a voice wave.

## 1. What the paper claims (and what it does not)

BA v0.1 does not train a new model. It drives the unmodified VoxCPM2 weights with an engine built for Egyptian Arabic, chosen and evaluated by a protocol fixed in advance. A paper that sells BA as a new model would be rejected by any careful reader. The strong, true story is this:

> **Right words, wrong dialect.** For Egyptian Arabic voice cloning, the standard intelligibility metric picks the wrong model. The open model with the lowest character error rate (Chatterbox Multilingual V3, CER 0.035) is heard as not Egyptian: the dialect classifier labels 21% of its Egyptian sentences Egyptian, and a native listener rated it not Egyptian on 6 of 6 sentences. A model with almost twice the CER (VoxCPM2, 0.065) is labelled Egyptian 82% of the time and was never rated not Egyptian. We show this, explain it, build BA on the model a dialect-aware protocol selects, and close most of the remaining gap at inference time, without training.

Contributions, in the order the paper argues them:

1. **The dissociation.** Across 8 open voice-cloning systems, Egyptian CER and Egyptian-dialect authenticity do not agree; CER-based selection fails. Shown with automatic metrics calibrated on real speech, a second dialect classifier, and a multi-rater native listening test.
2. **A dialect-aware evaluation protocol** for dialectal voice cloning: gates fixed before any model ran, every metric calibrated on real speech (same-speaker ceiling, colloquial vs MSA dialect rates, ASR floor), and a public log of the one gate amendment and why it was needed. The amendment is itself evidence for contribution 1.
3. **A licence-provenance audit** of the systems: weights licence, training-data documentation, and inherited restrictions (e.g. an Apache-2.0-labelled fine-tune of CC-BY-NC weights). Most Egyptian-capable open models cannot be built on commercially.
4. **BA, an inference-time engine** (Egyptian number and symbol verbalisation, loanword transliteration, automatic reference transcript, reference screening, long-text chunking with checks and retries, watermarking). Held-out results: Egyptian CER 0.069 → 0.030, dialect share 0.82 → 0.89, similarity 0.71 → 0.77; English similarity rises from failing to passing. Each component's share measured by ablation.
5. **A consent-first release:** required consent in the demo, no storage of uploads, an AudioSeal watermark on every output with measured robustness, and an open model, code and demo.

Things the paper must not claim: a new model or architecture; "first" anything unless the literature check in section 4 supports it; "Egyptian" quality beyond what the listening test shows; commercial readiness.

## 2. Title, name, and the naming paragraph

Working title: **"BA: Right Words, Wrong Dialect — Building and Evaluating Consent-First Voice Cloning for Egyptian Arabic"** (alternative: "Right Words, Wrong Dialect: Why Intelligibility Metrics Cannot Select Egyptian Arabic Voice Cloning, and BA").

The name goes in the introduction, one short paragraph: in ancient Egyptian belief the *ba* is the part of the soul that carries a person's personality and individuality, usually drawn as a bird with a human head; a voice is the most personal sound a person makes; the Arabic letter ب is pronounced *ba*, and the mark draws it with a voice wave. Cite an Egyptological source for the belief (to be found and verified, e.g. Taylor, *Death and Burial in Ancient Egypt*, or Assmann, *Death and Salvation in Ancient Egypt*), not Wikipedia.

## 3. Gaps a reviewer would hit today, and how each is closed

| Gap in the current evidence | Fix | Who / what | Priority |
|---|---|---|---|
| Human judgement is one rater, 6 sentences, raw outputs only | **Multi-rater blind listening test**: 12–20 native Egyptian listeners; per clip: naturalness MOS (1–5), Egyptian authenticity (1–5), speaker similarity to the reference (1–5), all words right (yes/no); systems: BA, raw VoxCPM2, VoxCPM2 + transcript, Chatterbox V3, NAMAA, VoiceTut (reference only), and real speech as a hidden anchor; ~20 min per rater | Owner recruits raters; executor builds the test page | Must |
| Reference voices are FLEURS clips of read MSA | **Colloquial reference voices from consenting volunteers**: 6–10 Egyptian speakers (balanced gender), 30–60 s each of free speech and read sentences, signed consent for research use and publication of their clones | Owner records friends or colleagues with a consent form the executor drafts | Must |
| The dissociation rests on one dialect classifier | Second, independent dialect-ID model; report agreement; correlate both with the listening test | Executor (GPU) | Must |
| No ablation of the BA engine | Remove one component at a time (normaliser, loanword policy, transcript mode, reference preparation, chunk checks) on the held-out set | Executor (GPU) | Must |
| Single seed, no uncertainty | 3 seeds for BA, raw VoxCPM2 and Chatterbox V3; bootstrap 95% CIs over sentences and voices; paired tests for every headline comparison | Executor | Must |
| 40 Egyptian sentences | Add about 60 new held-out colloquial sentences across registers (chat, service, narration, numbers, code-switch), checked by the owner for natural Egyptian | Planner writes, owner checks | Should |
| Watermark tested on WAV, MP3, resampling | Add noise, speed change, band-pass (phone), re-encoding chains, and partial clips | Executor | Should |
| Long-form evidence is 2 passages per language | 10 passages per language, 1–3 minutes each | Executor | Should |
| CER for digits measures the number, not the Egyptian way of saying it | Covered by the listening test's "words right" and authenticity on number sentences | Listening test | Should |
| No comparison with a commercial system | Optional: one commercial voice-cloning API on the same texts, only if its terms allow evaluation and publication of results; otherwise state the omission | Owner decides | Could |

## 4. Literature and novelty check (before writing claims)

Read and cite, with verified bibliographic data only (no reference from memory without checking it exists):
- Zero-shot TTS: VALL-E, NaturalSpeech 3, F5-TTS, CosyVoice, VoxCPM / VoxCPM2, Chatterbox, OmniVoice, Qwen3-TTS.
- Arabic and Egyptian TTS: NileTTS (LLM-to-speech pipeline, Egyptian dataset), Habibi (dialect zero-shot TTS), EGTTS, VoiceTut, NAMAA, lahgtna, Arabic TTS benchmarks.
- Dialect identification for Arabic speech (the classifiers used), MGB-3/5.
- Evaluation: WER/CER limits, UTMOS/DNSMOS, speaker-similarity practice (ECAPA, WavLM-SV), MOS test design (ITU-T P.800), known problems of automatic MOS on non-English speech.
- Watermarking: AudioSeal, WavMark, Perth; consent and misuse of voice cloning; FLEURS.

Novelty question to settle: has anyone shown that ASR-based intelligibility selects the wrong model for dialectal TTS? If yes, the paper confirms it for Egyptian voice cloning and adds the protocol and BA; if no, it says so carefully ("we are not aware of...").

## 5. Paper outline (about 9 pages plus appendix)

1. Introduction: Egyptian Arabic (the most widely understood Arabic dialect; tens of millions of speakers), MSA is not Egyptian, the dissociation in one figure, contributions, the name.
2. Related work.
3. Systems and licence-provenance audit (table).
4. Evaluation protocol: texts, voices, metrics and their real-speech calibration, gates, the amendment.
5. Right words, wrong dialect: bake-off results; CER vs dialect-share scatter with listening-test scores; second classifier; why (accent of the multilingual training data, Levantine/Gulf labels).
6. BA: the engine, component by component.
7. Results: BA vs raw (held-out, with CIs), ablations, listening test, cross-language, long form, speed and memory.
8. Responsible release: consent flow, no storage, watermark robustness, licence terms, what is published and what is not (no third-party voices).
9. Limitations and ethics statement.
10. Conclusion.
Appendices: normaliser rules with examples; full tables; texts; listening-test protocol and rater instructions; consent form; claim-to-results map (every number in the paper → the file in `BA/eval/results/` that produced it), as in the conservation-teacher paper.

Figures: (1) CER vs Egyptian share per system, point size = human authenticity; (2) BA pipeline; (3) ablation bars with CIs; (4) listening-test results per system; (5) watermark detection under attacks. Every figure generated by a script from results files.

## 6. Order of work

| Step | Content | Depends on |
|---|---|---|
| A | Automated experiments: second dialect classifier, ablations, 3 seeds + CIs, watermark attacks, long-form set; results files + scripts | GPU only |
| B | Literature collection and verification; related-work draft; naming source | — |
| C | Paper draft from results files only: all sections except the listening test, with clearly marked placeholders | A, B |
| D | New held-out sentences (owner checks), volunteer recordings with consent, then BA and baselines generated on them | Owner |
| E | Multi-rater listening test (static page, answers collected by a one-field form or message), analysis, inter-rater agreement | D, raters |
| F | Final paper, figures, claim map, arXiv bundle (`make_arxiv.py` adapted), metadata, test build with arXiv's TeX settings | C, E |

A preprint can go out after F. If the owner wants it sooner, a v1 after C (with the one-rater listening check reported honestly and the multi-rater test announced) is possible, and v2 replaces it; arXiv keeps both versions.

## 7. Needed from the owner

1. **Raters:** 12–20 native Egyptian listeners willing to spend about 20 minutes.
2. **Volunteer voices:** 6–10 Egyptian speakers who sign the consent form and record 30–60 s each.
3. **arXiv account and endorsement:** first-time submitters to cs.CL may need an endorsement from an established author; Queen's colleagues (e.g. the SimCLR co-authors) can endorse.
4. **AI-assistance disclosure:** arXiv makes authors responsible for all content and asks that significant use of generative AI be disclosed. Recommendation: one sentence in the acknowledgements stating that AI assistants were used for engineering and drafting, and that the author reviewed and takes responsibility for all content.
5. **Optional:** whether to include a commercial-system comparison.

## 8. Rules

- Numbers only from `BA/eval/results/` files produced by saved scripts; no number typed by hand.
- No third-party voice clips or their clones in the paper or supplementary material; volunteer voices only with signed consent.
- Every citation checked against its actual source.
- The plan's gates and the G1 amendment are reported as they happened.
