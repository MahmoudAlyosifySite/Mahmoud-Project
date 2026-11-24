# Novelty check (working notes, in progress)

Question: has prior work shown that ASR-based intelligibility (WER/CER) fails to reflect dialect authenticity in dialectal or accented TTS, or that it selects the wrong model?

Findings so far (all read in full text on 2026-11-15):
- Habibi (arXiv 2601.13802): WER-O vs WER-S rationale; "multilingual model risks cross-dialect recognition bias (e.g., incorrectly 'correcting' speech that lacks certain dialect features)"; human DMOS; Tables 4-5 show EGY: Uni.D2-I WER-O 14.96 / WER-S 15.79 vs 11Labs-3a 17.60 / 19.27, but DMOS 3.92 vs 4.21. Not discussed by authors.
- NileTTS (ACL abjadnlp 2026): WER/CER (Whisper Large) + ECAPA sim; no MOS, no dialect metric; informal listening of 50 utterances; notes ASR self-consistency bias.
- Doan et al. 2024 (arXiv 2406.16751): baseline XTTS lowest WER; fine-tunes higher WER but "capable of generating dialectal speech"; no dialect metric; interpreted WER rise as model failure.
- Voxlect (arXiv 2508.01691): dialect classifier as TTS evaluation tool (Mandarin, CosyVoice 2), agrees with native ratings; no WER comparison.
- Vox-Profile (arXiv 2505.14648): accent classifier to evaluate VC/TTS accent preservation (FreeVC vs VALL-E X); no WER comparison.
- To check: Teochew SSW 2025 (Wallet et al.); Issa et al. 2026 (Procedia CS); AccentBox; arXiv/ACL/Interspeech searches.
