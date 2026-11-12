# BA — Execution plan

Written 2026-11-12. This plan is fixed before any model is run, so the gates below cannot be bent to fit the results. Changes go in a dated "Changes" section at the end, with the reason.

## 1. What we are building

| Deliverable | Where | Done when |
|---|---|---|
| BA model | Hugging Face model repo `mahmoudalyosify/BA` | Loads from a clean machine with the documented snippet and produces speech in every language the card lists |
| Inference package | `BA/src/ba_tts/` (also shipped inside the model repo and the Space) | `BA.from_pretrained()`, `create_voice()`, `speak()`, a CLI, unit tests pass |
| Public demo | Hugging Face Space `mahmoudalyosify/BA` | A stranger can upload or record a voice, tick the consent box, type text, pick a language, listen and download |
| Model card and docs | model repo `README.md`, `BA/docs/` | Every number on the card is in `BA/eval/results/`, every listed capability passed its gate |
| Brand | `BA/assets/`, `BA/docs/brand.md` | Logo, banner, social card, avatar, palette, typography, Space theme all from one system |
| Reproducible deployment | `BA/hf/publish.py`, `BA/docs/reproduce.md` | One command rebuilds and uploads the model repo and the Space |

Languages: Egyptian Arabic and English first. French is listed only if it passes the same gate.

## 2. What is already known

**Server.** One NVIDIA A40 (46 GB, idle), 128 cores, 1 TB RAM, CUDA 12.6 wheels work. `/home/jovyan` is a 49 GB persistent volume with 35 GB free. `/tmp` has 390 GB but is lost if the container restarts.

**Credentials.** The `HF_TOKEN` in the server environment belongs to another account (`mirnaembaby`). It is never used. Publishing waits for a write token from `mahmoudalyosify`.

**Licence screening** (weights, not code — the two often differ):

| Model | Weights licence | Arabic | Usable as BA's base? |
|---|---|---|---|
| Chatterbox Multilingual V2 / V3 (Resemble AI, 0.5B) | MIT | yes (`ar`), plus `en`, `fr` | **Yes** |
| VoxCPM2 (OpenBMB, 2B, 48 kHz) | Apache-2.0 | yes, plus `en`, `fr` | **Yes** |
| NAMAA-Egyptian-TTS, oddadmix `lahgtna-chatterbox-v1` | MIT (Chatterbox V2 fine-tunes; training data not documented) | Egyptian | Baselines to beat, not a base |
| OmniVoice (k2-fsa) | CC-BY-NC (code is Apache-2.0) | yes, has `arz` | No — measure only |
| VoiceTut-TTS | labelled Apache-2.0, but fine-tuned from CC-BY-NC OmniVoice weights | Egyptian | No — the label does not hold; measure only |
| Habibi-TTS / F5-TTS | CC-BY-NC-SA / CC-BY-NC | dialects incl. Egyptian | No |
| XTTS-v2 and its Egyptian fine-tunes | Coqui CPML (non-commercial) | MSA | No |
| Fish / OpenAudio S1 | CC-BY-NC-SA | yes | No |
| CosyVoice 3, Qwen3-TTS, Zonos | Apache-2.0 | **no Arabic** | No |

Why permissive-only: BA is to carry its own restricted licence and may become a commercial platform. A non-commercial or share-alike base would decide BA's licence for us and block the owner's own commercial use.

**Chatterbox V3 loads with the PyPI package.** `chatterbox-tts` 0.1.7 on PyPI predates V3, but upstream V3 support is only a different T3 weights file (`t3_mtl23ls_v3.safetensors`) plus two small inference changes (default `repetition_penalty` 1.2, drop the last speech token's ~40 ms). Same S3Gen, same tokenizer.

**Egyptian data is the weak point.** Most Egyptian speech sets on the Hub are scraped from YouTube or podcasts with no licence or a non-commercial one. The clean one is NileTTS (`ahmedez10/egyptian-arabic-tts-dataset`, Apache-2.0, 38 h, two synthetic voices).

## 3. Phases and gates

Each phase ends with files in the repo and a gate decided by the planner from those files.

### Phase 1 — Bake-off (GPU)

Measure every candidate on the same texts and the same reference voices.

- Systems: Chatterbox V3, Chatterbox V2, NAMAA-Egyptian-TTS, `lahgtna-chatterbox-v1`, VoxCPM2. Reference-only (cannot ship): OmniVoice, VoiceTut-TTS.
- Evaluation set and metrics: section 4.
- Output: `BA/eval/` (texts, manifests, scripts), `BA/eval/results/bakeoff.json` and `bakeoff.md`, a small listening pack for the owner.

**Gate G1 — choose the base.** Among licence-eligible systems with a failure rate of 5% or less: lowest Egyptian CER wins. If two are within 10% relative of each other, the higher speaker similarity wins. If those are within 0.02, the smaller and faster model wins.

### Phase 2 — Model development (GPU)

1. **Data audit.** `BA/docs/data-ledger.md` lists every dataset considered, its licence, speaker provenance and the decision. Training may use only data with an explicit licence that permits commercial use and redistribution of derived models (Apache, MIT, CC-BY, CC0) and that is not scraped from identifiable people without stated consent. Data that fails this may be used for measurement only.
2. **Fine-tune for Egyptian.** If Chatterbox wins G1: train the T3 text-to-speech-token model only; S3Gen (which carries the timbre) and the speaker encoder stay frozen. Mix in English and French so they are not forgotten. If VoxCPM2 wins: LoRA with its published recipe.
3. **The BA engine**, built and tuned on the evaluation set:
   - Reference audio: mono, resample, trim, loudness-normalise, pick the cleanest 6–10 s, accept several recordings, reject audio that is too short or too noisy with a clear message.
   - Voice profile: `create_voice(recordings)` returns the conditioning for that voice, held in memory, saved only if the caller asks.
   - Text: Egyptian normaliser (numbers, times, currency, percentages, symbols to Egyptian words), English and French number normalisation, a tested policy for Latin words inside Arabic text.
   - Long text: split at sentence boundaries, generate per chunk with fixed voice and seed, check each chunk (duration against text length) and retry failures, join with natural pauses.
   - Output: 24 kHz (or the base's native rate), loudness-normalised, imperceptible watermark on by default.
   - Defaults per language chosen by a sweep, not by taste.

**Gate G2 — does the fine-tune ship?** Only if, on held-out speakers never seen in training: Egyptian CER improves by at least 15% relative, or the Egyptian dialect-ID rate rises by at least 10 points; and speaker similarity drops by no more than 0.02; and English and French WER get no worse than 2 points; and the failure rate does not rise. Otherwise BA v0.1 ships the base weights with the BA engine, and the card says so.

**Gate G3 — which languages are listed.** Per language, on the final system: failure rate 5% or less; intelligibility within the bound in section 4; speaker similarity at least 80% of the real same-speaker ceiling. Egyptian is called "Egyptian Arabic" only if the dialect-ID rate on outputs is at least 60% of the rate on real Egyptian speech; otherwise it is listed as "Arabic" with the limitation stated.

### Phase 3 — Product

- Package `ba_tts` with CLI and tests.
- Space (Gradio): reference upload or microphone, required consent box, text, language, a few advanced controls, player and download. Runs on free CPU hardware; uses a GPU automatically if the Space is given one. Latency measured on 2 CPU threads before launch and stated on the page.
- Model card: what it is, quick start, measured results, how it works, limitations, responsible use, licence, attribution.
- Brand assets (can run in parallel with Phase 1; no GPU).
- `BA/hf/publish.py` and `BA/docs/reproduce.md`.

### Phase 4 — Publish and validate (needs the owner's token)

1. Create the model repo and the Space as **private**. Upload.
2. From a clean environment: download the model repo, run the documented snippet, score the output.
3. Wait for the Space build, call it through its API with a test clip, score the output.
4. Make both public. Re-run step 3 against the public URL.

## 4. Evaluation protocol

**Reference voices.** FLEURS (CC-BY-4.0): six `ar_eg` speakers (three male, three female), three `en_us`, three `fr_fr`; clean clips of 8–15 s. Used for measurement only; these clips and their clones are not published.

**Texts** (`BA/eval/texts/`): 40 Egyptian colloquial sentences (everyday talk, questions, numbers and dates, long sentences, a diacritised copy of ten of them), 20 English, 20 French, 6 Arabic–English code-switched, and one long passage of about 120 words per language.

**Conditions.** Same-language cloning for each language; cross-language cloning (Egyptian speaker speaking English and French).

**Metrics.**

| Question | Metric | Calibrated against |
|---|---|---|
| Are the words right? | CER (Egyptian) and WER (English, French) from Whisper large-v3 after normalisation (Arabic: strip diacritics, unify alef, yaa, taa marbuta, drop punctuation) | The same ASR on real speech in that language |
| Is it the same voice? | Cosine similarity of speaker embeddings (ECAPA-TDNN and WavLM-SV) between reference and output | Two different clips of the same real speaker (ceiling), clips of different speakers (floor) |
| Does it sound Egyptian? | Share of outputs an Arabic dialect-ID model labels Egyptian | The same model on real Egyptian speech and on real MSA |
| Does it sound natural? | Predicted MOS | Real speech clips |
| Does it break? | Failure rate: empty, cut off, or duration far from what the text length implies | — |
| Is it fast enough? | Real-time factor and peak VRAM on the A40; real-time factor on 2 CPU threads | — |

**Intelligibility bounds for G3.** Egyptian: CER no more than 0.10 above the ASR's CER on real Egyptian speech. English: WER 0.08 or less. French: WER 0.10 or less.

**What the metrics cannot do.** No automatic metric certifies that speech sounds like a native Egyptian. The owner listens to the listening pack; the card records whether that check was done and what it found.

Every number goes to `BA/eval/results/*.json` with the command and seed that produced it. Documents quote those files and nothing else.

## 5. Licensing and responsible use

- **BA's licence (proposed, to be reviewed by a lawyer before any commercial launch):** a BA Community Licence — non-commercial use, attribution, no use of a voice without the speaker's permission, no impersonation, fraud or deception; commercial rights reserved to the owner. Third-party notices (Chatterbox MIT or VoxCPM Apache-2.0) ship in `THIRD_PARTY_LICENSES`.
- **Model repo access:** gated with automatic approval; the user accepts the acceptable-use terms before downloading.
- **Space:** required consent box; uploads processed in a temporary directory and deleted after the request; no logging of text or audio; watermark on every output; text length and queue limits.
- **Published samples:** only voices that are synthetic or belong to someone who gave permission (the owner, if he records one). Never a cloned FLEURS speaker.
- **Training:** user uploads are never used for training.

## 6. Risks and fallbacks

| Risk | Fallback |
|---|---|
| No eligible model speaks convincing Egyptian zero-shot | Fine-tune (Phase 2). If G2 fails too, ship as "Arabic" with the limitation stated; do not claim Egyptian |
| Fine-tuning on two synthetic voices harms cloning of new voices | T3-only training, low learning rate, held-out-speaker gate G2 |
| Free CPU Space is too slow to be usable | Shorter text limit and stated latency; ZeroGPU if the account is upgraded to PRO |
| Latin words inside Arabic text are mispronounced | Tested policy in the normaliser; documented if unsolved |
| Container restart loses `/tmp` | Checkpoints and results copied to `/home/jovyan/ba-work/`; code committed at every gate |

## 7. Needed from the owner

| What | Blocks | Notes |
|---|---|---|
| Hugging Face write token for `mahmoudalyosify` | Phase 4 only | Fine-grained, write access to the account's repos. Revoke it after publishing |
| Listening check of the Egyptian samples | The "Egyptian" claim on the card | A few minutes, after Phase 1 and again after Phase 2 |
| Optional: HF PRO | GPU in the demo | Free CPU works without it, slower |
| Optional: a recording of the owner's own voice | Real-voice samples on the card | Otherwise samples use synthetic voices |

## 8. Working rules for executors

- Scratch in `/tmp/ba` (venvs, caches, datasets); results that must survive in `/home/jovyan/ba-work/`; code and small text files in the repo under `BA/`. No audio or weights in git.
- `source /tmp/ba/env.sh` in every shell: it unsets the server's `HF_TOKEN` and points caches at `/tmp/ba`.
- Install from PyPI only. Do not install or run code cloned from third-party git repositories; reading it is fine. Model weights come from the Hub.
- One GPU job at a time; check `nvidia-smi` first.
- Do not touch other users' files, keys or caches.
- Do not commit or push; the planner does that at each gate.
- No number in a document unless it is in a results file. No capability claimed unless it was run.

## Changes

### 2026-11-12 — Gate G1 amended after the bake-off (made after seeing the results)

**What changed.** G1 now also requires the Egyptian dialect bar from G3. New G1: among licence-eligible systems with a failure rate of 5% or less, whose Egyptian CER is within the G3 bound, **and whose dialect-ID share on the Egyptian set is at least 60% of the rate on real colloquial Egyptian speech**: lowest Egyptian CER wins; then the original tie-breaks.

**Why.** As written, G1 ranked on CER alone, and section 4 of this plan already says CER cannot see dialect. Applied mechanically it selects Chatterbox V3 (CER 0.035), whose Egyptian outputs the dialect classifier labels Egyptian only 21% of the time (mostly Levantine, Gulf or MSA), against 90% and 98% on real colloquial Egyptian speech. That base fails G3's Egyptian bar, so G1 would have chosen a model that cannot be called "Egyptian Arabic" — the brief's first priority — without a fine-tune on data we do not have. The gate was underspecified, not the results inconvenient: the amendment uses a bar written into G3 before any model ran.

**Result.** Bar = 0.6 × 0.90 = 0.54. Passing: VoxCPM2 (0.82; 0.91 with a reference transcript). Failing: Chatterbox V3 (0.21), V2 (0.17). VoxCPM2's Egyptian CER 0.065 is within the G3 bound (real colloquial Egyptian ASR CER 0.071 + 0.10). **Selected: VoxCPM2.** Numbers: `BA/eval/results/bakeoff.md`.

**Independent check.** The owner's blind listening test (system names hidden) is pending. If the ear test contradicts the classifier, this choice is reopened.
