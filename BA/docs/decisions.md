# BA — Decision log

One entry per major decision: what was decided, why, and what would change it. Newest last.

## D1 — Only permissively licensed weights can be BA's base (2026-11-12)

**Decision.** BA is built on weights under MIT or Apache-2.0. Models whose weights are non-commercial or share-alike are measured for reference but never shipped.

**Why.** BA is to carry its own restricted licence and may become a commercial platform. A CC-BY-NC or CC-BY-NC-SA base would fix BA's licence for us and forbid the owner's own commercial use.

**Consequence.** OmniVoice (CC-BY-NC weights), Habibi-TTS and F5-TTS (CC-BY-NC-SA, CC-BY-NC), XTTS-v2 (Coqui CPML) and Fish/OpenAudio (CC-BY-NC-SA) are out, although several of them have Egyptian-specific fine-tunes. Chatterbox Multilingual (MIT) and VoxCPM2 (Apache-2.0) are the candidates.

**Would change if.** The owner decides BA will never be commercial and accepts a share-alike licence.

## D2 — A licence label on the Hub is not taken at face value (2026-11-12)

**Decision.** For every model and dataset, the licence of what it was derived from is checked, not only its own tag.

**Why.** VoiceTut-TTS is tagged Apache-2.0 but is a fine-tune of OmniVoice, whose weights are CC-BY-NC. A fine-tune cannot drop the non-commercial term of its parent. Building BA on it would import that problem.

## D3 — The base is chosen by measurement on Egyptian Arabic, with the rule fixed in advance (2026-11-12)

**Decision.** Gate G1 in `execution-plan.md`: lowest Egyptian CER among eligible systems, then speaker similarity, then size.

**Why.** Egyptian quality is the brief's first priority and the hardest part; published benchmarks for these models report MSA or do not report Arabic at all.

## D4 — Training data must be licensed for commercial use and not scraped from identifiable people (2026-11-12)

**Decision.** Only datasets with an explicit Apache, MIT, CC-BY or CC0 licence and documented speaker provenance may train weights BA publishes. Everything else is measurement-only.

**Why.** A voice product that asks its users for consent cannot be trained on voices taken without it. Most Egyptian speech sets on the Hub are YouTube or podcast scrapes with no licence.

**Consequence.** The usable Egyptian training set today is NileTTS (Apache-2.0, 38 h, two synthetic voices). That is small and narrow, which is why gate G2 tests on unseen speakers before a fine-tune ships.

## D5 — If the base is Chatterbox, only T3 is fine-tuned (2026-11-12)

**Decision.** Train the text-to-speech-token model (T3); keep S3Gen and the speaker encoder frozen.

**Why.** In Chatterbox the reference voice's timbre is rendered by S3Gen from the reference audio; T3 decides content and prosody. Dialect is a content-and-prosody problem. Leaving S3Gen untouched protects cloning of unseen voices while the dialect is learned from few speakers.

## D6 — The server's Hugging Face token is not used (2026-11-12)

**Decision.** All downloads are anonymous; publishing waits for the owner's own token.

**Why.** The token in the server environment belongs to another account.

## D7 — Published audio samples use only synthetic or consenting voices (2026-11-12)

**Decision.** FLEURS speakers serve as reference voices for measurement. Their clips and clones are not published. Card and Space samples use synthetic reference voices, or the owner's voice if he provides one.

**Why.** FLEURS speakers licensed recordings for speech research, not for public demonstrations of their cloned voice saying new things.

## D8 — The demo targets free CPU hardware and upgrades itself if a GPU is present (2026-11-12)

**Decision.** The Space must work on the free CPU tier, with measured latency stated on the page.

**Why.** The owner's account is not PRO, and hosting a ZeroGPU Space requires PRO. The same app uses a GPU when the Space has one.

## D9 — Proposed licence: BA Community Licence, gated access (2026-11-12)

**Decision (proposed).** Non-commercial, attribution, consent required for any voice, no impersonation or deception, commercial rights reserved to the owner; model repo gated with automatic approval so users accept the terms.

**Why.** The brief asks for a restricted licence and misuse safeguards. MIT and Apache-2.0 bases allow a derivative to carry stricter terms as long as their notices are kept.

**Open.** The licence text should be reviewed by a lawyer before any commercial launch.
