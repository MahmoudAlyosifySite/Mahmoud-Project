# Multi-listener blind listening test (protocol, fixed before any rating is collected)

**Goal.** Test, with native listeners, the paper's two claims that rest today on one listener: (1) the lowest-CER
system is heard as less Egyptian than VoxCPM2 and BA; (2) BA's engine changes do not make the speech less Egyptian or
less natural than plain VoxCPM2.

**Listeners.** 12-20 adults who grew up in Egypt and speak Egyptian Arabic natively; not involved in the project.
Headphones required. Collected: age band, gender, governorate (no names).

**Voices.** 6-10 consenting volunteers (consent_form.md), balanced by gender; each gives ~60 s (recording_instructions.md).
The reference for cloning is a 10 s excerpt of the free-speech part; the rest is unused.

**Systems (hidden, order randomised per listener).** BA; plain VoxCPM2; Chatterbox V3; NAMAA; real speech (a different
sentence by the same volunteer, as a hidden upper anchor). VoiceTut/OmniVoice optional (reference only).

**Texts.** 12 held-out colloquial sentences never used for tuning: 6 everyday, 3 with numbers/times/prices,
3 code-switched. Each listener hears every system on different sentence-voice pairs (Latin-square design), about
60 clips, about 20 minutes.

**Questions per clip.**
1. How Egyptian does this sound? 1 (not Egyptian at all) to 5 (a native Egyptian speaker).
2. How natural is it? 1 (clearly machine) to 5 (completely natural). (MOS, ITU-T P.800 wording adapted)
3. Did you hear any wrong, missing or garbled word? yes / no.
4. (pairs only) Same speaker as this reference? 1 (definitely different) to 5 (definitely the same).

**Quality control.** 4 repeated clips (consistency); 2 real-speech clips that must score >= 4 on both scales; listeners
failing both checks are excluded, and the exclusion rule is applied before looking at system scores.

**Analysis (pre-specified).** Cumulative-link (ordinal) mixed model with random intercepts for listener, voice and
sentence; contrasts BA vs VoxCPM2, BA vs Chatterbox V3, VoxCPM2 vs Chatterbox V3 on Egyptian-ness and naturalness, Holm
correction over the six contrasts. Also: mean scores with listener-and-item bootstrap CIs, Krippendorff's alpha, and
the correlation of per-system Egyptian-ness with both dialect classifiers and with CER.

**Delivery.** A private web page (one link per listener) plays the clips and stores answers; no audio or answers are
public. Clips of a volunteer are published only if they ticked the optional box.
