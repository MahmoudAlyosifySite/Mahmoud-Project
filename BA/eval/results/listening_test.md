# Owner's blind listening test (Phase 1 outputs)

Raw bake-off outputs, before the BA engine. 6 Egyptian sentences (one with digits, one code-switched) × 8 systems, names hidden and order shuffled; one rater, the owner, a native Egyptian speaker; 2026-11-12. Data: `listening_test.json`; script: the tally below was computed from it.

| system | Egyptian | partly | not Egyptian | all words right |
|---|---|---|---|---|
| voxcpm2 | 3/6 | 3/6 | 0/6 | 3/6 |
| voicetut | 3/6 | 3/6 | 0/6 | 5/6 |
| voxcpm2-prompt | 1/6 | 5/6 | 0/6 | 5/6 |
| omnivoice | 2/6 | 3/6 | 1/6 | 3/6 |
| lahgtna | 2/6 | 2/6 | 2/6 | 4/6 |
| namaa | 1/6 | 3/6 | 2/6 | 4/6 |
| cb-v3 | 0/6 | 0/6 | 6/6 | 4/6 |
| cb-v2 | 0/6 | 0/6 | 6/6 | 2/6 |

Rater's note on the digits sentence (egy28): "الرقم مش بيتقال بالمصري في اغلبهم" — in most systems the number is not said the Egyptian way.

**Reading.** The ear test agrees with the dialect classifier on the decision that mattered: both Chatterbox bases were rated not Egyptian on 6/6 sentences, and VoxCPM2 was never rated not Egyptian (3 Egyptian, 3 partly), tied with VoiceTut (which cannot ship) for the best score. It disagrees on one detail: by ear, VoxCPM2 with the reference transcript (BA's default mode) sounded less Egyptian than without it (1 vs 3 fully Egyptian) while getting more words right (5 vs 3); the classifier ranked it the other way. With 6 clips per system and one rater, neither difference is settled. BA's own outputs (with the Egyptian number normaliser) have not been rated by ear yet.
