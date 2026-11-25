# v1 learned hand encoder

v1 replaces the 36 hand-made features of v0 (SPEC 7) with a learned 128-d embedding. A large
teacher is trained on the A40 on simulated hands, distilled into a student of 220k parameters,
and the student is exported to Core ML for the phone (SPEC 14). This document records the
architecture, data, training schedule, results, how to retrain on real recordings, and how the
phone will use the model.

All numbers below come from the physics simulator (`research/sim`). They check the pipeline and
compare v0 and v1 on the same synthetic recordings; they are not evidence about real hands
(DECISIONS D9). The ship decision for the phone (D7) has to be made again on the pilot and study
data.

**Verdict (simulator): RESULTS_VERDICT**

## 1. Contract (SPEC 14)

- Input `x`: float32 `[1, 10, 256]`, built from the processed signal of SPEC 5: accelHP / 0.02
  (channels 0-2), gyroHP / 0.05 (3-5), device-motion gravity (6-8) and the lag-0 template (9),
  aligned so the pattern starts at `u = 25`, clipped to `[-50, 50]` (`v1.data`).
  `v1.sigproc` is a vectorised copy of the SPEC 5 path for bulk data;
  `tests/test_v1_data.py` checks it against `auditor_ref`.
- Outputs: `embedding` `[1, 128]` (L2-normalized) and `hand` `[1, 3]` logits
  (right, left, twoHanded).
- Artifact: `research/v1/artifacts/HandEncoder.mlpackage` (ML Program, iOS 17, 8-bit palettized
  weights), `reference_io.npz` (16 inputs with the PyTorch outputs) and the model card
  `artifacts/README.md`.

## 2. Architecture (`research/v1/models.py`)

Both networks read the template channel through a small convolutional encoder whose outputs
modulate every residual block (FiLM: `h * (1 + g) + b`), so the network knows which random
pattern was played.

| | Teacher (A40) | Student (iPhone) |
|---|---|---|
| Body | 1D ResNet, 4 stages x 2 basic blocks, widths 128/256/384/512, kernel 5, strides to T/8 | stem 32 ch + 7 inverted residual blocks (pointwise expand x2, depthwise k=7, pointwise project), widths 48-128, strides to T/16 |
| Conditioning | template encoder (64 ch) + FiLM per block | template encoder (16 ch) + FiLM per block |
| Pooling | mean and standard deviation over time | mean and max over time |
| Embedding | MLP 1024 -> 512 -> 128 + BatchNorm, L2 | linear 256 -> 128 + BatchNorm, L2 |
| Hand head | MLP 1024 -> 256 -> 3 | MLP 256 -> 32 -> 3 |
| Parameters | 10.7 M | 219,747 |

Training-only heads: ArcFace (additive angular margin) over (person, hand) identities, masked
reconstruction head (optional self-supervised term) and a projection head (contrastive
pretraining, `v1.pretrain_ssl`).

## 3. Data

Simulator: `research/sim` (person = 7 hand contacts with stiffness, damping, mass and a cubic
term per hand; variability person/hand -> session -> grip -> trial; phones with their own
actuator, IMU position, latency and noise). Simulator code hash at generation time:
`SIM_HASH` (md5 of `research/sim/*.py`). Generated with `v1.simdata` into
`~/7oda-work/v1/data` (not in the repository):

| Set | People | Phones | IMU | Layout | Recordings |
|---|---|---|---|---|---|
| `train10` | 10,000 (seeds 0..) | pool of 200 (seeds 1000..1199) | mixed none / lpf per person | 4 sessions x (12 per hand + 2 two-handed), 15 % with an accessory | 1,040,000 |
| `val` | 400 (seeds 5,000,000..) | 50 unseen (5000..) | none | as train | 41,600 |
| `proto_none` | 300 (seeds 6,000,000..) | 1 unseen phone (seed 3) | none | protocol (below) | 63,000 |
| `proto_lpf` | 300 (seeds 7,000,000..) | 1 unseen phone (seed 5) | lpf | protocol | 63,000 |

Protocol layout (the proposal's evaluation plan): session 0 = enrollment, 15 patterns per hand
(rounds of 5: sitting, standing, fresh grip); sessions 1-3 = 10 checks per hand of up to 3
patterns each, a fresh grip per check. The protocol shards also store the v0 features
(`auditor_ref`) and the SPEC 6 quality verdict of every recording, so v0 and v1 see the same
recordings.

```bash
cd research
python -m v1.simdata --backend sim --layout train --out ~/7oda-work/v1/data/train10 --people 10000 \
  --per-hand 12 --sessions 4 --two-handed 2 --accessory-prob 0.15 --workers 14 --people-per-shard 10 \
  --prefix T --seed 1 --option imu_mode='"mixed"' --option person_seed_base=0
python -m v1.simdata --backend sim --layout protocol --out ~/7oda-work/v1/data/proto_none --people 300 \
  --sessions 4 --enroll-per-hand 15 --checks-per-hand 10 --patterns 3 --workers 14 --people-per-shard 5 \
  --prefix E --seed 3 --option imu_mode='"none"' --option person_seed_base=6000000 --option phone_seed=3
```

Keep `--people-per-shard` small (about 0.3 GB of memory per worker at 10 people per shard).

## 4. Learnability check

Before the main runs: can identity be read from the raw tensor at all, and does v1 see more than
v0? `v1.learnability`: 200 simulated people on one phone, closed set of 400 (person, hand)
classes, trained on sessions 0-3 and tested on held-out patterns of the same grip, a new grip on
the same day, and the next session (chance 0.25 %). v0 = the 36 features with the SPEC 9.2
Gaussian model; v1 = a 3.1 M-parameter teacher trained only on these 200 people
(`research/results/v1_learnability_*.json`).

| Model (top-1, 400 classes) | same grip | new grip | next session |
|---|---|---|---|
| v0 features + Gaussian model | 31.3 % | 29.7 % | 25.9 % |
| v1, heavy augmentation, 40 epochs (underfits) | 20.1 % | 21.0 % | 17.7 % |
| v1, no augmentation, 80 epochs | 30.5 % | 30.3 % | 25.4 % |
| v1, no augmentation, plain cosine softmax | 45.1 % | 45.1 % | 35.7 % |
| v1, light augmentation, 80 epochs | 52.4 % | 53.2 % | 44.3 % |

Findings: identity is learnable from the raw tensor, and even with 32 training patterns per
class v1 beats v0 by 18 points on the next session. Same grip and next session differ by only
about 8 points for both models: trial-to-trial variation (random pattern, sensor noise, timing),
not day-to-day drift, dominates the simulator. Heavy augmentation (15 % gain jitter, simulated
case damping) removes the amplitude cues the simulated hands differ in, so simulated training
uses the light setting (`AugmentConfig.light`). v0's poor numbers in the simulated study come
from the open-set thresholds (below), not from features that carry no identity.

## 5. Training

### 5.1 Teacher (`v1.train_teacher`)

TEACHER_SCHEDULE

Mirror augmentation (hand head only): a random 25 % of each batch is appended mirrored across
the phone's y-z plane with swapped hand labels and no identity label. The Taptic Engine pushes
along the same phone axis whichever hand holds the phone, so the mirrored world also has its
excitation reversed: for a linear (or odd-nonlinear) phone+hand system the vibration channels
transform by `-M` (accelerometer `diag(1,-1,-1)`, gyroscope `diag(-1,1,1)`) and gravity by `M`
(`--mirror-mode actuator`, now the default). The plain reflection of the proposal flips the
polarity of the vibration relative to the excitation, so mirrored copies contradict real
recordings of the other hand; a first run with it (T1, stopped after 2,600 steps) had a
validation hand accuracy of 0.58.

### 5.2 Self-supervised term

`train_teacher --ssl-weight` adds masked-patch reconstruction (30 % of 8-sample patches masked,
reconstructed from the last feature map) to the supervised loss. A/B on the simulator, 3.1 M
teacher, 4 epochs on 520k recordings, everything else equal (`research/results/v1_SSL0_*`,
`v1_SSL1_*`):

| Run | Val top-1 (400 identities, next sessions) | Val EER |
|---|---|---|
| SSL0: supervised only | 82.9 % | 1.59 % |
| SSL1: + masked reconstruction (weight 0.2) | 77.0 % | 2.08 % |

With unlimited labelled simulated data the self-supervised term only costs accuracy (the masked
input also weakens the identity loss), so the simulated teacher is trained without it.
`v1.pretrain_ssl` (masked reconstruction + contrastive views) stays for the real recordings,
where most data will be unlabelled (proposal step 2); test it there against fine-tuning alone.

### 5.3 Student (`v1.distill`)

STUDENT_SCHEDULE

## 6. Evaluation (`v1.evaluate_v1`)

Protocol: per IMU mode 300 held-out people on one unseen phone, split into 10 groups of 30:
20 enroll (session 0), 5 tuning strangers, 5 test strangers. Thresholds (absolute slack, name
margin, hand margin; for v0 absFactor and its margins) are tuned on session 1 checks of the
enrolled people plus the tuning strangers to maximize name+hand accuracy with stranger accept
<= 2 % (also 5 % and 10 %), then frozen and applied to sessions 2-3 plus the test strangers.
Sequential checks: up to 3 patterns, stop as soon as the SPEC 9.4 / 14.3 rule identifies;
patterns failing the SPEC 6 quality gate are skipped (the app retries them). v0 runs the
`auditor_ref` features and Gaussian model (absolute features: one phone, no calibration step)
with the same rule and the same tuning. Its numbers match the separate v0 study harness
(`research/results/sim_study_none.md`: top-1 6.3 % at 1.3 % stranger accept, forced choice
56 %).

Columns: top-1 = right name given (identified or hand unclear) over enrolled checks; hand = fully
identified with the right hand; stranger accept = strangers given any name; false Unknown =
enrolled checks answered Unknown; closed-set = nearest prototype with 1 pattern and no Unknown
option.

RESULTS_TABLES

## 7. Core ML export (`v1.export_coreml`, `v1.verify_coreml`)

EXPORT_RESULT

## 8. How the phone uses it (SPEC 14.3)

1. Each passing pattern -> tensor `x` (SPEC 14.1, same code path as SPEC 5) -> `embedding`,
   `hand`.
2. Enrollment: prototype = L2-normalized mean embedding per (person, hand, accessory); keep the
   leave-one-out similarities of the enrollment samples.
3. Check: per pattern the cosine similarity to every prototype, averaged over the patterns played
   so far; best prototype, name margin (best minus best other person) and hand margin (best minus
   the same person's other hand), absolute test `best >= quantile_5%(LOO) - slack`; the SPEC 9.4
   sequential rule with at most 3 patterns.
4. Hand rule: HAND_RULE
5. Template updates (SPEC 9.5) carry over unchanged with similarities.

Thresholds tuned on the simulator are placeholders; tune them on the pilot data (held-out
session, stranger accept target), as for v0.

## 9. Retraining on real recordings

App exports (JSON Lines of SPEC 4 recordings) go through the same tensor path
(`v1.data.tensorset_from_jsonl`). With 20 participants, hold people out (k-fold over people) so
the reported numbers are for people the model never saw:

```bash
cd research
# 1. optional: self-supervised pretraining on every real recording, labelled or not
EXPORTS=$(ls exports/*.jsonl | paste -sd, -)     # comma-separated app exports
python -m v1.pretrain_ssl --run SSL-real --init ~/7oda-work/v1/runs/T2/best.pt --jsonl "$EXPORTS"
# 2. fine-tune the simulator teacher on the study data (person-held-out folds)
python -m v1.finetune --run FT-f0 --init ~/7oda-work/v1/runs/T2/best.pt --jsonl "$EXPORTS" --folds 4 --fold 0
# 3. distill into the student (study data + simulated data), export and check
python -m v1.distill --run S-real --teachers ~/7oda-work/v1/runs/FT-f0/best.pt --train 'DATA/*.npz' --val 'VAL/*.npz'
~/7oda-work/venv-export/bin/python -m v1.export_coreml --ckpt ~/7oda-work/v1/runs/S-real/best.pt --val 'VAL/*.npz'
python v1/verify_coreml.py   # macOS (CI runs it whenever reference_io.npz exists)
# 4. evaluate against v0 on the same recordings with v1.evaluate_v1 (protocol shards from the study)
```

Use the heavy augmentation (`--aug full`, the default) for real data, and re-tune the simulator
ranges to the pilot recordings before regenerating simulated data (proposal step 1). The export
environment is `~/7oda-work/venv-export` (torch 2.7 CPU + coremltools 9.0); training uses
`requirements-v1.txt`.

## 10. Open issues

OPEN_ISSUES
