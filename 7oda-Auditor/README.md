# 7oda Auditor

An offline iPhone app that names who is holding the phone, and with which hand, from how that hand absorbs a short random vibration. The Taptic Engine plays a fresh random pattern of taps and intensity ramps; the motion sensors record how the phone rings and settles in the hand (~100 Hz, the iOS limit); the app answers with a name and a hand ("Mahmoud · right hand") or **Unknown**.

Proposal: Nov 24, 2026, Mahmoud Sayed Youssef. This folder is the proof-of-concept implementation.

> **Status:** MVP complete and green in CI (core, app, Core ML); not yet run on a physical iPhone — see [Status](#status).

## Status

| Part | State | How it was checked |
|---|---|---|
| Swift core (`AuditorCore`) | Complete: challenge, recording, DSP, alignment, quality gate (with replay check), features v2, calibration, Gaussian model, sequential decision, v1 encoder input and embedding identifier | 47 XCTest tests pass on Linux (Swift 6.2) and macOS (Xcode 26.3), including parity tests against fixtures written by the Python reference (RNG, challenge, DSP, full pipeline, model, v1) |
| iOS app (`Auditor`) | Complete: calibration, enrollment (both hands, 3 rounds), "Who's holding?" with sequential checks, people and accessories, Lab (signal plot, scores, test runs with confusion matrices, thresholds, signal check, export), encrypted storage, v0 + v1 recognizers | 17 unit tests pass on the iOS simulator (Core ML encoder, encrypted store, enrollment flow, export, haptic patterns, end-to-end with synthetic hands); unsigned Release build for a generic iOS device succeeds |
| v1 model (`HandEncoder.mlpackage`, 323 KB) | Teacher (10.7 M) trained on the A40, distilled into a 220 k student, 8-bit palettized | Core ML outputs match PyTorch on macOS (max embedding difference 0.002, hand argmax identical) |
| Android app (`android/`) | Kotlin port of the core (same algorithms, parity-tested against the same fixtures), Jetpack Compose app with the same screens, sensor capture converted to the iOS units, vibration waveforms, ONNX encoder (v1), encrypted storage | Core unit and parity tests and app unit tests pass in CI; release APK built and signed in CI. Not yet run on a physical phone. See [docs/ANDROID.md](docs/ANDROID.md) |
| Python reference, simulator, evaluation | Complete, including `analyze_export` for pilot data | 335 tests pass locally and in CI |
| Accuracy on real hands | **Not measured yet** | Needs a physical iPhone and the week-2 pilot; all numbers so far are simulated ([DECISIONS D9](docs/DECISIONS.md)) |

Simulated results (20 people + strangers, cross-session, ≤ 2 % strangers accepted): v1 names the right person in 65 % (aliasing IMU) / 84 % (filtered IMU) of checks, v0 in 6 % / 8 %; without the Unknown option v0 is right in 62 % / 86 % ([V1_MODEL](docs/V1_MODEL.md), [feature comparison](research/results/feature_comparison.md)). The app therefore answers with v1 and logs v0 beside it ([D12](docs/DECISIONS.md)).

## Run it on an iPhone

**No Mac?** Download the latest `.ipa` (and the Android `.apk`) from the [7oda-auditor-latest release](https://github.com/MahmoudAlyosifySite/Mahmoud-Project/releases/tag/7oda-auditor-latest) and follow [docs/INSTALL.md](docs/INSTALL.md) (Sideloadly on Windows, free Apple ID).

With a Mac:

Requirements: a Mac with **Xcode 16 or newer**, an iPhone 8 or newer on **iOS 17+** (Taptic Engine + Core Haptics), and an Apple ID (a free personal team is enough).

1. Clone the repository and open `7oda-Auditor/Auditor.xcodeproj`.
2. Select the **Auditor** target → *Signing & Capabilities* → choose your **Team**. If Xcode says the bundle identifier is taken, change `com.mahmoudalyosify.auditor` to something unique.
3. Plug in the iPhone, pick it as the run destination, press **Run** (⌘R). On first install, trust the developer certificate on the phone: *Settings → General → VPN & Device Management*.
4. In the app:
   - **Calibrate** (first launch): put the phone flat on a table and don't touch it for ~40 s.
   - **People → Add person**: type a name, take off watch/rings/bracelets, then hold the phone through 15 short vibrations per hand (sitting, standing, fresh grip). About two minutes per person.
   - **Who's holding?**: press the button; one pattern (~1.5 s) usually answers, up to three when the match is unclear.
   - **Lab**: live signal plot, scores, thresholds, test runs with a confusion matrix, signal check, and export of every raw recording for analysis.

The Simulator builds and runs the app, but it has no motion sensors or Taptic Engine, so recognition only works on a real iPhone.

## How it works

1. **Challenge** — a random pattern from a secure seed: 6–8 taps with random intensity, sharpness and gaps, plus two intensity ramps (`AuditorCore/Sources/AuditorCore/Challenge.swift`, SPEC §3).
2. **Recording** — accelerometer, gyroscope and device motion at 100 Hz from 0.25 s before to 0.25 s after the pattern, resampled onto a uniform grid by timestamp (SPEC §4–5).
3. **Alignment and quality gate** — the recorded vibration energy is cross-correlated with the expected envelope of the pattern; a phone resting on a table, a moving hand, sensor gaps, a weak signal or a recording that does not line up with the fresh pattern (replay) are rejected before judging (SPEC §5.6, §6).
4. **Features (v0)** — 39 numbers built to survive 100 Hz sampling of a faster vibration: per-axis gains summed over taps, axis shares and cross products of simultaneously sampled channels (where the sampling phase cancels), driven-burst vs free-response shape, ramp gains and shares, tremor, and grip angle (SPEC §7, [D10](docs/DECISIONS.md)).
5. **Decision** — a shared-covariance Gaussian model with Ledoit–Wolf shrinkage scores every (person, hand, accessory) prototype; the answer needs an absolute distance threshold, a margin over every other person, and a margin plus a hand-head agreement for the hand; otherwise another pattern is played, up to three, and then "Unknown" (SPEC §9).
6. **v1 (default recognizer)** — a learned encoder: a 10.7 M teacher trained on 1 M simulated patterns on the A40, distilled into a 220 k student and shipped as an 8-bit Core ML model (323 KB) that turns each pattern into a 128-d embedding compared with cosine prototypes (SPEC §14, [docs/V1_MODEL.md](docs/V1_MODEL.md)). v0 runs beside it on every check so the pilot can compare them on real hands.

## Repository layout

| Path | What |
|---|---|
| `Auditor/`, `AuditorTests/` | SwiftUI app (Core Haptics, Core Motion, encrypted storage) and its unit tests |
| `AuditorCore/` | Swift package with all signal processing, features and recognition logic (Foundation only, tested on Linux and macOS) |
| `Auditor.xcodeproj` | Generated by `tools/gen_xcodeproj.py` (Xcode 16 format); regenerate instead of editing |
| `research/auditor_ref/` | Python reference implementation of the same pipeline (writes the parity fixtures the Swift tests check) |
| `research/sim/`, `research/evaluation/` | Physics simulator and the evaluation protocol of the proposal |
| `research/v1/` | v1 teacher/student training on the GPU and Core ML export |
| `docs/` | [SPEC](docs/SPEC.md), [DECISIONS](docs/DECISIONS.md), [APP](docs/APP.md), [SIMULATION](docs/SIMULATION.md), [V1_MODEL](docs/V1_MODEL.md) |
| `android/` | Android version: `core/` (Kotlin port of AuditorCore) and `app/` (Compose) — [docs/ANDROID.md](docs/ANDROID.md) |
| `tools/` | Project generator, icon generator, CI log annotator |

## Development

```bash
# Python reference, simulator, evaluation (fast tests)
cd 7oda-Auditor/research && pip install -r requirements.txt && python -m pytest -q

# Regenerate the parity fixtures after changing the reference
python scripts/make_fixtures.py

# Swift core (needs a Swift 5.9+ toolchain)
swift test --package-path 7oda-Auditor/AuditorCore

# Regenerate the Xcode project after changing build settings
python3 7oda-Auditor/tools/gen_xcodeproj.py
```

CI (`.github/workflows/7oda-auditor.yml`) runs the Python tests, the Swift core tests on Linux and macOS, the app's unit tests on an iOS simulator, an unsigned Release build for a generic iOS device, and the Core ML conversion check.

## Pilot data

In the Lab, export the recordings (JSON Lines, names can be anonymized), copy the file to a computer and run:

```bash
cd 7oda-Auditor/research
python -m evaluation.analyze_export /path/to/recordings.jsonl
```

This computes the proposal's metrics (cross-session top-1, hand accuracy, stranger accept rate, false Unknown rate, patterns per answer) on real data.

## Privacy

Everything stays on the phone. There is no network code. Stored data (people, templates, raw recordings, calibration) is encrypted with AES-GCM under a key kept in the Keychain (this device only), and files use complete Data Protection. Exports happen only when the user shares a file.
