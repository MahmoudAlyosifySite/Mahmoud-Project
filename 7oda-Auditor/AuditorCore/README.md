# AuditorCore

Platform-independent core of **7oda Auditor**: everything between the raw sensor
recording and the "who is holding the phone" decision. Foundation only (no
Accelerate, Core Motion, Core Haptics or SwiftUI), so it builds and tests on Linux
and on Apple platforms. `docs/SPEC.md` is the normative description; section numbers
below refer to it, and section 10.1 lists the public API the app is written against.

## Run the tests

```sh
swift test --package-path 7oda-Auditor/AuditorCore
```

(from the repository root; inside `7oda-Auditor/` it is `swift test --package-path AuditorCore`).
Swift 5.9+ toolchain, Swift 5 language mode, platforms iOS 17 / macOS 13.

## Module map

| File | SPEC | Contents |
|---|---|---|
| `Random.swift` | 2 | `SplitMix64`, `Xoshiro256StarStar` (also a `RandomNumberGenerator`), `SeedFormat` (`0x` + 16 hex digits) |
| `Challenge.swift` | 3 | `CurvePoint`, `TapEvent`, `RampEvent`, `ChallengeEvent` (JSON tagged by `kind`), `ChallengeSpec`, `ChallengeConfig`, `ChallengeGenerator` |
| `Recording.swift` | 4 | `SensorStream`, `MotionStream`, `Hand`, `Accessory`, `Posture`, `RecordingPurpose`, `RecordingLabel`, `DeviceInfo`, `Recording`, `RecordingCoding` (JSON / JSON Lines) |
| `DSP.swift` | 5.2–5.4 | `Biquad` (RBJ designs, direct form I, zero-phase `filtfiltZ`), `Resampler`, `StreamCleaner` (stable sort + dedup) |
| `Signal.swift` | 5 | `SignalConfig`, `AlignmentResult`, `ProcessedSignal`, `SignalProcessor` (grid, filters, envelope, template, lag search, event indices) |
| `Quality.swift` | 6 | `QualityMode`, `QualityIssue`, `QualityMetrics`, `QualityConfig`, `QualityReport`, `QualityGate` |
| `Features.swift` | 7 | `FeatureExtractor` (feature names, version, calibration-relative indices) |
| `Calibration.swift` | 8 | `Calibration` and `Calibration.build` (median over passing on-table patterns) |
| `Model.swift` | 9.1–9.2 | `PrototypeKey`, `SampleSource`, `TemplateSample`, `ModelConfig`, `GaussianModel` (Ledoit–Wolf shrinkage, Cholesky, leave-one-out threshold) |
| `Identifier.swift` | 9.3–9.5 | `IdentifierConfig`, `Decision`, `DecisionOutcome`, `PersonScore`, `PrototypeDistance`, `Identifier` (hand head, sequential decision, adaptive samples and cap) |
| `Evaluation.swift` | – | `ConfusionMatrix` for the Lab test run |
| `Numerics.swift`, `LinearAlgebra.swift` | – | internal helpers: left-to-right sums, numpy median and linear quantile, Gaussian elimination, Cholesky, forward substitution |

Typical flow in the app:

```swift
let spec = ChallengeGenerator.generate(seed: seed)                 // play it, record sensors
let recording = Recording(purpose: .check, challenge: spec, hapticStart: start,
                          accel: accel, gyro: gyro, motion: motion)
let processed = SignalProcessor.process(recording)
let report = QualityGate.evaluate(processed, calibration: calibration, mode: .handheld)
if report.passed {
    let x = FeatureExtractor.extract(processed, calibration: calibration)
    let decision = identifier.decide(patternsSoFar + [x])               // up to maxPatterns
}
```

## Parity with the Python reference

`research/auditor_ref/` is the reference implementation. `research/scripts/make_fixtures.py`
writes `Tests/AuditorCoreTests/Fixtures/*.json` (SPEC 11) and the parity tests read them
through `Bundle.module`; a missing fixture is a failure, never a skip. Tolerances: RNG and
challenge exact; DSP `1e-9·max(1,|x|)`; pipeline indices and issues exact, metrics and
features `1e-6·max(1,|x|)`; model `1e-6·max(1,|x|)`, outcomes exact. Reductions are written
as left-to-right loops in the same order as the reference so that results agree to the
last bits wherever the reference is sequential.

## Notes for the app layer

- Encode and decode every stored JSON file that contains dates (recordings, calibration,
  template samples, settings) with `RecordingCoding.encoder()` / `decoder()`: ISO-8601 UTC
  dates without fractional seconds (fractional seconds are accepted when decoding), sorted
  keys. Non-finite doubles are written as the strings `"Infinity"`, `"-Infinity"`, `"NaN"`
  instead of making the whole file fail.
- `Identifier` uses the feature dimension of the first sample and ignores samples of
  another dimension; recompute stored features from the raw recordings when
  `FeatureExtractor.version` changes. A feature vector of the wrong dimension is infinitely
  far from every prototype (the decision becomes Unknown, never a crash).
- Grid length is capped at 100 000 samples so that a corrupted duration cannot exhaust memory.
