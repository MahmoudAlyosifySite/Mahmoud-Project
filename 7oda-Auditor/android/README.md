# 7oda Auditor for Android

Android version of the offline app that recognises who holds the phone (and with which hand) from how the hand
absorbs a short random vibration. The algorithms are those of `docs/SPEC.md`; this directory holds a Kotlin port
of `AuditorCore` and the Android app.

## Layout

| Path | What |
|---|---|
| `core/` | Pure Kotlin/JVM library (no Android dependencies): random numbers, challenges, recordings and their JSON, DSP, signal processing, quality gate, features v2 (d = 39), calibration, v0 Gaussian model, v1 encoder input and embedding identifier. Public API: [`core/API.md`](core/API.md). |
| `core/src/test/` | JUnit 4 tests, including parity tests against the six fixture files shared with the Swift package (`../AuditorCore/Tests/AuditorCoreTests/Fixtures/*.json`, written by `research/scripts/make_fixtures.py` and `make_v1_fixtures.py`). A missing fixture fails the tests. |
| `app/` | The Android app (Jetpack Compose). |
| `app/src/main/assets/hand_encoder.onnx` | v1 encoder (SPEC 14): input `x` float32 `[1, 10, 256]`, outputs `embedding` `[1, 128]` and `hand` `[1, 3]`. Exported by `research/v1/export_onnx.py` from the same BN-folded, 8-bit palettized student as the iOS `HandEncoder.mlpackage` (weights stored in fp32, about 0.9 MB); the onnxruntime check against `research/v1/artifacts/reference_io.npz` is in `research/v1/artifacts/onnx_check.json`. |
| `signing/7oda-release.p12` | Release signing key (see below). |

## Build

No Gradle wrapper is committed. With Gradle 8.11.1 and JDK 17 (as in CI):

```sh
gradle -p 7oda-Auditor/android --no-daemon test assembleRelease
```

Versions are pinned in `build.gradle.kts` (Android Gradle Plugin 8.7.3, Kotlin 2.1.0) and `core/build.gradle.kts`
(kotlinx-serialization-json 1.7.3, JUnit 4.13.2). The core tests alone: `gradle -p android :core:test`.

## Recordings

`RecordingCoding` writes the same JSON as the iOS app (SPEC 4): sorted keys, ISO-8601 UTC dates without fractional
seconds, seeds as hex strings, optional keys left out. Android exports (JSON Lines) can therefore be analysed with
`research/evaluation/analyze_export.py` like iOS exports. Accelerometer values are stored in g including gravity
(Android reports m/s²; divide by 9.80665), gyroscope values in rad/s, and all timestamps, including `hapticStart`,
in seconds on one monotonic clock.

## Signing key

`signing/7oda-release.p12` is a PKCS#12 keystore (RSA 2048, self-signed certificate `CN=7oda Auditor`, valid for
30 years), alias `7oda`, store and key password `7oda-auditor`. It is deliberately public: it only exists so that
research builds from CI and from any machine carry the same signature and install over each other. It is **not**
suitable for publishing to an app store; a store release needs its own private key.

To recreate an equivalent key:

```sh
openssl req -x509 -newkey rsa:2048 -sha256 -keyout key.pem -out cert.pem -days 10958 -nodes -subj "/CN=7oda Auditor"
openssl pkcs12 -export -inkey key.pem -in cert.pem -name 7oda -out 7oda-release.p12 -passout pass:7oda-auditor
rm key.pem cert.pem
```
(a recreated key has a different certificate, so builds signed with it do not update installs signed with the old one).
