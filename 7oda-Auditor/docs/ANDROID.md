# 7oda Auditor on Android

The Android app is a port of the iOS app (`Auditor/`, described in [APP.md](APP.md)): the same screens,
flows, settings, storage layout and export files. It is written in Kotlin with Jetpack Compose
(Material 3) and lives in `android/app`; the recognition pipeline is the pure Kotlin library
`android/core` (API in [`android/core/API.md`](../android/core/API.md)), a line-by-line port of the Swift
`AuditorCore`, checked against the same fixtures.

## Install

1. On the phone, open the GitHub release **`7oda-auditor-latest`** of this repository and download the
   APK (`7oda-auditor-*.apk`).
2. Open the file. Android asks to allow installs from this source (the browser or the Files app):
   allow it once ("Install unknown apps"), then tap Install.
3. Later builds install over the old one and keep the data, because every build is signed with the same
   key (`android/signing/7oda-release.p12`, see `android/README.md`).

Requirements: Android 8.0 (API 26) or newer, an accelerometer, a gyroscope and a vibration motor.
On a phone without the sensors, the app explains that it cannot record and lets you look around.

## Architecture

| Layer | Files (`android/app/src/main/java/com/mahmoudalyosify/auditor/`) | iOS counterpart |
|---|---|---|
| Device | `device/MotionRecorder.kt`, `device/SensorConversion.kt` | `Device/MotionRecorder.swift` |
| | `device/HapticTimeline.kt` (pure), `device/HapticPlayer.kt` | `Device/HapticPlayer.swift` |
| | `device/CheckRunner.kt` (challenge → play → wait → cut window) | `Device/CheckRunner.swift` |
| v1 model | `ml/EmbeddingEncoder.kt` (ONNX Runtime, `assets/hand_encoder.onnx`) | `ML/EmbeddingEncoder.swift` |
| Storage | `storage/EncryptedStore.kt` (AES-GCM, key in the Android Keystore) | `Storage/EncryptedStore.swift` |
| Model | `model/AppModel.kt`, `EnrollmentFlow.kt`, `EnrollmentController.kt`, `PatternAnalyzer.kt`, `Exporter.kt`, `Records.kt`, `AppSettings.kt` | `Model/*` |
| UI | `ui/*` (Compose): Calibrate, Who's holding?, People, Enroll, Lab and its tools | `Views/*` |

**Recording.** `MotionRecorder` registers the accelerometer, gyroscope, gravity, linear acceleration
and game rotation vector at 10 000 µs on a `HandlerThread` and keeps the last 12 s in a ring buffer.
Values are converted to the iOS conventions of SPEC §4, so Android recordings go through the same
pipeline and research tools:

- `accel` = −(m/s²) / 9.80665 per axis (Android reports +1 g on z when the phone lies face up,
  Core Motion −1 g); `gravity` and `userAccel` (from the linear acceleration sensor) likewise;
- `gyro` and `rotation` in rad/s, unchanged;
- `attitude` = unit quaternion (w, x, y, z) from the game rotation vector;
- every timestamp, including `hapticStart`, is `elapsedRealtimeNanos` in seconds.

The motion stream is emitted on each gravity event with the latest linear acceleration, gyroscope rate
and attitude. Without a gravity sensor it is derived from a low-pass filter of the accelerometer.

**Haptics.** `HapticTimeline` lays the whole challenge out on a 1 ms timeline — taps are 12 ms pulses
at round(intensity · 255), ramps are 10 ms steps whose amplitude follows the ramp curve, gaps are
silence — and run-length encodes it into one `VibrationEffect.createWaveform`, so a pattern plays from
a single `vibrate()` call with pattern time 0 at its start. Without amplitude control, taps use the
default amplitude and ramp steps are pulse-width modulated. A composition cannot hold waveform segments
on public APIs, so composition primitives (click, or tick when sharpness > 0.66, API 30+) are only used
for tap-only patterns; generated challenges always contain ramps. `hapticStart` is read right before
`vibrate()`. On Android 13+ the vibration uses the media usage.

**Checks.** Same timing and logic as iOS (SPEC §12, §9.4, §14.3): the recorder is warm at least 0.5 s,
the seed is 8 bytes from `SecureRandom`, the window is
`[start − preRoll − 0.1, start + duration + postRoll + 0.1]`, up to `maxPatterns` patterns with up to
3 retries per pattern on a quality failure (SPEC §6.3 messages). Both recognizers decide on the same
patterns; v1 answers by default and v0 is logged beside it (DECISIONS D12). If the ONNX model cannot be
loaded, v0 answers and the screens say so.

**Storage.** Every file is AES-256-GCM encrypted (12-byte IV + ciphertext + tag) with a key generated in
and never leaving the Android Keystore, under `filesDir/auditor`: `people.json`, `samples.json`,
`embeddings.json`, `calibration.json`, `settings.json`, `checks.json` and `recordings/<uuid>.json`. The
JSON inside matches the iOS files. Backup and device transfer are disabled.

**Export.** Lab → Export writes `recordings.jsonl`, `features.csv` and `checks.csv` (optionally with names
replaced by P01, P02, …) to the app cache and shares them through a `FileProvider` and the system share
sheet. Exported files are not encrypted.

## Differences from iOS

- **Vibration motors vary.** iPhones have one kind of Taptic Engine; Android phones range from strong
  linear resonant actuators to weak eccentric-mass motors, with or without amplitude control. The v1
  encoder was trained on simulated iPhone-like actuators, so its embeddings may transfer less well than
  the hand-made v0 features. Compare both in the Lab (Test run shows a confusion matrix per recognizer)
  and pick the recognizer in Lab → Settings.
- **Sensor rates.** Lab → Device shows the measured accelerometer, gyroscope and motion rates and the
  vibrator's capabilities. Some phones deliver fewer than 100 samples per second; the quality gate then
  reports "Motion sensors are too slow on this device".
- **Device field.** Recordings carry `device.model` = manufacturer and model, `device.system` =
  "Android <version> (API <level>)".
- **Navigation.** Enrollment is a full screen instead of a sheet; Back or Cancel discards it.

## Privacy

The app has no INTERNET permission: nothing leaves the phone except files you export and share yourself.
Names, samples and raw recordings are stored encrypted with a hardware-backed key where the phone
supports it. "Delete all data" in the Lab removes every file; uninstalling the app removes the files and
the key.

## Build

CI builds and tests with `gradle -p 7oda-Auditor/android --no-daemon test assembleRelease` (Gradle 8.11.1,
JDK 17). The app's JVM unit tests (`android/app/src/test`) cover the haptic timeline, the unit
conversions, the enrollment state machine, the export files and JSON round trips, and the encrypted
store with a software key.
