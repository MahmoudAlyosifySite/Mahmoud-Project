# `:core` public API (package `com.mahmoudalyosify.auditor.core`)

Pure Kotlin/JVM port of `AuditorCore` (Swift). Same type and function names as SPEC §10.1 / §14.5.
Conventions:

- Swift `static let default` is `companion object { val default }` (`ChallengeConfig.default`, `SignalConfig.default`, ...).
  All config classes are data classes with the same defaults in the constructor, so `X()` == `X.default`.
- Swift enums with raw values are Kotlin enums whose constant names equal the JSON raw values
  (`Hand.right`, `Accessory.none`, `Posture.freshGrip`, ...). Each has `rawValue` and `companion fun from(raw: String): T?`.
- `Date` is `java.time.Instant`, `UUID` is `java.util.UUID`, `UInt64` is `ULong`.
- Public value types use `List<Double>`; signal arrays inside `ProcessedSignal` and the DSP use `DoubleArray`.
- Swift `Range<Int>` (half-open) is a Kotlin `IntRange` built with `until` (`start = first`, `end = last + 1`).

## Random (SPEC 2)
```kotlin
class SplitMix64(seed: ULong) { var state: ULong; fun next(): ULong }
class Xoshiro256StarStar(seed: ULong) {
    fun next(): ULong
    fun uniform(): Double                    // [0, 1)
    fun uniform(a: Double, b: Double): Double
    fun int(lo: Int, hi: Int): Int           // inclusive
}
object SeedFormat { fun string(seed: ULong): String; fun parse(s: String): ULong? }   // "0x" + 16 hex digits
```

## Challenge (SPEC 3)
```kotlin
data class CurvePoint(val time: Double, val intensity: Double)
data class TapEvent(val time: Double, val intensity: Double, val sharpness: Double)
data class RampEvent(val time: Double, val duration: Double, val sharpness: Double, val curve: List<CurvePoint>) {
    fun intensity(at: Double): Double        // tau relative to the ramp start
}
sealed class ChallengeEvent {
    abstract val time: Double
    data class Tap(val event: TapEvent) : ChallengeEvent()
    data class Ramp(val event: RampEvent) : ChallengeEvent()
}
data class ChallengeSpec(val version: Int = 1, val seed: ULong, val duration: Double, val events: List<ChallengeEvent>) {
    val taps: List<TapEvent>; val ramps: List<RampEvent>
}
data class ChallengeConfig(leadSilence, tapCount: IntRange, tapIntensity, tapSharpness, tapGap, rampDuration,
                           rampSharpness, rampLow, rampMid, rampHigh, rampGap /* ClosedFloatingPointRange<Double> */) {
    val rampCount: Int /* 2 */; companion object { val default: ChallengeConfig }
}
object ChallengeGenerator { fun generate(seed: ULong, config: ChallengeConfig = ChallengeConfig.default): ChallengeSpec }
```
Haptic playback (app): taps are transient events at `time` with `intensity`/`sharpness`; ramps are continuous
events of `duration` whose intensity follows `curve` (piecewise linear, `intensity(at)`).

## Recording (SPEC 4)
```kotlin
data class SensorStream(val t: List<Double> = emptyList(), val x: ..., val y: ..., val z: ...) { companion object { val empty } }
data class MotionStream(val t, gravityX, gravityY, gravityZ, userAccelX, userAccelY, userAccelZ,
                        rotationX, rotationY, rotationZ, attitudeW, attitudeX, attitudeY, attitudeZ: List<Double> = emptyList())
    { companion object { val empty } }
enum class Hand { right, left }
enum class Accessory { none, watch, ring, bracelet, other }
enum class Posture { sitting, standing, freshGrip, unspecified }
enum class RecordingPurpose { calibration, enrollment, accessory, check, test, signalCheck }
data class RecordingLabel(val personID: String, val personName: String, val hand: Hand,
                          val accessory: Accessory = Accessory.none, val posture: Posture = Posture.unspecified, val sessionID: String)
data class DeviceInfo(val model: String, val system: String, val app: String)
data class Recording(
    val schema: String = Recording.currentSchema, val id: UUID = UUID.randomUUID(), val createdAt: Instant = Instant.now(),
    val purpose: RecordingPurpose, val label: RecordingLabel? = null, val challenge: ChallengeSpec,
    val hapticStart: Double,            // seconds, same clock as the sensor timestamps
    val accel: SensorStream,            // g, INCLUDING gravity (Android: m/s² / 9.80665)
    val gyro: SensorStream,             // rad/s
    val motion: MotionStream = MotionStream.empty, val device: DeviceInfo? = null
) { companion object { const val currentSchema = "7oda-auditor.recording.v1" } }

object RecordingCoding {
    // JSON identical in structure to the iOS export: sorted keys, ISO-8601 UTC dates without fractional
    // seconds, seeds as hex strings, absent optional keys, uppercase UUIDs, non-finite numbers as "Infinity"/"-Infinity"/"NaN".
    fun encode(recording: Recording): String                 // one line
    fun decodeRecording(text: String): Recording             // throws IllegalArgumentException / SerializationException
    fun jsonLines(recordings: List<Recording>): String       // one recording per line, each ends with '\n'
    fun parseJSONLines(text: String): List<Recording>        // skips blank lines and a UTF-8 BOM
    fun encode(spec: ChallengeSpec): String;           fun decodeChallengeSpec(text: String): ChallengeSpec
    fun encode(calibration: Calibration): String;      fun decodeCalibration(text: String): Calibration
    fun encode(report: QualityReport): String
    fun encodeTemplateSamples(samples: List<TemplateSample>): String;   fun decodeTemplateSamples(text: String): List<TemplateSample>
    fun encodeEmbeddingSamples(samples: List<EmbeddingSample>): String; fun decodeEmbeddingSamples(text: String): List<EmbeddingSample>
    fun formatDate(date: Instant): String;  fun parseDate(text: String): Instant?
    // JsonElement level (kotlinx.serialization.json is an `api` dependency):
    fun toJson(recording: Recording): JsonObject; fun recordingFromJson(e: JsonElement): Recording
    fun toJson(spec: ChallengeSpec): JsonObject;  fun challengeSpecFromJson(e: JsonElement): ChallengeSpec
    fun toJson(calibration: Calibration): JsonObject; fun calibrationFromJson(e: JsonElement): Calibration
    fun toJson(sample: TemplateSample): JsonObject;   fun templateSampleFromJson(e: JsonElement): TemplateSample
    fun toJson(sample: EmbeddingSample): JsonObject;  fun embeddingSampleFromJson(e: JsonElement): EmbeddingSample
}
```

## DSP (SPEC 5.2–5.4)
```kotlin
data class Biquad(val b0: Double, val b1: Double, val b2: Double, val a1: Double, val a2: Double) {
    fun filter(x: DoubleArray): DoubleArray
    fun filtfiltZ(x: DoubleArray, pad: Int, dcGain: Double): DoubleArray
    companion object { val butterworthQ: Double; fun highPass(cutoff: Double, fs: Double, q: Double): Biquad
                       fun lowPass(cutoff, fs, q): Biquad; fun bandPass(center: Double, fs: Double, q: Double): Biquad }
}
class Resampled(val values: DoubleArray, val missing: BooleanArray)
object Resampler { fun resample(t: DoubleArray, v: DoubleArray, grid: DoubleArray, gapTolerance: Double): Resampled }
object StreamCleaner { fun sortDedup(s: SensorStream): SensorStream; fun sortDedup(m: MotionStream): MotionStream }
```

## Signal (SPEC 5)
```kotlin
data class SignalConfig(val fs: Double = 100.0, ... /* SPEC 5.1 */) { companion object { val default } }
data class AlignmentResult(val lag: Int, val ncc: Double, val peakMargin: Double)
class ProcessedSignal {
    val config: SignalConfig; val challenge: ChallengeSpec; val n: Int; val gridStart: Double
    val accel: List<DoubleArray> /* 3×N raw */; val gyro; val gravity; val rotation; val accelHP; val gyroHP
    val energyAccel: DoubleArray; val energyGyro: DoubleArray; val noiseAccel: Double; val noiseGyro: Double
    val envelope: DoubleArray; val template: DoubleArray; val alignment: AlignmentResult
    val gapFraction: Double; val sampleRate: Double
    fun index(ofPatternTime: Double): Int
    val quietRange: IntRange; val eventRange: IntRange
}
object SignalProcessor { fun process(recording: Recording, config: SignalConfig = SignalConfig.default): ProcessedSignal }
```

## Quality (SPEC 6)
```kotlin
enum class QualityMode { handheld, onTable }
enum class QualityIssue { sensorGaps, lowSampleRate, resting, moving, weakAlignment, weakSignal, notResting; val message: String }
data class QualityMetrics(val gapFraction, sampleRate, quietGyroRMS, quietAccelRMS, gravityZ, motionRMS,
                          gravityAngleDeg, ncc, peakMargin, snr: Double, val lag: Int)
data class QualityConfig(/* SPEC 6.2 defaults */) { companion object { val default } }
data class QualityReport(val metrics: QualityMetrics, val issues: List<QualityIssue>, val passed: Boolean = issues.isEmpty())
object QualityGate { fun evaluate(p: ProcessedSignal, calibration: Calibration?, mode: QualityMode,
                                  config: QualityConfig = QualityConfig.default): QualityReport }
```

## Features and calibration (SPEC 7, 8)
```kotlin
object FeatureExtractor {
    const val version = 2
    val names: List<String>                      // 39
    val calibrationRelativeIndices: List<Int>
    fun extractAbsolute(p: ProcessedSignal): List<Double>
    fun extract(p: ProcessedSignal, calibration: Calibration?): List<Double>
}
data class Calibration(val version: Int = 1, val createdAt: Instant, val tableFeatures: List<Double>, val quietGyroRMS: Double,
                       val quietAccelRMS: Double, val sampleRate: Double, val ncc: Double, val snr: Double,
                       val patternsUsed: Int, val device: String) {
    companion object {
        const val patternCount = 20; const val minimumPassing = 12; const val failureMessage: String
        fun build(processed: List<ProcessedSignal>, reports: List<QualityReport>, device: String, now: Instant): Calibration?
    }
}
```

## v0 model (SPEC 9)
```kotlin
data class PrototypeKey(val personID: String, val hand: Hand, val accessory: Accessory = Accessory.none) : Comparable<PrototypeKey>
enum class SampleSource { enrollment, accessory, adaptive }
data class TemplateSample(val id: UUID = UUID.randomUUID(), val prototype: PrototypeKey, val source: SampleSource,
                          val createdAt: Instant = Instant.now(), val features: List<Double>,
                          val featureVersion: Int = FeatureExtractor.version, val recordingID: UUID? = null)
data class ModelConfig(val shrinkageMin: Double = 0.1, val looQuantile: Double = 0.95, val absFactor: Double = 1.5) { companion object { val default } }
class GaussianModel {
    val mean: List<Double>; val scale: List<Double>; val classMeans: List<List<Double>>; val classCounts: List<Int>
    val shrinkage: Double; val absThreshold: Double; val looDistances: List<Double>
    fun distances(x: List<Double>): List<Double>
    companion object { fun fit(features: List<List<Double>>, labels: List<Int>, classCount: Int, config: ModelConfig): GaussianModel? }
}
data class IdentifierConfig(val nameMargin: Double = 6.0, val handMargin: Double = 3.0, val maxPatterns: Int = 3,
                            val adaptiveMaxPerPrototype: Int = 20, val adaptiveNameFactor: Double = 2.0,
                            val adaptiveAbsFactor: Double = 0.75) { companion object { val default } }
enum class UnknownReason { noPeople, noMatch, ambiguous }
sealed class DecisionOutcome {
    data class Identified(val personID: String, val hand: Hand) : DecisionOutcome()
    data class IdentifiedHandUnclear(val personID: String) : DecisionOutcome()
    data object NeedMorePatterns : DecisionOutcome()
    data class Unknown(val reason: UnknownReason) : DecisionOutcome()
}
data class PersonScore(val personID: String, val probability: Double, val bestDistance: Double)
data class PrototypeDistance(val prototype: PrototypeKey, val distance: Double)
data class Decision(val outcome: DecisionOutcome, val patternsUsed: Int, val bestPrototype: PrototypeKey? = null,
                    val bestDistance: Double = +inf, val marginName: Double = +inf, val marginHand: Double = +inf,
                    val handLogOdds: Double = 0.0, val absThreshold: Double = 0.0,
                    val personScores: List<PersonScore> = emptyList() /* sorted desc */,
                    val prototypeDistances: List<PrototypeDistance> = emptyList() /* prototype order */)
class Identifier(samples: List<TemplateSample>, modelConfig: ModelConfig = ModelConfig.default,
                 config: IdentifierConfig = IdentifierConfig.default) {
    val prototypes: List<PrototypeKey>; val model: GaussianModel?; val handModel: GaussianModel?
    val modelConfig: ModelConfig; val config: IdentifierConfig
    fun decide(patterns: List<List<Double>>): Decision
    fun adaptiveSamples(decision: Decision, patterns: List<List<Double>>, now: Instant): List<TemplateSample>
    companion object { fun applyAdaptiveCap(samples: List<TemplateSample>, cap: Int): List<TemplateSample> }
}
class ConfusionMatrix(labels: List<String> = emptyList()) {
    val labels: List<String>; val counts: List<List<Int>>
    fun add(truth: String, predicted: String); val accuracy: Double; val total: Int; fun count(truth: String, predicted: String): Int
}
```

## v1 learned encoder (SPEC 14)
```kotlin
object EncoderInput {
    const val channels = 10; const val length = 256; const val patternStart = 25
    fun build(p: ProcessedSignal): FloatArray   // row-major [channel][u], size 2560 -> ONNX input "x", shape [1, 10, 256]
}
data class EmbeddingSample(val prototype: PrototypeKey, val embedding: List<Double>,
                           val source: SampleSource = SampleSource.enrollment, val createdAt: Instant = Instant.now())
data class EmbeddingConfig(val looQuantile: Double = 0.95, val slack: Double = 0.0, val nameMargin: Double = 0.03,
                           val handMargin: Double = 0.0, val handShrinkage: Double = 0.1, val maxPatterns: Int = 3,
                           val adaptiveMaxPerPrototype: Int = 20) { companion object { val default } }
class EmbeddingIdentifier(samples: List<EmbeddingSample>, config: EmbeddingConfig = EmbeddingConfig.default) {
    val prototypes: List<PrototypeKey>; val absThreshold: Double; val config: EmbeddingConfig
    fun decide(patterns: List<List<Double>>): Decision        // cosine-distance units
    fun adaptiveSamples(decision: Decision, patterns: List<List<Double>>, now: Instant): List<EmbeddingSample>
    companion object { fun applyAdaptiveCap(samples: List<EmbeddingSample>, cap: Int): List<EmbeddingSample> }
}
```
ONNX model `app/src/main/assets/hand_encoder.onnx`: input `x` float32 `[1, 10, 256]`; outputs `embedding` float32
`[1, 128]` (L2-normalized) and `hand` `[1, 3]` (logits right/left/twoHanded; the decision uses the embedding hand model
instead). Pass `embedding` to `EmbeddingIdentifier` as `List<Double>`.

## Typical check flow
```kotlin
val spec = ChallengeGenerator.generate(seed)                    // play haptics, record sensors
val rec = Recording(purpose = RecordingPurpose.check, challenge = spec, hapticStart = t0, accel = a, gyro = g, motion = m)
val p = SignalProcessor.process(rec)
val q = QualityGate.evaluate(p, calibration, QualityMode.handheld)
if (q.passed) {
    val f = FeatureExtractor.extract(p, calibration)            // v0
    val x = EncoderInput.build(p)                               // v1 -> ONNX -> embedding
}
val decision = EmbeddingIdentifier(samples).decide(listOfEmbeddings)   // or Identifier(templateSamples).decide(features)
```
