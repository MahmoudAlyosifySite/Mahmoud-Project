package com.mahmoudalyosify.auditor.core

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import java.time.Instant
import java.time.OffsetDateTime
import java.time.format.DateTimeFormatter
import java.time.temporal.ChronoUnit
import java.util.Locale
import java.util.UUID

/**
 * JSON coding identical in structure to the iOS app (SPEC 4): sorted keys, ISO-8601 UTC dates without
 * fractional seconds, seeds as hex strings, optional keys left out when absent, uppercase UUIDs and
 * non-finite numbers written as the strings "Infinity", "-Infinity" and "NaN". Decoding errors throw
 * [IllegalArgumentException] (kotlinx `SerializationException` is a subclass).
 */
object RecordingCoding {

    // MARK: - Text level

    fun encode(recording: Recording): String = render(toJson(recording))

    fun decodeRecording(text: String): Recording = recordingFromJson(parse(text))

    /** One recording per line; every line ends with `\n`. */
    fun jsonLines(recordings: List<Recording>): String {
        val sb = StringBuilder()
        for (r in recordings) {
            sb.append(encode(r))
            sb.append('\n')
        }
        return sb.toString()
    }

    /** Parses JSON Lines, skipping blank lines and a leading UTF-8 byte-order mark. */
    fun parseJSONLines(text: String): List<Recording> {
        val body = if (text.startsWith("﻿")) text.substring(1) else text
        val out = ArrayList<Recording>()
        for (line in body.split('\n')) {
            val trimmed = line.trim { it == ' ' || it == '\t' || it == '\r' || it == '\n' }
            if (trimmed.isNotEmpty()) out.add(recordingFromJson(parse(trimmed)))
        }
        return out
    }

    fun encode(spec: ChallengeSpec): String = render(toJson(spec))

    fun decodeChallengeSpec(text: String): ChallengeSpec = challengeSpecFromJson(parse(text))

    fun encode(calibration: Calibration): String = render(toJson(calibration))

    fun decodeCalibration(text: String): Calibration = calibrationFromJson(parse(text))

    fun encode(report: QualityReport): String = render(toJson(report))

    fun encodeTemplateSamples(samples: List<TemplateSample>): String = render(JsonArray(samples.map { toJson(it) }))

    fun decodeTemplateSamples(text: String): List<TemplateSample> = array(parse(text)).map { templateSampleFromJson(it) }

    fun encodeEmbeddingSamples(samples: List<EmbeddingSample>): String = render(JsonArray(samples.map { toJson(it) }))

    fun decodeEmbeddingSamples(text: String): List<EmbeddingSample> = array(parse(text)).map { embeddingSampleFromJson(it) }

    /** `2026-11-24T10:00:00Z` (UTC, fractional seconds dropped). */
    fun formatDate(date: Instant): String = DateTimeFormatter.ISO_INSTANT.format(date.truncatedTo(ChronoUnit.SECONDS))

    /** Parses ISO-8601 with or without fractional seconds and with `Z` or an offset; null if invalid. */
    fun parseDate(text: String): Instant? {
        try {
            return Instant.parse(text)
        } catch (e: Exception) {
        }
        return try {
            OffsetDateTime.parse(text).toInstant()
        } catch (e: Exception) {
            null
        }
    }

    // MARK: - JsonElement level: encoding

    fun toJson(recording: Recording): JsonObject {
        val m = LinkedHashMap<String, JsonElement>()
        m["schema"] = JsonPrimitive(recording.schema)
        m["id"] = uuid(recording.id)
        m["createdAt"] = JsonPrimitive(formatDate(recording.createdAt))
        m["purpose"] = JsonPrimitive(recording.purpose.name)
        recording.label?.let { m["label"] = toJson(it) }
        m["challenge"] = toJson(recording.challenge)
        m["hapticStart"] = num(recording.hapticStart)
        m["accel"] = toJson(recording.accel)
        m["gyro"] = toJson(recording.gyro)
        m["motion"] = toJson(recording.motion)
        recording.device?.let { m["device"] = toJson(it) }
        return JsonObject(m)
    }

    fun toJson(label: RecordingLabel): JsonObject = JsonObject(
        linkedMapOf(
            "personID" to JsonPrimitive(label.personID),
            "personName" to JsonPrimitive(label.personName),
            "hand" to JsonPrimitive(label.hand.name),
            "accessory" to JsonPrimitive(label.accessory.name),
            "posture" to JsonPrimitive(label.posture.name),
            "sessionID" to JsonPrimitive(label.sessionID),
        ),
    )

    fun toJson(device: DeviceInfo): JsonObject = JsonObject(
        linkedMapOf(
            "model" to JsonPrimitive(device.model),
            "system" to JsonPrimitive(device.system),
            "app" to JsonPrimitive(device.app),
        ),
    )

    fun toJson(stream: SensorStream): JsonObject = JsonObject(
        linkedMapOf("t" to nums(stream.t), "x" to nums(stream.x), "y" to nums(stream.y), "z" to nums(stream.z)),
    )

    fun toJson(motion: MotionStream): JsonObject = JsonObject(
        linkedMapOf(
            "t" to nums(motion.t),
            "gravityX" to nums(motion.gravityX), "gravityY" to nums(motion.gravityY), "gravityZ" to nums(motion.gravityZ),
            "userAccelX" to nums(motion.userAccelX), "userAccelY" to nums(motion.userAccelY),
            "userAccelZ" to nums(motion.userAccelZ),
            "rotationX" to nums(motion.rotationX), "rotationY" to nums(motion.rotationY), "rotationZ" to nums(motion.rotationZ),
            "attitudeW" to nums(motion.attitudeW), "attitudeX" to nums(motion.attitudeX),
            "attitudeY" to nums(motion.attitudeY), "attitudeZ" to nums(motion.attitudeZ),
        ),
    )

    fun toJson(spec: ChallengeSpec): JsonObject = JsonObject(
        linkedMapOf(
            "version" to JsonPrimitive(spec.version),
            "seed" to JsonPrimitive(SeedFormat.string(spec.seed)),
            "duration" to num(spec.duration),
            "events" to JsonArray(spec.events.map { toJson(it) }),
        ),
    )

    fun toJson(event: ChallengeEvent): JsonObject = when (event) {
        is ChallengeEvent.Tap -> JsonObject(
            linkedMapOf(
                "kind" to JsonPrimitive("tap"),
                "time" to num(event.event.time),
                "intensity" to num(event.event.intensity),
                "sharpness" to num(event.event.sharpness),
            ),
        )
        is ChallengeEvent.Ramp -> JsonObject(
            linkedMapOf(
                "kind" to JsonPrimitive("ramp"),
                "time" to num(event.event.time),
                "duration" to num(event.event.duration),
                "sharpness" to num(event.event.sharpness),
                "curve" to JsonArray(
                    event.event.curve.map {
                        JsonObject(linkedMapOf("time" to num(it.time), "intensity" to num(it.intensity)))
                    },
                ),
            ),
        )
    }

    fun toJson(calibration: Calibration): JsonObject = JsonObject(
        linkedMapOf(
            "version" to JsonPrimitive(calibration.version),
            "createdAt" to JsonPrimitive(formatDate(calibration.createdAt)),
            "tableFeatures" to nums(calibration.tableFeatures),
            "quietGyroRMS" to num(calibration.quietGyroRMS),
            "quietAccelRMS" to num(calibration.quietAccelRMS),
            "sampleRate" to num(calibration.sampleRate),
            "ncc" to num(calibration.ncc),
            "snr" to num(calibration.snr),
            "patternsUsed" to JsonPrimitive(calibration.patternsUsed),
            "device" to JsonPrimitive(calibration.device),
        ),
    )

    fun toJson(metrics: QualityMetrics): JsonObject = JsonObject(
        linkedMapOf(
            "gapFraction" to num(metrics.gapFraction),
            "sampleRate" to num(metrics.sampleRate),
            "quietGyroRMS" to num(metrics.quietGyroRMS),
            "quietAccelRMS" to num(metrics.quietAccelRMS),
            "gravityZ" to num(metrics.gravityZ),
            "motionRMS" to num(metrics.motionRMS),
            "gravityAngleDeg" to num(metrics.gravityAngleDeg),
            "ncc" to num(metrics.ncc),
            "peakMargin" to num(metrics.peakMargin),
            "snr" to num(metrics.snr),
            "lag" to JsonPrimitive(metrics.lag),
        ),
    )

    fun toJson(report: QualityReport): JsonObject = JsonObject(
        linkedMapOf(
            "metrics" to toJson(report.metrics),
            "issues" to JsonArray(report.issues.map { JsonPrimitive(it.name) }),
            "passed" to JsonPrimitive(report.passed),
        ),
    )

    fun toJson(key: PrototypeKey): JsonObject = JsonObject(
        linkedMapOf(
            "personID" to JsonPrimitive(key.personID),
            "hand" to JsonPrimitive(key.hand.name),
            "accessory" to JsonPrimitive(key.accessory.name),
        ),
    )

    fun toJson(sample: TemplateSample): JsonObject {
        val m = LinkedHashMap<String, JsonElement>()
        m["id"] = uuid(sample.id)
        m["prototype"] = toJson(sample.prototype)
        m["source"] = JsonPrimitive(sample.source.name)
        m["createdAt"] = JsonPrimitive(formatDate(sample.createdAt))
        m["features"] = nums(sample.features)
        m["featureVersion"] = JsonPrimitive(sample.featureVersion)
        sample.recordingID?.let { m["recordingID"] = uuid(it) }
        return JsonObject(m)
    }

    fun toJson(sample: EmbeddingSample): JsonObject = JsonObject(
        linkedMapOf(
            "prototype" to toJson(sample.prototype),
            "embedding" to nums(sample.embedding),
            "source" to JsonPrimitive(sample.source.name),
            "createdAt" to JsonPrimitive(formatDate(sample.createdAt)),
        ),
    )

    // MARK: - JsonElement level: decoding

    fun recordingFromJson(e: JsonElement): Recording {
        val o = obj(e)
        return Recording(
            schema = o.opt("schema")?.let { str(it) } ?: Recording.currentSchema,
            id = parseUUID(str(o.req("id"))),
            createdAt = date(o.req("createdAt")),
            purpose = enumValue(o.req("purpose")) { RecordingPurpose.from(it) },
            label = o.opt("label")?.let { labelFromJson(it) },
            challenge = challengeSpecFromJson(o.req("challenge")),
            hapticStart = dbl(o.req("hapticStart")),
            accel = o.opt("accel")?.let { sensorStreamFromJson(it) } ?: SensorStream.empty,
            gyro = o.opt("gyro")?.let { sensorStreamFromJson(it) } ?: SensorStream.empty,
            motion = o.opt("motion")?.let { motionStreamFromJson(it) } ?: MotionStream.empty,
            device = o.opt("device")?.let { deviceFromJson(it) },
        )
    }

    fun labelFromJson(e: JsonElement): RecordingLabel {
        val o = obj(e)
        return RecordingLabel(
            personID = str(o.req("personID")),
            personName = str(o.req("personName")),
            hand = enumValue(o.req("hand")) { Hand.from(it) },
            accessory = o.opt("accessory")?.let { enumValue(it) { Accessory.from(it) } } ?: Accessory.none,
            posture = o.opt("posture")?.let { enumValue(it) { Posture.from(it) } } ?: Posture.unspecified,
            sessionID = str(o.req("sessionID")),
        )
    }

    fun deviceFromJson(e: JsonElement): DeviceInfo {
        val o = obj(e)
        return DeviceInfo(str(o.req("model")), str(o.req("system")), str(o.req("app")))
    }

    fun sensorStreamFromJson(e: JsonElement): SensorStream {
        val o = obj(e)
        fun arr(key: String): List<Double> = o.opt(key)?.let { dbls(it) } ?: emptyList()
        return SensorStream(arr("t"), arr("x"), arr("y"), arr("z"))
    }

    fun motionStreamFromJson(e: JsonElement): MotionStream {
        val o = obj(e)
        fun arr(key: String): List<Double> = o.opt(key)?.let { dbls(it) } ?: emptyList()
        return MotionStream(
            t = arr("t"),
            gravityX = arr("gravityX"), gravityY = arr("gravityY"), gravityZ = arr("gravityZ"),
            userAccelX = arr("userAccelX"), userAccelY = arr("userAccelY"), userAccelZ = arr("userAccelZ"),
            rotationX = arr("rotationX"), rotationY = arr("rotationY"), rotationZ = arr("rotationZ"),
            attitudeW = arr("attitudeW"), attitudeX = arr("attitudeX"),
            attitudeY = arr("attitudeY"), attitudeZ = arr("attitudeZ"),
        )
    }

    fun challengeSpecFromJson(e: JsonElement): ChallengeSpec {
        val o = obj(e)
        val seedElement = o.req("seed")
        val seed: ULong = if (seedElement is JsonPrimitive && seedElement.isString) {
            SeedFormat.parse(seedElement.content) ?: throw IllegalArgumentException("Invalid seed '${seedElement.content}'")
        } else {
            val text = prim(seedElement).content
            text.toULongOrNull() ?: throw IllegalArgumentException("Invalid seed '$text'")
        }
        return ChallengeSpec(
            version = o.opt("version")?.let { int(it) } ?: 1,
            seed = seed,
            duration = dbl(o.req("duration")),
            events = array(o.req("events")).map { challengeEventFromJson(it) },
        )
    }

    fun challengeEventFromJson(e: JsonElement): ChallengeEvent {
        val o = obj(e)
        return when (val kind = str(o.req("kind"))) {
            "tap" -> ChallengeEvent.Tap(
                TapEvent(dbl(o.req("time")), dbl(o.req("intensity")), dbl(o.req("sharpness"))),
            )
            "ramp" -> ChallengeEvent.Ramp(
                RampEvent(
                    time = dbl(o.req("time")),
                    duration = dbl(o.req("duration")),
                    sharpness = dbl(o.req("sharpness")),
                    curve = array(o.req("curve")).map {
                        val p = obj(it)
                        CurvePoint(dbl(p.req("time")), dbl(p.req("intensity")))
                    },
                ),
            )
            else -> throw IllegalArgumentException("Unknown challenge event kind '$kind'")
        }
    }

    fun calibrationFromJson(e: JsonElement): Calibration {
        val o = obj(e)
        return Calibration(
            version = o.opt("version")?.let { int(it) } ?: 1,
            createdAt = date(o.req("createdAt")),
            tableFeatures = dbls(o.req("tableFeatures")),
            quietGyroRMS = dbl(o.req("quietGyroRMS")),
            quietAccelRMS = dbl(o.req("quietAccelRMS")),
            sampleRate = dbl(o.req("sampleRate")),
            ncc = dbl(o.req("ncc")),
            snr = dbl(o.req("snr")),
            patternsUsed = int(o.req("patternsUsed")),
            device = str(o.req("device")),
        )
    }

    fun prototypeKeyFromJson(e: JsonElement): PrototypeKey {
        val o = obj(e)
        return PrototypeKey(
            personID = str(o.req("personID")),
            hand = enumValue(o.req("hand")) { Hand.from(it) },
            accessory = o.opt("accessory")?.let { enumValue(it) { Accessory.from(it) } } ?: Accessory.none,
        )
    }

    fun templateSampleFromJson(e: JsonElement): TemplateSample {
        val o = obj(e)
        return TemplateSample(
            id = parseUUID(str(o.req("id"))),
            prototype = prototypeKeyFromJson(o.req("prototype")),
            source = enumValue(o.req("source")) { SampleSource.from(it) },
            createdAt = date(o.req("createdAt")),
            features = dbls(o.req("features")),
            featureVersion = int(o.req("featureVersion")),
            recordingID = o.opt("recordingID")?.let { parseUUID(str(it)) },
        )
    }

    fun embeddingSampleFromJson(e: JsonElement): EmbeddingSample {
        val o = obj(e)
        return EmbeddingSample(
            prototype = prototypeKeyFromJson(o.req("prototype")),
            embedding = dbls(o.req("embedding")),
            source = o.opt("source")?.let { enumValue(it) { SampleSource.from(it) } } ?: SampleSource.enrollment,
            createdAt = date(o.req("createdAt")),
        )
    }

    // MARK: - Helpers

    private val parser: Json = Json { isLenient = false }

    internal fun parse(text: String): JsonElement = parser.parseToJsonElement(text)

    /** Compact JSON with keys sorted at every level. */
    internal fun render(e: JsonElement): String = sortKeys(e).toString()

    internal fun sortKeys(e: JsonElement): JsonElement = when (e) {
        is JsonObject -> {
            val m = LinkedHashMap<String, JsonElement>()
            for (key in e.keys.sorted()) m[key] = sortKeys(e.getValue(key))
            JsonObject(m)
        }
        is JsonArray -> JsonArray(e.map { sortKeys(it) })
        else -> e
    }

    private fun uuid(id: UUID): JsonPrimitive = JsonPrimitive(id.toString().uppercase(Locale.ROOT))

    private fun parseUUID(text: String): UUID = UUID.fromString(text)

    internal fun num(v: Double): JsonPrimitive = when {
        v.isNaN() -> JsonPrimitive("NaN")
        v == Double.POSITIVE_INFINITY -> JsonPrimitive("Infinity")
        v == Double.NEGATIVE_INFINITY -> JsonPrimitive("-Infinity")
        else -> JsonPrimitive(v)
    }

    internal fun nums(values: List<Double>): JsonArray = JsonArray(values.map { num(it) })

    private fun obj(e: JsonElement): JsonObject =
        e as? JsonObject ?: throw IllegalArgumentException("expected a JSON object")

    private fun array(e: JsonElement): JsonArray =
        e as? JsonArray ?: throw IllegalArgumentException("expected a JSON array")

    private fun prim(e: JsonElement): JsonPrimitive {
        if (e is JsonNull || e !is JsonPrimitive) throw IllegalArgumentException("expected a JSON value")
        return e
    }

    private fun JsonObject.req(key: String): JsonElement {
        val v = this[key]
        if (v == null || v is JsonNull) throw IllegalArgumentException("missing key '$key'")
        return v
    }

    private fun JsonObject.opt(key: String): JsonElement? {
        val v = this[key]
        return if (v == null || v is JsonNull) null else v
    }

    private fun str(e: JsonElement): String {
        val p = prim(e)
        if (!p.isString) throw IllegalArgumentException("expected a string")
        return p.content
    }

    internal fun dbl(e: JsonElement): Double {
        val p = prim(e)
        if (p.isString) {
            return when (p.content) {
                "Infinity" -> Double.POSITIVE_INFINITY
                "-Infinity" -> Double.NEGATIVE_INFINITY
                "NaN" -> Double.NaN
                else -> throw IllegalArgumentException("expected a number, found '${p.content}'")
            }
        }
        return p.content.toDoubleOrNull() ?: throw IllegalArgumentException("expected a number, found '${p.content}'")
    }

    private fun dbls(e: JsonElement): List<Double> = array(e).map { dbl(it) }

    private fun int(e: JsonElement): Int {
        val p = prim(e)
        p.content.toIntOrNull()?.let { return it }
        val d = dbl(p)
        if (!d.isFinite() || d != Math.rint(d) || Math.abs(d) > Int.MAX_VALUE.toDouble()) {
            throw IllegalArgumentException("expected an integer, found '${p.content}'")
        }
        return d.toInt()
    }

    private fun date(e: JsonElement): Instant {
        val text = str(e)
        return parseDate(text) ?: throw IllegalArgumentException("Expected an ISO-8601 date, found '$text'")
    }

    private fun <T> enumValue(e: JsonElement, from: (String) -> T?): T {
        val text = str(e)
        return from(text) ?: throw IllegalArgumentException("unknown value '$text'")
    }
}
