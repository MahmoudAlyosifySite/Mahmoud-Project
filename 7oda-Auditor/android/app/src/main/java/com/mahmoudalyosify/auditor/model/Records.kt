@file:UseSerializers(UUIDSerializer::class, InstantSerializer::class)

package com.mahmoudalyosify.auditor.model

import com.mahmoudalyosify.auditor.core.Decision
import com.mahmoudalyosify.auditor.core.DecisionOutcome
import com.mahmoudalyosify.auditor.core.EmbeddingSample
import com.mahmoudalyosify.auditor.core.Hand
import com.mahmoudalyosify.auditor.core.PrototypeKey
import com.mahmoudalyosify.auditor.core.RecordingCoding
import com.mahmoudalyosify.auditor.core.RecordingPurpose
import com.mahmoudalyosify.auditor.core.SampleSource
import kotlinx.serialization.KSerializer
import kotlinx.serialization.Serializable
import kotlinx.serialization.UseSerializers
import kotlinx.serialization.descriptors.PrimitiveKind
import kotlinx.serialization.descriptors.PrimitiveSerialDescriptor
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonDecoder
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonEncoder
import java.time.Instant
import java.util.UUID

/** JSON settings shared by the app's own files: tolerant of missing and unknown keys. */
val AppJson: Json = Json {
    ignoreUnknownKeys = true
    explicitNulls = false
    encodeDefaults = true
    allowSpecialFloatingPointValues = true
}

/** UUIDs are written uppercase, like Foundation's `UUID.uuidString`. */
object UUIDSerializer : KSerializer<UUID> {
    override val descriptor: SerialDescriptor = PrimitiveSerialDescriptor("UUID", PrimitiveKind.STRING)
    override fun serialize(encoder: Encoder, value: UUID) = encoder.encodeString(value.toString().uppercase())
    override fun deserialize(decoder: Decoder): UUID = UUID.fromString(decoder.decodeString())
}

/** Dates are ISO-8601 UTC without fractional seconds (SPEC 4). */
object InstantSerializer : KSerializer<Instant> {
    override val descriptor: SerialDescriptor = PrimitiveSerialDescriptor("Instant", PrimitiveKind.STRING)
    override fun serialize(encoder: Encoder, value: Instant) = encoder.encodeString(RecordingCoding.formatDate(value))
    override fun deserialize(decoder: Decoder): Instant {
        val text = decoder.decodeString()
        return RecordingCoding.parseDate(text) ?: Instant.parse(text)
    }
}

/** Core embedding samples go through the core's own JSON mapping. */
object EmbeddingSampleSerializer : KSerializer<EmbeddingSample> {
    override val descriptor: SerialDescriptor = JsonElement.serializer().descriptor
    override fun serialize(encoder: Encoder, value: EmbeddingSample) {
        val json = encoder as? JsonEncoder ?: error("EmbeddingSample can only be written as JSON")
        json.encodeJsonElement(RecordingCoding.toJson(value))
    }

    override fun deserialize(decoder: Decoder): EmbeddingSample {
        val json = decoder as? JsonDecoder ?: error("EmbeddingSample can only be read from JSON")
        return RecordingCoding.embeddingSampleFromJson(json.decodeJsonElement())
    }
}

/** Which recognizer answers a check. */
@Serializable
enum class RecognizerKind {
    /** Learned encoder + cosine prototypes (SPEC 14). */
    v1,
    /** Hand-made features + Gaussian model (SPEC 9). */
    v0;

    val title: String
        get() = when (this) {
            v1 -> "v1 (learned encoder)"
            v0 -> "v0 (hand-made features)"
        }

    val other: RecognizerKind get() = if (this == v1) v0 else v1
}

/** An enrolled person. `id` is the `personID` of prototypes and labels. */
@Serializable
data class Person(
    val id: String = UUID.randomUUID().toString().uppercase(),
    val name: String,
    val createdAt: Instant = Instant.now(),
)

/**
 * A v1 embedding sample with the app-side links: the raw recording it came from and the
 * encoder version that produced it (so it can be recomputed when the model changes).
 */
@Serializable
data class StoredEmbedding(
    val id: UUID = UUID.randomUUID(),
    val recordingID: UUID? = null,
    val encoderVersion: String,
    @Serializable(with = EmbeddingSampleSerializer::class)
    val sample: EmbeddingSample,
) {
    val prototype: PrototypeKey get() = sample.prototype
    val source: SampleSource get() = sample.source
}

/**
 * Keeps at most `cap` adaptive embeddings per prototype (the newest); enrollment and
 * accessory samples are never dropped. Same rule as `Identifier.applyAdaptiveCap`.
 */
fun applyAdaptiveCap(items: List<StoredEmbedding>, cap: Int): List<StoredEmbedding> {
    val limit = maxOf(cap, 0)
    val adaptiveByPrototype = LinkedHashMap<PrototypeKey, MutableList<Int>>()
    items.forEachIndexed { i, item ->
        if (item.source == SampleSource.adaptive) adaptiveByPrototype.getOrPut(item.prototype) { mutableListOf() }.add(i)
    }
    val drop = HashSet<Int>()
    for (indices in adaptiveByPrototype.values) {
        if (indices.size <= limit) continue
        val ordered = indices.sortedWith(compareBy<Int>({ items[it].sample.createdAt }, { it }))
        drop.addAll(ordered.take(indices.size - limit))
    }
    return items.filterIndexed { i, _ -> i !in drop }
}

/** Compact, storable summary of one recognizer's decision. */
@Serializable
data class DecisionSummary(
    /** identified | handUnclear | needMorePatterns | unknown */
    val outcome: String,
    val personID: String? = null,
    val hand: Hand? = null,
    val reason: String? = null,
    val patternsUsed: Int,
    val bestDistance: Double? = null,
    val marginName: Double? = null,
    val marginHand: Double? = null,
    val handLogOdds: Double,
    val absThreshold: Double,
) {
    companion object {
        fun of(d: Decision): DecisionSummary {
            fun finite(x: Double): Double? = if (x.isFinite()) x else null
            val base = DecisionSummary(
                outcome = "", patternsUsed = d.patternsUsed,
                bestDistance = finite(d.bestDistance), marginName = finite(d.marginName),
                marginHand = finite(d.marginHand), handLogOdds = d.handLogOdds, absThreshold = d.absThreshold,
            )
            return when (val o = d.outcome) {
                is DecisionOutcome.Identified -> base.copy(outcome = "identified", personID = o.personID, hand = o.hand)
                is DecisionOutcome.IdentifiedHandUnclear -> base.copy(outcome = "handUnclear", personID = o.personID)
                DecisionOutcome.NeedMorePatterns -> base.copy(outcome = "needMorePatterns")
                is DecisionOutcome.Unknown -> base.copy(outcome = "unknown", reason = o.reason.rawValue)
            }
        }
    }
}

/** One finished check, logged for the checks CSV (both recognizers' outcomes). */
@Serializable
data class CheckLogEntry(
    val id: UUID,
    val createdAt: Instant,
    val purpose: RecordingPurpose,
    val recognizer: RecognizerKind,
    val recordingIDs: List<UUID>,
    val patternsUsed: Int,
    val attempts: Int,
    val seconds: Double,
    val v0: DecisionSummary? = null,
    val v1: DecisionSummary? = null,
    val truthPersonID: String? = null,
    val truthHand: Hand? = null,
    val requestedHand: Hand? = null,
    val challengePassed: Boolean? = null,
    val qualityFailure: String? = null,
)
