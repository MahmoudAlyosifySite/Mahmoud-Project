package com.mahmoudalyosify.auditor.model

import com.mahmoudalyosify.auditor.core.FeatureExtractor
import com.mahmoudalyosify.auditor.core.Recording
import com.mahmoudalyosify.auditor.core.RecordingCoding
import com.mahmoudalyosify.auditor.core.TemplateSample
import java.util.Locale

/** Export files of the Lab: recordings (JSON Lines), v0 features (CSV) and checks (CSV). */
object Exporter {
    /** personID → display name; with `anonymize`, P01, P02, … in enrollment order. */
    fun nameMap(people: List<Person>, anonymize: Boolean): Map<String, String> {
        val ordered = people.sortedWith(compareBy<Person>({ it.createdAt }, { it.id }))
        val map = LinkedHashMap<String, String>()
        ordered.forEachIndexed { i, p ->
            val code = if (i + 1 < 10) "P0${i + 1}" else "P${i + 1}"
            map[p.id] = if (anonymize) code else p.name
        }
        return map
    }

    /** Recordings with label names taken from `names` (current or anonymized names). */
    fun prepare(recordings: List<Recording>, names: Map<String, String>): List<Recording> =
        recordings.map { r ->
            val label = r.label ?: return@map r
            r.copy(label = label.copy(personName = names[label.personID] ?: "P??"))
        }

    fun csvField(s: String): String =
        if (s.contains(',') || s.contains('"') || s.contains('\n')) "\"" + s.replace("\"", "\"\"") + "\"" else s

    /** `%.6g`-style number; empty for null, inf / -inf / nan for non-finite values. */
    fun number(x: Double?): String {
        if (x == null) return ""
        if (!x.isFinite()) return if (x > 0) "inf" else if (x < 0) "-inf" else "nan"
        if (x == 0.0) return "0"
        val formatted = String.format(Locale.US, "%.6g", x)
        // Like C's %g: drop trailing zeros of the fraction and use a short exponent.
        val (mantissa, exponent) = formatted.lowercase(Locale.US).split('e').let { parts ->
            parts[0] to parts.getOrNull(1)
        }
        val trimmed = if (mantissa.contains('.')) mantissa.trimEnd('0').trimEnd('.') else mantissa
        if (exponent == null) return trimmed
        val sign = if (exponent.startsWith("-")) "-" else "+"
        val digits = exponent.trimStart('+', '-').trimStart('0').padStart(2, '0')
        return "${trimmed}e$sign$digits"
    }

    /** One row per v0 template sample. */
    fun featuresCSV(samples: List<TemplateSample>, names: Map<String, String>): String {
        val width = samples.maxOfOrNull { it.features.size } ?: FeatureExtractor.names.size
        val featureNames = (0 until width).map { k -> FeatureExtractor.names.getOrNull(k) ?: "f$k" }
        val header = listOf("sample_id", "recording_id", "person", "person_id", "hand", "accessory", "source",
            "created_at", "feature_version") + featureNames
        val lines = mutableListOf(header.joinToString(",") { csvField(it) })
        for (s in samples) {
            val row = mutableListOf(
                s.id.toString().uppercase(), s.recordingID?.toString()?.uppercase() ?: "",
                names[s.prototype.personID] ?: "P??", s.prototype.personID, s.prototype.hand.rawValue,
                s.prototype.accessory.rawValue, s.source.rawValue, RecordingCoding.formatDate(s.createdAt),
                s.featureVersion.toString(),
            )
            row += (0 until width).map { k -> if (k < s.features.size) number(s.features[k]) else "" }
            lines.add(row.joinToString(",") { csvField(it) })
        }
        return lines.joinToString("\n") + "\n"
    }

    private fun decisionColumns(prefix: String): List<String> =
        listOf("outcome", "person", "hand", "reason", "patterns", "best_distance", "margin_name", "margin_hand",
            "hand_log_odds", "abs_threshold").map { "${prefix}_$it" }

    private fun decisionFields(d: DecisionSummary?, names: Map<String, String>): List<String> {
        if (d == null) return List(10) { "" }
        val person = d.personID?.let { names[it] ?: "P??" } ?: ""
        return listOf(d.outcome, person, d.hand?.rawValue ?: "", d.reason ?: "", d.patternsUsed.toString(),
            number(d.bestDistance), number(d.marginName), number(d.marginHand),
            number(d.handLogOdds), number(d.absThreshold))
    }

    /** One row per check with both recognizers' outcomes. */
    fun checksCSV(entries: List<CheckLogEntry>, names: Map<String, String>): String {
        val header = listOf("check_id", "created_at", "purpose", "answered_by", "patterns_used", "attempts", "seconds",
            "truth_person", "truth_hand", "requested_hand", "challenge_passed", "quality_failure",
            "recording_ids") + decisionColumns("v0") + decisionColumns("v1")
        val lines = mutableListOf(header.joinToString(",") { csvField(it) })
        for (e in entries) {
            val truthName = e.truthPersonID?.let { names[it] ?: "P??" } ?: ""
            val challenge = e.challengePassed?.let { if (it) "yes" else "no" } ?: ""
            val recordingList = e.recordingIDs.joinToString(" ") { it.toString().uppercase() }
            val row = mutableListOf(e.id.toString().uppercase(), RecordingCoding.formatDate(e.createdAt),
                e.purpose.rawValue, e.recognizer.name)
            row += listOf(e.patternsUsed.toString(), e.attempts.toString(), number(e.seconds))
            row += listOf(truthName, e.truthHand?.rawValue ?: "", e.requestedHand?.rawValue ?: "", challenge)
            row += listOf(e.qualityFailure ?: "", recordingList)
            row += decisionFields(e.v0, names)
            row += decisionFields(e.v1, names)
            lines.add(row.joinToString(",") { csvField(it) })
        }
        return lines.joinToString("\n") + "\n"
    }
}
