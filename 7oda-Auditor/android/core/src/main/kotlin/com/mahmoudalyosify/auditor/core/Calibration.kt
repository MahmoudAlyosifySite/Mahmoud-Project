package com.mahmoudalyosify.auditor.core

import java.time.Instant

/** Per-device table calibration (SPEC 8). */
data class Calibration(
    val version: Int = 1,
    val createdAt: Instant,
    val tableFeatures: List<Double>,
    val quietGyroRMS: Double,
    val quietAccelRMS: Double,
    val sampleRate: Double,
    val ncc: Double,
    val snr: Double,
    val patternsUsed: Int,
    val device: String,
) {
    companion object {
        const val patternCount: Int = 20
        const val minimumPassing: Int = 12
        const val failureMessage: String = "Calibration failed: keep the phone still on a hard table"

        /** Uses only `processed[i]` whose `reports[i].passed`; null if fewer than [minimumPassing]. */
        fun build(processed: List<ProcessedSignal>, reports: List<QualityReport>, device: String, now: Instant): Calibration? {
            val used = processed.zip(reports).filter { it.second.passed }
            if (used.size < minimumPassing) return null
            val features = used.map { FeatureExtractor.extractAbsolute(it.first) }
            val d = features.minOfOrNull { it.size } ?: 0
            val table = List(d) { k -> Numerics.median(features.map { it[k] }) }
            fun med(value: (QualityMetrics) -> Double): Double = Numerics.median(used.map { value(it.second.metrics) })
            return Calibration(
                version = 1,
                createdAt = now,
                tableFeatures = table,
                quietGyroRMS = med { it.quietGyroRMS },
                quietAccelRMS = med { it.quietAccelRMS },
                sampleRate = med { it.sampleRate },
                ncc = med { it.ncc },
                snr = med { it.snr },
                patternsUsed = used.size,
                device = device,
            )
        }
    }
}
