package com.mahmoudalyosify.auditor.model

import com.mahmoudalyosify.auditor.core.EmbeddingConfig
import com.mahmoudalyosify.auditor.core.IdentifierConfig
import com.mahmoudalyosify.auditor.core.ModelConfig
import com.mahmoudalyosify.auditor.core.QualityConfig
import kotlinx.serialization.Serializable

/** User settings (Lab). Missing keys decode to their defaults so older files keep working. */
@Serializable
data class AppSettings(
    val recognizer: RecognizerKind = RecognizerKind.v1,
    // v0 (SPEC 9.2 / 9.4)
    val absFactor: Double = ModelConfig.default.absFactor,
    val nameMargin: Double = IdentifierConfig.default.nameMargin,
    val handMargin: Double = IdentifierConfig.default.handMargin,
    // v1 (SPEC 14.3)
    val slack: Double = EmbeddingConfig.default.slack,
    val v1NameMargin: Double = EmbeddingConfig.default.nameMargin,
    // shared
    val maxPatterns: Int = IdentifierConfig.default.maxPatterns,
    val minNCC: Double = QualityConfig.default.minNCC,
    val minPeakMargin: Double = QualityConfig.default.minPeakMargin,
    val handChallenge: Boolean = false,
    val learnFromConfidentChecks: Boolean = false,
    val anonymizeExports: Boolean = false,
    val calibrationSkipped: Boolean = false,
) {
    val modelConfig: ModelConfig get() = ModelConfig.default.copy(absFactor = absFactor)

    val identifierConfig: IdentifierConfig
        get() = IdentifierConfig.default.copy(
            nameMargin = nameMargin, handMargin = handMargin, maxPatterns = maxOf(1, maxPatterns),
        )

    val embeddingConfig: EmbeddingConfig
        get() = EmbeddingConfig.default.copy(
            slack = slack, nameMargin = v1NameMargin, maxPatterns = maxOf(1, maxPatterns),
        )

    val qualityConfig: QualityConfig
        get() = QualityConfig.default.copy(minNCC = minNCC, minPeakMargin = minPeakMargin)
}
