package com.mahmoudalyosify.auditor.model

import com.mahmoudalyosify.auditor.core.Calibration
import com.mahmoudalyosify.auditor.core.FeatureExtractor
import com.mahmoudalyosify.auditor.core.ProcessedSignal
import com.mahmoudalyosify.auditor.core.QualityConfig
import com.mahmoudalyosify.auditor.core.QualityGate
import com.mahmoudalyosify.auditor.core.QualityMode
import com.mahmoudalyosify.auditor.core.QualityReport
import com.mahmoudalyosify.auditor.core.Recording
import com.mahmoudalyosify.auditor.core.SignalProcessor
import kotlin.math.sqrt

/** Output of the v1 encoder for one pattern. */
class EncoderOutput(
    /** L2-normalized embedding (128 values). */
    val embedding: List<Double>,
    /** Hand logits: right, left, twoHanded (logged only; the decision uses its own hand model). */
    val handLogits: List<Double>,
)

/** The v1 encoder as the app model sees it (the ONNX implementation lives in `ml`). */
interface PatternEncoder {
    /** Identifies the model that produced stored embeddings; a change triggers recomputation. */
    val version: String
    fun encode(processed: ProcessedSignal): EncoderOutput
}

/**
 * One recorded pattern after the whole pipeline: processing, quality gate, v0 features and
 * (when the model is available) the v1 embedding.
 */
class AnalyzedPattern(
    val recording: Recording,
    val processed: ProcessedSignal,
    val quality: QualityReport,
    /** v0 features relative to the calibration (SPEC 7). */
    val features: List<Double>,
    /** v1 embedding (SPEC 14), null when the encoder is unavailable or failed. */
    val embedding: List<Double>?,
    val handLogits: List<Double>?,
    val encoderError: String?,
) {
    val passed: Boolean get() = quality.passed

    /** User message of the first quality issue (SPEC 6.3), null when passed. */
    val qualityMessage: String? get() = quality.issues.firstOrNull()?.message
}

object PatternAnalyzer {
    /** SignalProcessor.process → QualityGate.evaluate → FeatureExtractor.extract and the encoder. */
    fun analyze(
        recording: Recording,
        mode: QualityMode,
        calibration: Calibration?,
        qualityConfig: QualityConfig,
        encoder: PatternEncoder?,
    ): AnalyzedPattern {
        val processed = SignalProcessor.process(recording)
        val quality = QualityGate.evaluate(processed, calibration, mode, qualityConfig)
        val features = FeatureExtractor.extract(processed, calibration)
        var embedding: List<Double>? = null
        var hand: List<Double>? = null
        var encoderError: String? = null
        if (encoder != null) {
            try {
                val out = encoder.encode(processed)
                embedding = out.embedding
                hand = out.handLogits
            } catch (e: Exception) {
                encoderError = e.message ?: e.javaClass.simpleName
            }
        }
        return AnalyzedPattern(recording, processed, quality, features, embedding, hand, encoderError)
    }
}

/** Small numeric helpers for the Lab (plots and per-channel SNR). */
object SignalSummary {
    /** Euclidean magnitude per grid index of a 3×N matrix. */
    fun magnitude(m: List<DoubleArray>): DoubleArray {
        if (m.size != 3) return DoubleArray(0)
        val n = minOf(m[0].size, m[1].size, m[2].size)
        return DoubleArray(n) { i -> sqrt(m[0][i] * m[0][i] + m[1][i] * m[1][i] + m[2][i] * m[2][i]) }
    }

    /**
     * Mean power in the event window over mean power in the quiet window, per channel
     * (accel HP x, y, z, then gyro HP x, y, z).
     */
    fun channelSNR(p: ProcessedSignal): List<Pair<String, Double>> {
        val names = listOf("accel x", "accel y", "accel z", "gyro x", "gyro y", "gyro z")
        val channels = p.accelHP + p.gyroHP
        return names.zip(channels).map { (name, x) ->
            name to meanPower(x, p.eventRange) / maxOf(meanPower(x, p.quietRange), 1e-12)
        }
    }

    private fun meanPower(x: DoubleArray, r: IntRange): Double {
        val from = maxOf(r.first, 0)
        val to = minOf(r.last, x.size - 1)
        if (to < from) return 0.0
        var sum = 0.0
        for (i in from..to) sum += x[i] * x[i]
        return sum / (to - from + 1)
    }
}
