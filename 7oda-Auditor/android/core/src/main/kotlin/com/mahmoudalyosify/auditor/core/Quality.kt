package com.mahmoudalyosify.auditor.core

import kotlin.math.abs

// MARK: - Types (SPEC 6)

@Suppress("EnumEntryName")
enum class QualityMode {
    handheld, onTable;

    val rawValue: String get() = name

    companion object {
        fun from(raw: String): QualityMode? = entries.firstOrNull { it.name == raw }
    }
}

@Suppress("EnumEntryName")
enum class QualityIssue {
    sensorGaps, lowSampleRate, resting, moving, weakAlignment, weakSignal, notResting;

    val rawValue: String get() = name

    /** Short user-facing hint. */
    val message: String
        get() = when (this) {
            QualityIssue.resting -> "Hold the phone in one hand"
            QualityIssue.moving -> "Hold still"
            QualityIssue.notResting -> "Put the phone flat on a table and don't touch it"
            QualityIssue.sensorGaps, QualityIssue.weakAlignment -> "Try again"
            QualityIssue.weakSignal -> "Vibration too weak — remove a thick case?"
            QualityIssue.lowSampleRate -> "Motion sensors are too slow on this device"
        }

    companion object {
        fun from(raw: String): QualityIssue? = entries.firstOrNull { it.name == raw }
    }
}

data class QualityMetrics(
    val gapFraction: Double,
    val sampleRate: Double,
    val quietGyroRMS: Double,
    val quietAccelRMS: Double,
    val gravityZ: Double,
    val motionRMS: Double,
    val gravityAngleDeg: Double,
    val ncc: Double,
    val peakMargin: Double,
    val snr: Double,
    val lag: Int,
)

data class QualityConfig(
    val maxGapFraction: Double = 0.10,
    val minSampleRate: Double = 70.0,
    val restingGyroFactor: Double = 2.5,
    val restingGyroAbsolute: Double = 0.004,
    val flatGravityZ: Double = 0.95,
    val maxMotionRMS: Double = 0.30,
    val maxGravityAngleDeg: Double = 10.0,
    val minNCC: Double = 0.30,
    val minPeakMargin: Double = 0.05,
    val minSNR: Double = 2.0,
    val tableMaxMotionRMS: Double = 0.05,
) {
    companion object {
        val default: QualityConfig = QualityConfig()
    }
}

data class QualityReport(
    val metrics: QualityMetrics,
    val issues: List<QualityIssue>,
    val passed: Boolean = issues.isEmpty(),
)

// MARK: - Gate (SPEC 6.1, 6.3)

object QualityGate {
    fun evaluate(
        p: ProcessedSignal,
        calibration: Calibration?,
        mode: QualityMode,
        config: QualityConfig = QualityConfig.default,
    ): QualityReport {
        val m = metrics(p)
        val cfg = config
        val issues = ArrayList<QualityIssue>()
        val weakAlignment = m.ncc < cfg.minNCC || m.peakMargin < cfg.minPeakMargin
        val weakSignal = m.snr < cfg.minSNR
        if (m.gapFraction > cfg.maxGapFraction) issues.add(QualityIssue.sensorGaps)
        if (m.sampleRate < cfg.minSampleRate) issues.add(QualityIssue.lowSampleRate)
        when (mode) {
            QualityMode.handheld -> {
                val restThreshold = if (calibration != null) {
                    cfg.restingGyroFactor * calibration.quietGyroRMS
                } else {
                    cfg.restingGyroAbsolute
                }
                if (m.gravityZ >= cfg.flatGravityZ && m.quietGyroRMS < restThreshold) issues.add(QualityIssue.resting)
                if (m.motionRMS > cfg.maxMotionRMS || m.gravityAngleDeg > cfg.maxGravityAngleDeg) issues.add(QualityIssue.moving)
                if (weakAlignment) issues.add(QualityIssue.weakAlignment)
                if (weakSignal) issues.add(QualityIssue.weakSignal)
            }
            QualityMode.onTable -> {
                if (weakAlignment) issues.add(QualityIssue.weakAlignment)
                if (weakSignal) issues.add(QualityIssue.weakSignal)
                if (m.gravityZ < cfg.flatGravityZ || m.motionRMS > cfg.tableMaxMotionRMS) issues.add(QualityIssue.notResting)
            }
        }
        return QualityReport(m, issues, issues.isEmpty())
    }

    internal fun metrics(p: ProcessedSignal): QualityMetrics {
        val n = p.n
        val q = p.quietRange
        val e = p.eventRange
        val lp = p.config.motionLowPassFilter
        val rl = p.rotation.map { lp.filtfiltZ(it, p.config.pad, 1.0) }
        val motionEnergy = DoubleArray(n)
        if (rl.size == 3) {
            for (i in 0 until n) {
                if (i < rl[0].size && i < rl[1].size && i < rl[2].size) {
                    motionEnergy[i] = rl[0][i] * rl[0][i] + rl[1][i] * rl[1][i] + rl[2][i] * rl[2][i]
                }
            }
        }
        val snr = Numerics.mean(p.envelope, e) / Numerics.smax(Numerics.mean(p.envelope, q), 1e-9)
        val gravityZ = if (n > 0 && p.gravity.size == 3) abs(Numerics.mean(p.gravity[2])) else 0.0
        return QualityMetrics(
            gapFraction = p.gapFraction,
            sampleRate = p.sampleRate,
            quietGyroRMS = Math.sqrt(Numerics.mean(p.energyGyro, q)),
            quietAccelRMS = Math.sqrt(Numerics.mean(p.energyAccel, q)),
            gravityZ = gravityZ,
            motionRMS = Math.sqrt(Numerics.mean(motionEnergy)),
            gravityAngleDeg = gravityAngleDeg(p),
            ncc = p.alignment.ncc,
            peakMargin = p.alignment.peakMargin,
            snr = snr,
            lag = p.alignment.lag,
        )
    }

    internal fun gravityAngleDeg(p: ProcessedSignal): Double {
        val n = p.n
        if (n <= 0 || p.gravity.size != 3 || !p.gravity.all { it.size == n }) return 0.0
        val k = minOf(20, n)
        val g = p.gravity
        val head = 0 until k
        val tail = (n - k) until n
        val a0 = Numerics.mean(g[0], head)
        val a1 = Numerics.mean(g[1], head)
        val a2 = Numerics.mean(g[2], head)
        val b0 = Numerics.mean(g[0], tail)
        val b1 = Numerics.mean(g[1], tail)
        val b2 = Numerics.mean(g[2], tail)
        val na = Math.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
        val nb = Math.sqrt(b0 * b0 + b1 * b1 + b2 * b2)
        if (na < 1e-9 || nb < 1e-9) return 0.0
        val dot = a0 * b0 + a1 * b1 + a2 * b2
        val cosine = Numerics.smin(Numerics.smax(dot / (na * nb), -1.0), 1.0)
        return Math.acos(cosine) * 180.0 / Math.PI
    }
}
