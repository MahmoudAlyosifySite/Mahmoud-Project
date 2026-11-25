package com.mahmoudalyosify.auditor.core

import kotlin.math.ln

/** Hand-made features v2 (SPEC 7, d = 39). */
object FeatureExtractor {
    const val version: Int = 2

    val names: List<String> = run {
        val a = listOf("ax", "ay", "az")
        val g = listOf("gx", "gy", "gz")
        val x = listOf("axy", "axz", "ayz")
        val n = ArrayList<String>()
        n += listOf("tap_gain_accel", "tap_log_accel_gyro")
        n += a.map { "tap_burst_share_$it" }
        n += x.map { "tap_burst_cross_$it" }
        n += g.map { "tap_burst_share_$it" }
        n += a.map { "tap_tail_share_$it" }
        n += x.map { "tap_tail_cross_$it" }
        n += g.map { "tap_tail_share_$it" }
        n += listOf("tap_tail_log_accel_gyro", "ramp_gain_accel", "ramp_log_accel_gyro")
        n += a.map { "ramp_share_$it" }
        n += x.map { "ramp_cross_$it" }
        n += g.map { "ramp_share_$it" }
        n += listOf(
            "quiet_tremor_accel", "quiet_tremor_gyro", "quiet_rms_accel", "quiet_rms_gyro",
            "gravity_x", "gravity_y", "gravity_z",
        )
        n.toList()
    }

    /** Indices R that are made relative to the calibration table (SPEC 7.5). */
    val calibrationRelativeIndices: List<Int> = listOf(0, 1, 20, 21, 22)

    internal const val epsilon: Double = 1e-12

    fun extractAbsolute(p: ProcessedSignal): List<Double> = tapFeatures(p) + rampFeatures(p) + quietFeatures(p)

    fun extract(p: ProcessedSignal, calibration: Calibration?): List<Double> {
        val f = extractAbsolute(p).toMutableList()
        val table = calibration?.tableFeatures
        if (table != null) {
            for (r in calibrationRelativeIndices) {
                if (r < f.size && r < table.size) f[r] = f[r] - table[r]
            }
        }
        return f.toList()
    }

    // MARK: - Helpers

    internal fun lnRatio(a: Double, b: Double): Double = ln((a + epsilon) / (b + epsilon))

    internal class Shape(private val withCross: Boolean) {
        val energy = doubleArrayOf(0.0, 0.0, 0.0)
        val crossSum = doubleArrayOf(0.0, 0.0, 0.0)

        fun add(signal: List<DoubleArray>, a: Int, b: Int) {
            if (signal.size != 3) return
            val lo = maxOf(a, 0)
            val hi = minOf(minOf(b, signal[0].size), minOf(signal[1].size, signal[2].size))
            if (hi <= lo) return
            for (c in 0 until 3) {
                var s = 0.0
                for (i in lo until hi) {
                    val v = signal[c][i]
                    s += v * v
                }
                energy[c] = energy[c] + s
            }
            if (withCross) {
                for ((k, pair) in PAIRS.withIndex()) {
                    var s = 0.0
                    for (i in lo until hi) {
                        s += signal[pair.first][i] * signal[pair.second][i]
                    }
                    crossSum[k] = crossSum[k] + s
                }
            }
        }

        val total: Double get() = (energy[0] + energy[1]) + energy[2]

        fun shares(): List<Double> {
            val t = total
            return energy.map { it / (t + FeatureExtractor.epsilon) }
        }

        fun cross(): List<Double> {
            val t = total
            return crossSum.map { it / (t + FeatureExtractor.epsilon) }
        }

        private companion object {
            val PAIRS = listOf(Pair(0, 1), Pair(0, 2), Pair(1, 2))
        }
    }

    internal fun windowSum(values: DoubleArray, a: Int, b: Int): Double {
        val lo = maxOf(a, 0)
        val hi = minOf(b, values.size)
        if (hi <= lo) return 0.0
        return Numerics.sum(values, lo until hi)
    }

    // MARK: - Taps (SPEC 7.1, 7.2): indices 0-20

    internal fun tapFeatures(p: ProcessedSignal): List<Double> {
        val events = p.challenge.events
        val n = p.n
        val endNext = p.endIndex + 25
        var eaFull = 0.0
        var egFull = 0.0
        var i2 = 0.0
        val burstA = Shape(true)
        val burstG = Shape(false)
        val tailA = Shape(true)
        val tailG = Shape(false)
        for ((ei, event) in events.withIndex()) {
            if (event !is ChallengeEvent.Tap) continue
            val tap = event.event
            val o = p.index(tap.time)
            val next = if (ei + 1 < events.size) p.index(events[ei + 1].time) else endNext
            val u = minOf(next - 1, n)
            val fullA = maxOf(o - 1, 0)
            val fullB = minOf(o + 9, u)
            if (fullB <= fullA) continue // a tap whose full window is empty is skipped entirely
            eaFull += windowSum(p.energyAccel, fullA, fullB)
            egFull += windowSum(p.energyGyro, fullA, fullB)
            i2 += tap.intensity * tap.intensity
            val burstLo = maxOf(o - 1, 0)
            val burstHi = minOf(o + 2, u)
            val tailLo = maxOf(o + 2, 0)
            val tailHi = minOf(o + 9, u)
            burstA.add(p.accelHP, burstLo, burstHi)
            burstG.add(p.gyroHP, burstLo, burstHi)
            tailA.add(p.accelHP, tailLo, tailHi)
            tailG.add(p.gyroHP, tailLo, tailHi)
        }
        val f = ArrayList<Double>(21)
        f += lnRatio(eaFull, i2)
        f += lnRatio(eaFull, egFull)
        f += burstA.shares()
        f += burstA.cross()
        f += burstG.shares()
        f += tailA.shares()
        f += tailA.cross()
        f += tailG.shares()
        f += lnRatio(tailA.total, tailG.total)
        return f
    }

    // MARK: - Ramps (SPEC 7.3): indices 21-31

    internal fun rampFeatures(p: ProcessedSignal): List<Double> {
        val cfg = p.config
        val lag = p.alignment.lag
        val shapeA = Shape(true)
        val shapeG = Shape(false)
        var i2 = 0.0
        for (event in p.challenge.events) {
            if (event !is ChallengeEvent.Ramp) continue
            val ramp = event.event
            val s = p.clamp(p.index(ramp.time))
            val e = p.clamp(p.index(ramp.time + ramp.duration))
            val a = s + 2
            val b = e
            if (b <= a) continue
            val tj = ramp.time
            val dj = ramp.duration
            for (i in a until b) {
                var tau = (i - lag).toDouble() / cfg.fs - cfg.preRoll - tj
                tau = Numerics.smin(Numerics.smax(tau, 0.0), dj)
                val c = ramp.intensity(tau)
                i2 += c * c
            }
            shapeA.add(p.accelHP, a, b)
            shapeG.add(p.gyroHP, a, b)
        }
        val f = ArrayList<Double>(11)
        f += lnRatio(shapeA.total, i2)
        f += lnRatio(shapeA.total, shapeG.total)
        f += shapeA.shares()
        f += shapeA.cross()
        f += shapeG.shares()
        return f
    }

    // MARK: - Quiet and grip (SPEC 7.4): indices 32-38

    internal fun quietFeatures(p: ProcessedSignal): List<Double> {
        val cfg = p.config
        val q = p.quietRange
        val bp = cfg.tremorBandPass
        val ba = p.accel.map { bp.filtfiltZ(it, cfg.pad, 0.0) }
        val bg = p.gyro.map { bp.filtfiltZ(it, cfg.pad, 0.0) }
        val tremorA = sumOfSquares(ba, p.n)
        val tremorG = sumOfSquares(bg, p.n)
        val f = ArrayList<Double>(7)
        f += ln(Numerics.mean(tremorA, q) + 1e-10)
        f += ln(Numerics.mean(tremorG, q) + 1e-10)
        f += ln(Math.sqrt(Numerics.mean(p.energyAccel, q)) + 1e-6)
        f += ln(Math.sqrt(Numerics.mean(p.energyGyro, q)) + 1e-6)
        val gx = if (p.gravity.isNotEmpty()) Numerics.mean(p.gravity[0]) else 0.0
        val gy = if (p.gravity.size > 1) Numerics.mean(p.gravity[1]) else 0.0
        val gz = if (p.gravity.size > 2) Numerics.mean(p.gravity[2]) else 0.0
        val norm = Math.sqrt(gx * gx + gy * gy + gz * gz)
        if (norm < 1e-6) {
            f += listOf(0.0, 0.0, -1.0)
        } else {
            f += listOf(gx / norm, gy / norm, gz / norm)
        }
        return f
    }

    internal fun sumOfSquares(s: List<DoubleArray>, count: Int): DoubleArray {
        val out = DoubleArray(maxOf(count, 0))
        if (s.size != 3 || !s.all { it.size >= count }) return out
        for (i in out.indices) {
            out[i] = s[0][i] * s[0][i] + s[1][i] * s[1][i] + s[2][i] * s[2][i]
        }
        return out
    }
}
