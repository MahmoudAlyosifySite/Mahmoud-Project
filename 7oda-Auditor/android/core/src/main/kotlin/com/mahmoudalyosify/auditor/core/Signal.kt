package com.mahmoudalyosify.auditor.core

import kotlin.math.abs

// MARK: - Config (SPEC 5.1)

data class SignalConfig(
    val fs: Double = 100.0,
    val preRoll: Double = 0.25,
    val postRoll: Double = 0.25,
    val hpCutoff: Double = 5.0,
    val hpQ: Double = Biquad.butterworthQ,
    val motionLowPass: Double = 3.0,
    val motionQ: Double = Biquad.butterworthQ,
    val tremorCenter: Double = Math.sqrt(8.0 * 12.0),
    val tremorQ: Double = Math.sqrt(8.0 * 12.0) / 4.0,
    val pad: Int = 30,
    val gapTolerance: Double = 0.025,
    val lagMin: Int = -10,
    val lagMax: Int = 25,
    val templateDecay: Double = 0.03,
    val accelNoiseFloor: Double = 1e-6,
    val gyroNoiseFloor: Double = 1e-6,
) {
    internal val highPass: Biquad get() = Biquad.highPass(hpCutoff, fs, hpQ)
    internal val motionLowPassFilter: Biquad get() = Biquad.lowPass(motionLowPass, fs, motionQ)
    internal val tremorBandPass: Biquad get() = Biquad.bandPass(tremorCenter, fs, tremorQ)

    companion object {
        val default: SignalConfig = SignalConfig()
    }
}

// MARK: - Results

data class AlignmentResult(val lag: Int, val ncc: Double, val peakMargin: Double)

/** Output of [SignalProcessor.process] (SPEC 5). Three-axis signals are lists of 3 arrays of length [n]. */
class ProcessedSignal internal constructor(
    val config: SignalConfig,
    val challenge: ChallengeSpec,
    val n: Int,
    val gridStart: Double,
    val accel: List<DoubleArray>,
    val gyro: List<DoubleArray>,
    val gravity: List<DoubleArray>,
    val rotation: List<DoubleArray>,
    val accelHP: List<DoubleArray>,
    val gyroHP: List<DoubleArray>,
    val energyAccel: DoubleArray,
    val energyGyro: DoubleArray,
    val noiseAccel: Double,
    val noiseGyro: Double,
    val envelope: DoubleArray,
    val template: DoubleArray,
    val alignment: AlignmentResult,
    val gapFraction: Double,
    val sampleRate: Double,
    internal val grid: DoubleArray,
    internal val accelMissing: BooleanArray,
    internal val gyroMissing: BooleanArray,
    internal val amplitude: DoubleArray,
    internal val nccByLag: DoubleArray,
) {
    /** `idx(t)` of SPEC 5.7 (unclamped): grid index of pattern time `t` after alignment. */
    fun index(ofPatternTime: Double): Int =
        Numerics.truncatingInt((config.preRoll + ofPatternTime) * config.fs + 0.5) + alignment.lag

    /** Quiet range Q of SPEC 5.7 (half-open; `end = last + 1`). */
    val quietRange: IntRange
        get() {
            val end = clamp(onsetFirst - 2)
            if (end < 10) {
                val start = maxOf(0, minOf(endIndex + 5, n - 10))
                return start until maxOf(start, n)
            }
            return 0 until end
        }

    /** `[onsetFirst, endIdx)` clamped to the grid (half-open; `end = last + 1`). */
    val eventRange: IntRange
        get() {
            val a = clamp(onsetFirst)
            val b = clamp(endIndex)
            return a until maxOf(a, b)
        }

    internal val endIndex: Int get() = index(challenge.duration)

    internal val onsetFirst: Int get() = index(challenge.events.firstOrNull()?.time ?: 0.0)

    internal fun clamp(i: Int): Int = minOf(maxOf(i, 0), n)
}

// MARK: - Processing (SPEC 5.2-5.7)

object SignalProcessor {
    internal const val maximumGridLength = 100_000

    fun process(recording: Recording, config: SignalConfig = SignalConfig.default): ProcessedSignal {
        val cfg = config
        val challenge = recording.challenge
        val fs = cfg.fs

        // 5.2 Grid.
        val t0 = recording.hapticStart - cfg.preRoll
        val rawN = Numerics.truncatingInt((cfg.preRoll + challenge.duration + cfg.postRoll) * fs + 0.5)
        val n = minOf(maxOf(rawN, 0), maximumGridLength)
        val grid = DoubleArray(n) { k -> t0 + k.toDouble() / fs }

        // 5.2 Resampling.
        val acc = StreamCleaner.sortDedup(recording.accel)
        val gyr = StreamCleaner.sortDedup(recording.gyro)
        val mot = StreamCleaner.sortDedup(recording.motion)
        val tol = cfg.gapTolerance

        fun resample3(t: List<Double>, channels: List<List<Double>>): Pair<List<DoubleArray>, BooleanArray> {
            val tt = t.toDoubleArray()
            val values = ArrayList<DoubleArray>(channels.size)
            var missing = BooleanArray(n) { true }
            for ((i, channel) in channels.withIndex()) {
                val r = Resampler.resample(tt, channel.toDoubleArray(), grid, tol)
                values.add(r.values)
                if (i == 0) missing = r.missing
            }
            return Pair(values, missing)
        }

        val accelR = resample3(acc.t, listOf(acc.x, acc.y, acc.z))
        val gyroR = resample3(gyr.t, listOf(gyr.x, gyr.y, gyr.z))
        val gravityR = resample3(mot.t, listOf(mot.gravityX, mot.gravityY, mot.gravityZ))
        val rotationR = resample3(mot.t, listOf(mot.rotationX, mot.rotationY, mot.rotationZ))

        var missingCount = 0
        for (k in 0 until n) {
            if (accelR.second[k] || gyroR.second[k]) missingCount += 1
        }
        val gapFraction = if (n > 0) missingCount.toDouble() / n.toDouble() else 0.0

        var sampleRate = 0.0
        if (acc.t.size >= 2) {
            val diffs = DoubleArray(acc.t.size - 1) { i -> acc.t[i + 1] - acc.t[i] }
            sampleRate = 1.0 / Numerics.median(diffs)
        }

        // 5.5 Derived signals.
        val hp = cfg.highPass
        val accelHP = accelR.first.map { hp.filtfiltZ(it, cfg.pad, 0.0) }
        val gyroHP = gyroR.first.map { hp.filtfiltZ(it, cfg.pad, 0.0) }
        val ea = DoubleArray(n)
        val eg = DoubleArray(n)
        for (i in 0 until n) {
            ea[i] = accelHP[0][i] * accelHP[0][i] + accelHP[1][i] * accelHP[1][i] + accelHP[2][i] * accelHP[2][i]
            eg[i] = gyroHP[0][i] * gyroHP[0][i] + gyroHP[1][i] * gyroHP[1][i] + gyroHP[2][i] * gyroHP[2][i]
        }
        val preEnd = Numerics.truncatingInt(cfg.preRoll * fs + 0.5) - 5
        val noiseA: Double
        val noiseG: Double
        if (preEnd < 1) {
            noiseA = cfg.accelNoiseFloor
            noiseG = cfg.gyroNoiseFloor
        } else {
            val range = 0 until minOf(preEnd, n)
            noiseA = Numerics.smax(Numerics.mean(ea, range), cfg.accelNoiseFloor)
            noiseG = Numerics.smax(Numerics.mean(eg, range), cfg.gyroNoiseFloor)
        }
        val env = DoubleArray(n)
        val amp = DoubleArray(n)
        for (i in 0 until n) {
            env[i] = ea[i] / noiseA + eg[i] / noiseG
            amp[i] = Math.sqrt(env[i])
        }

        // 5.6 Template and alignment.
        val tmpl = template(challenge, n, cfg)
        val aligned = align(amp, tmpl, cfg.lagMin, cfg.lagMax)

        return ProcessedSignal(
            config = cfg,
            challenge = challenge,
            n = n,
            gridStart = t0,
            accel = accelR.first,
            gyro = gyroR.first,
            gravity = gravityR.first,
            rotation = rotationR.first,
            accelHP = accelHP,
            gyroHP = gyroHP,
            energyAccel = ea,
            energyGyro = eg,
            noiseAccel = noiseA,
            noiseGyro = noiseG,
            envelope = env,
            template = tmpl,
            alignment = aligned.first,
            gapFraction = gapFraction,
            sampleRate = sampleRate,
            grid = grid,
            accelMissing = accelR.second,
            gyroMissing = gyroR.second,
            amplitude = amp,
            nccByLag = aligned.second,
        )
    }

    /** Lag-0 template `h` of SPEC 5.6. */
    internal fun template(challenge: ChallengeSpec, n: Int, cfg: SignalConfig): DoubleArray {
        val taps = challenge.taps
        val ramps = challenge.ramps
        val decay = cfg.templateDecay
        val out = DoubleArray(maxOf(n, 0))
        for (k in out.indices) {
            val tau = k.toDouble() / cfg.fs - cfg.preRoll
            var tapSum = 0.0
            for (tap in taps) {
                if (tau >= tap.time) {
                    tapSum += tap.intensity * Math.exp(-(tau - tap.time) / decay)
                }
            }
            var rampSum = 0.0
            for (ramp in ramps) {
                val t = ramp.time
                val d = ramp.duration
                if (t <= tau && tau < t + d) {
                    rampSum += ramp.intensity(tau - t)
                } else if (tau >= t + d) {
                    rampSum += ramp.intensity(d) * Math.exp(-(tau - t - d) / decay)
                }
            }
            out[k] = tapSum + rampSum
        }
        return out
    }

    internal fun pearson(x: DoubleArray, xStart: Int, h: DoubleArray, hStart: Int, count: Int): Double {
        if (count <= 0) return 0.0
        var sx = 0.0
        var sh = 0.0
        for (i in 0 until count) {
            sx += x[xStart + i]
            sh += h[hStart + i]
        }
        val mx = sx / count.toDouble()
        val mh = sh / count.toDouble()
        var sxx = 0.0
        var shh = 0.0
        var sxh = 0.0
        for (i in 0 until count) {
            val dx = x[xStart + i] - mx
            val dh = h[hStart + i] - mh
            sxx += dx * dx
            shh += dh * dh
            sxh += dx * dh
        }
        if (sxx == 0.0 || shh == 0.0) return 0.0
        return sxh / Math.sqrt(sxx * shh)
    }

    /** Lag search of SPEC 5.6: best lag (ties to the smallest |lag|, then the most negative) and peak margin. */
    internal fun align(x: DoubleArray, h: DoubleArray, lagMin: Int, lagMax: Int): Pair<AlignmentResult, DoubleArray> {
        if (lagMin > lagMax) return Pair(AlignmentResult(0, 0.0, 0.0), DoubleArray(0))
        val n = minOf(x.size, h.size)
        val lags = (lagMin..lagMax).toList()
        val ncc = DoubleArray(lags.size)
        for ((i, lag) in lags.withIndex()) {
            val lo = maxOf(0, lag)
            val hi = minOf(n, n + lag)
            if (hi - lo < 10) continue
            ncc[i] = pearson(x, lo, h, lo - lag, hi - lo)
        }
        val order = lags.indices.sortedWith(Comparator { a, b ->
            val la = abs(lags[a])
            val lb = abs(lags[b])
            if (la != lb) la.compareTo(lb) else lags[a].compareTo(lags[b])
        })
        var best = order[0]
        for (i in order.drop(1)) {
            if (ncc[i] > ncc[best]) best = i
        }
        val bestLag = lags[best]
        val bestValue = ncc[best]
        var far: Double? = null
        for ((i, lag) in lags.withIndex()) {
            if (abs(lag - bestLag) < 4) continue
            val current = far
            if (current == null || ncc[i] > current) {
                far = ncc[i]
            }
        }
        val farValue = far
        val peakMargin = if (farValue != null) bestValue - farValue else bestValue
        return Pair(AlignmentResult(bestLag, bestValue, peakMargin), ncc)
    }
}
