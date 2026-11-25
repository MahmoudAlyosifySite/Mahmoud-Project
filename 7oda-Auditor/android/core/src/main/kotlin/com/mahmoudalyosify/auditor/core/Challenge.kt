package com.mahmoudalyosify.auditor.core

// MARK: - Events (SPEC 3.3, 3.4)

/** One point of a ramp's intensity curve; `time` is relative to the ramp start. */
data class CurvePoint(val time: Double, val intensity: Double)

/** A transient haptic tap. */
data class TapEvent(val time: Double, val intensity: Double, val sharpness: Double)

/** A continuous haptic event whose intensity follows a piecewise-linear [curve]. */
data class RampEvent(val time: Double, val duration: Double, val sharpness: Double, val curve: List<CurvePoint>) {
    /** Intensity at `at` seconds after the ramp start (SPEC 3.3), clamped to the end points. */
    fun intensity(at: Double): Double {
        val tau = at
        if (curve.isEmpty()) return 0.0
        var k = -1
        for ((index, point) in curve.withIndex()) {
            if (point.time <= tau) k = index
        }
        if (k < 0) return curve.first().intensity
        if (k == curve.size - 1) return curve.last().intensity
        val p0 = curve[k]
        val p1 = curve[k + 1]
        val w = (tau - p0.time) / (p1.time - p0.time)
        return p0.intensity + w * (p1.intensity - p0.intensity)
    }
}

/** A challenge event: a tap or a ramp. JSON uses a `kind` key (`"tap"` or `"ramp"`). */
sealed class ChallengeEvent {
    abstract val time: Double

    data class Tap(val event: TapEvent) : ChallengeEvent() {
        override val time: Double get() = event.time
    }

    data class Ramp(val event: RampEvent) : ChallengeEvent() {
        override val time: Double get() = event.time
    }
}

// MARK: - ChallengeSpec (SPEC 3.4)

/** A random haptic pattern. The seed is written to JSON as the hex string of SPEC 2. */
data class ChallengeSpec(
    val version: Int = 1,
    val seed: ULong,
    val duration: Double,
    val events: List<ChallengeEvent>,
) {
    val taps: List<TapEvent>
        get() = events.mapNotNull { (it as? ChallengeEvent.Tap)?.event }

    val ramps: List<RampEvent>
        get() = events.mapNotNull { (it as? ChallengeEvent.Ramp)?.event }
}

// MARK: - Config (SPEC 3.1)

/** Ranges of the challenge generator (SPEC 3.1). The ramp count is fixed at 2. */
data class ChallengeConfig(
    val leadSilence: ClosedFloatingPointRange<Double> = 0.05..0.10,
    val tapCount: IntRange = 6..8,
    val tapIntensity: ClosedFloatingPointRange<Double> = 0.4..1.0,
    val tapSharpness: ClosedFloatingPointRange<Double> = 0.0..1.0,
    val tapGap: ClosedFloatingPointRange<Double> = 0.08..0.15,
    val rampDuration: ClosedFloatingPointRange<Double> = 0.15..0.25,
    val rampSharpness: ClosedFloatingPointRange<Double> = 0.2..0.8,
    val rampLow: ClosedFloatingPointRange<Double> = 0.25..0.5,
    val rampMid: ClosedFloatingPointRange<Double> = 0.25..1.0,
    val rampHigh: ClosedFloatingPointRange<Double> = 0.75..1.0,
    val rampGap: ClosedFloatingPointRange<Double> = 0.08..0.12,
) {
    val rampCount: Int get() = 2

    companion object {
        val default: ChallengeConfig = ChallengeConfig()
    }
}

// MARK: - Generator (SPEC 3.2)

object ChallengeGenerator {
    private class RampDraw(
        val duration: Double,
        val sharpness: Double,
        val ascending: Boolean,
        val low: Double,
        val mid: Double,
        val high: Double,
        val gap: Double,
    )

    /** Generates the pattern of `seed`; the draw order is normative (SPEC 3.2). */
    fun generate(seed: ULong, config: ChallengeConfig = ChallengeConfig.default): ChallengeSpec {
        val rng = Xoshiro256StarStar(seed)
        val lead = draw(rng, config.leadSilence)
        // At least 2 taps are needed to place the two ramps in distinct slots.
        val n = maxOf(2, rng.int(config.tapCount.first, config.tapCount.last))
        val s1 = 1 + minOf(floorIndex(rng.uniform() * n.toDouble()), n - 1)
        var s2 = 1 + minOf(floorIndex(rng.uniform() * (n - 1).toDouble()), n - 2)
        if (s2 >= s1) s2 += 1
        val slots = listOf(s1, s2).sorted()

        val tapIntensities = ArrayList<Double>(n)
        val tapSharpnesses = ArrayList<Double>(n)
        val tapGaps = ArrayList<Double>(n)
        for (i in 0 until n) {
            val intensity = draw(rng, config.tapIntensity)
            val sharpness = draw(rng, config.tapSharpness)
            val gap = draw(rng, config.tapGap)
            tapIntensities.add(intensity)
            tapSharpnesses.add(sharpness)
            tapGaps.add(gap)
        }
        val ramps = ArrayList<RampDraw>(2)
        for (r in 0 until 2) {
            val duration = draw(rng, config.rampDuration)
            val sharpness = draw(rng, config.rampSharpness)
            val ascending = rng.uniform() < 0.5
            val low = draw(rng, config.rampLow)
            val mid = draw(rng, config.rampMid)
            val high = draw(rng, config.rampHigh)
            val gap = draw(rng, config.rampGap)
            ramps.add(RampDraw(duration, sharpness, ascending, low, mid, high, gap))
        }

        var t = lead
        val events = ArrayList<ChallengeEvent>(n + 2)
        for (i in 0 until n) {
            events.add(ChallengeEvent.Tap(TapEvent(t, tapIntensities[i], tapSharpnesses[i])))
            t += tapGaps[i]
            for (j in 0 until 2) {
                if (slots[j] != i + 1) continue
                val ramp = ramps[j]
                val d = ramp.duration
                val values = if (ramp.ascending) listOf(ramp.low, ramp.mid, ramp.high) else listOf(ramp.high, ramp.mid, ramp.low)
                val curve = listOf(
                    CurvePoint(0.0, values[0]),
                    CurvePoint(d / 2.0, values[1]),
                    CurvePoint(d, values[2]),
                )
                events.add(ChallengeEvent.Ramp(RampEvent(t, d, ramp.sharpness, curve)))
                t += d + ramp.gap
            }
        }
        return ChallengeSpec(version = 1, seed = seed, duration = t, events = events)
    }

    private fun draw(rng: Xoshiro256StarStar, range: ClosedFloatingPointRange<Double>): Double =
        rng.uniform(range.start, range.endInclusive)

    private fun floorIndex(x: Double): Int {
        val f = Math.floor(x)
        if (!f.isFinite() || f <= -2.0e9 || f >= 2.0e9) return 0
        return f.toInt()
    }
}
