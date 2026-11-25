package com.mahmoudalyosify.auditor.device

import com.mahmoudalyosify.auditor.core.ChallengeEvent
import com.mahmoudalyosify.auditor.core.ChallengeSpec
import kotlin.math.ceil
import kotlin.math.roundToInt
import kotlin.math.roundToLong

/** Timings (ms) and amplitudes (0–255, or −1 for the default amplitude) of one waveform. */
class WaveformPlan(val timings: LongArray, val amplitudes: IntArray) {
    val totalMillis: Long get() = timings.sum()
}

/** One tap played as a composition primitive. */
data class PrimitiveStep(val tick: Boolean, val scale: Float, val delayMillis: Int)

/**
 * Pure mapping of a challenge to Android vibration effects (the Android counterpart of SPEC 3.5).
 *
 * The whole pattern is laid out on a 1 ms timeline: taps are 12 ms pulses at
 * round(intensity · 255); ramps are 10 ms steps whose amplitude follows the ramp curve at the
 * middle of each step; gaps are amplitude 0. The timeline is run-length encoded into one
 * waveform so the pattern plays from a single vibrate() call, with pattern time 0 at the start.
 * Without amplitude control, taps use the default amplitude and ramp steps are pulse-width
 * modulated (on for round(10 · intensity) ms of each 10 ms step).
 */
object HapticTimeline {
    const val STEP_MS = 10
    const val TAP_MS = 12
    const val DEFAULT_AMPLITUDE = -1

    /** Sharpness above this plays a tap as a tick instead of a click (primitive mode). */
    const val TICK_SHARPNESS = 0.66

    fun amplitude(intensity: Double): Int = (intensity.coerceIn(0.0, 1.0) * 255.0).roundToInt()

    private fun millis(seconds: Double): Int = (seconds * 1000.0).roundToLong().toInt().coerceAtLeast(0)

    /** Amplitude per millisecond of the whole pattern. */
    fun timeline(spec: ChallengeSpec, amplitudeControl: Boolean): IntArray {
        var end = millis(spec.duration)
        for (event in spec.events) {
            end = maxOf(end, when (event) {
                is ChallengeEvent.Tap -> millis(event.event.time) + TAP_MS
                is ChallengeEvent.Ramp -> millis(event.event.time) + ceil(event.event.duration * 1000.0).toInt()
            })
        }
        val amp = IntArray(end)
        fun put(ms: Int, value: Int) {
            if (ms !in amp.indices) return
            amp[ms] = when {
                value == DEFAULT_AMPLITUDE || amp[ms] == DEFAULT_AMPLITUDE -> DEFAULT_AMPLITUDE
                else -> maxOf(amp[ms], value)
            }
        }
        for (event in spec.events) {
            when (event) {
                is ChallengeEvent.Tap -> {
                    val start = millis(event.event.time)
                    val value = if (amplitudeControl) amplitude(event.event.intensity) else DEFAULT_AMPLITUDE
                    if (value != 0) for (ms in start until start + TAP_MS) put(ms, value)
                }
                is ChallengeEvent.Ramp -> {
                    val ramp = event.event
                    val start = millis(ramp.time)
                    val length = ceil(ramp.duration * 1000.0).toInt()
                    var offset = 0
                    while (offset < length) {
                        val stepLength = minOf(STEP_MS, length - offset)
                        val tau = (offset + stepLength / 2.0) / 1000.0
                        val intensity = ramp.intensity(tau).coerceIn(0.0, 1.0)
                        if (amplitudeControl) {
                            val value = amplitude(intensity)
                            if (value != 0) for (ms in 0 until stepLength) put(start + offset + ms, value)
                        } else {
                            val on = (intensity * stepLength).roundToInt()
                            for (ms in 0 until on) put(start + offset + ms, DEFAULT_AMPLITUDE)
                        }
                        offset += stepLength
                    }
                }
            }
        }
        return amp
    }

    /** Run-length encoded waveform; trailing silence is dropped. */
    fun waveform(spec: ChallengeSpec, amplitudeControl: Boolean): WaveformPlan {
        val amp = timeline(spec, amplitudeControl)
        var last = amp.size - 1
        while (last >= 0 && amp[last] == 0) last--
        val timings = ArrayList<Long>()
        val amplitudes = ArrayList<Int>()
        var i = 0
        while (i <= last) {
            var j = i
            while (j <= last && amp[j] == amp[i]) j++
            timings.add((j - i).toLong())
            amplitudes.add(amp[i])
            i = j
        }
        return WaveformPlan(timings.toLongArray(), amplitudes.toIntArray())
    }

    /**
     * Taps as composition primitives (only for specs without ramps: a composition cannot hold
     * waveform segments). `durationMillis(tick)` is the device's primitive duration; delays are
     * measured from the end of the previous primitive, as `Composition.addPrimitive` expects.
     */
    fun primitives(spec: ChallengeSpec, durationMillis: (Boolean) -> Int): List<PrimitiveStep> {
        val taps = spec.events.filterIsInstance<ChallengeEvent.Tap>().map { it.event }.sortedBy { it.time }
        val out = ArrayList<PrimitiveStep>()
        var cursor = 0
        for (tap in taps) {
            val tick = tap.sharpness > TICK_SHARPNESS
            val start = millis(tap.time)
            val delay = maxOf(0, start - cursor)
            out.add(PrimitiveStep(tick, tap.intensity.coerceIn(0.0, 1.0).toFloat(), delay))
            cursor = maxOf(cursor, start) + maxOf(durationMillis(tick), 0)
        }
        return out
    }

    fun hasRamps(spec: ChallengeSpec): Boolean = spec.events.any { it is ChallengeEvent.Ramp }
}
