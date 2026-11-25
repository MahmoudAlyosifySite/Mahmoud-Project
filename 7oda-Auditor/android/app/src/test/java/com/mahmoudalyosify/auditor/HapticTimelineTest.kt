package com.mahmoudalyosify.auditor

import com.mahmoudalyosify.auditor.core.ChallengeEvent
import com.mahmoudalyosify.auditor.core.ChallengeGenerator
import com.mahmoudalyosify.auditor.core.ChallengeSpec
import com.mahmoudalyosify.auditor.core.CurvePoint
import com.mahmoudalyosify.auditor.core.RampEvent
import com.mahmoudalyosify.auditor.core.TapEvent
import com.mahmoudalyosify.auditor.device.HapticTimeline
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class HapticTimelineTest {
    private val spec = ChallengeSpec(
        version = 1, seed = 1234UL, duration = 0.5,
        events = listOf(
            ChallengeEvent.Tap(TapEvent(time = 0.05, intensity = 1.0, sharpness = 0.4)),
            ChallengeEvent.Ramp(RampEvent(time = 0.2, duration = 0.1, sharpness = 0.5,
                curve = listOf(CurvePoint(0.0, 0.2), CurvePoint(0.05, 0.6), CurvePoint(0.1, 1.0)))),
            ChallengeEvent.Tap(TapEvent(time = 0.4, intensity = 0.5, sharpness = 0.9)),
        ),
    )

    @Test
    fun waveformStartsWithLeadSilenceAndPlaysTapsAtTheirTimes() {
        val plan = HapticTimeline.waveform(spec, amplitudeControl = true)
        assertEquals(plan.timings.size, plan.amplitudes.size)
        assertEquals(50L, plan.timings[0])
        assertEquals(0, plan.amplitudes[0])
        assertEquals(HapticTimeline.TAP_MS.toLong(), plan.timings[1])
        assertEquals(255, plan.amplitudes[1])
        // Last segment is the second tap at round(0.5 · 255), ending at 400 + 12 ms.
        assertEquals(128, plan.amplitudes.last())
        assertEquals(412L, plan.totalMillis)
        assertTrue(plan.timings.all { it > 0 })
    }

    @Test
    fun rampStepsFollowTheCurve() {
        val amp = HapticTimeline.timeline(spec, amplitudeControl = true)
        val steps = (0 until 10).map { amp[200 + it * 10 + 5] }
        assertEquals(HapticTimeline.amplitude(spec.ramps[0].intensity(0.005)), steps[0])
        for (i in 1 until steps.size) assertTrue("ramp rises", steps[i] >= steps[i - 1])
        assertEquals(0, amp[150])
    }

    @Test
    fun withoutAmplitudeControlTapsUseTheDefaultAmplitude() {
        val plan = HapticTimeline.waveform(spec, amplitudeControl = false)
        assertTrue(plan.amplitudes.all { it == 0 || it == HapticTimeline.DEFAULT_AMPLITUDE })
        assertEquals(HapticTimeline.DEFAULT_AMPLITUDE, plan.amplitudes[1])
    }

    @Test
    fun primitivesUseTicksForSharpTapsAndRelativeDelays() {
        val steps = HapticTimeline.primitives(spec) { 10 }
        assertEquals(2, steps.size)
        assertEquals(false, steps[0].tick)
        assertEquals(50, steps[0].delayMillis)
        assertEquals(true, steps[1].tick)
        assertEquals(400 - 60, steps[1].delayMillis)
    }

    @Test
    fun generatedChallengesFitTheirDuration() {
        for (seed in 1UL..20UL) {
            val s = ChallengeGenerator.generate(seed)
            val plan = HapticTimeline.waveform(s, amplitudeControl = true)
            assertTrue(plan.totalMillis <= (s.duration * 1000).toLong() + HapticTimeline.TAP_MS + 1)
            assertTrue(plan.amplitudes.all { it in 0..255 })
            assertTrue(plan.amplitudes.any { it > 0 })
        }
    }
}
