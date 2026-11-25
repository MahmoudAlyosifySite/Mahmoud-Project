package com.mahmoudalyosify.auditor.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class ChallengeTest {
    @Test
    fun challengeFixtureParity() = checks { c ->
        val cases = Fixtures.json("challenge_vectors").at("cases").arr()
        c.check(cases.size >= 50) { "expected at least 50 challenge cases, found ${cases.size}" }
        for (case in cases) {
            val seedText = case.at("seed").str()
            val seed = SeedFormat.parse(seedText)
            if (seed == null) {
                c.fail("invalid seed $seedText")
                continue
            }
            val expected = RecordingCoding.challengeSpecFromJson(case.at("spec"))
            val generated = ChallengeGenerator.generate(seed)
            c.equal(generated.seed, expected.seed, "seed $seedText")
            c.equal(generated.version, expected.version, "version for $seedText")
            c.equal(generated.duration, expected.duration, "duration for $seedText")
            c.equal(generated.events.size, expected.events.size, "event count for $seedText")
            for ((i, pair) in generated.events.zip(expected.events).withIndex()) {
                c.equal(pair.first, pair.second, "event $i for $seedText")
            }
            c.equal(generated, expected, "spec for $seedText")
        }
    }

    @Test
    fun structureInvariants() {
        for (s in 0uL until 300uL) {
            val spec = ChallengeGenerator.generate(s * 0x9E3779B97F4A7C15uL)
            val taps = spec.taps
            val ramps = spec.ramps
            assertTrue(taps.size in 6..8)
            assertEquals(2, ramps.size)
            assertEquals(taps.size + 2, spec.events.size)
            assertTrue(spec.events[0] is ChallengeEvent.Tap)
            var lastWasRamp = false
            for (event in spec.events) {
                if (event is ChallengeEvent.Ramp) {
                    assertFalse("two consecutive ramps for seed $s", lastWasRamp)
                    lastWasRamp = true
                } else {
                    lastWasRamp = false
                }
            }
            for (i in 1 until spec.events.size) assertTrue(spec.events[i].time > spec.events[i - 1].time)
            assertTrue(spec.duration > spec.events.last().time)
            assertTrue("duration ${spec.duration}", spec.duration > 0.8 && spec.duration < 2.2)
            for (ramp in ramps) {
                assertEquals(3, ramp.curve.size)
                assertEquals(0.0, ramp.curve[0].time, 0.0)
                assertEquals(ramp.duration / 2.0, ramp.curve[1].time, 0.0)
                assertEquals(ramp.duration, ramp.curve[2].time, 0.0)
            }
        }
    }

    @Test
    fun curveInterpolationAndClamping() {
        val ramp = RampEvent(
            0.5, 0.2, 0.5,
            listOf(CurvePoint(0.0, 0.3), CurvePoint(0.1, 0.7), CurvePoint(0.2, 0.9)),
        )
        assertEquals(0.3, ramp.intensity(-1.0), 0.0)
        assertEquals(0.3, ramp.intensity(0.0), 0.0)
        assertEquals(0.3 + 0.5 * (0.7 - 0.3), ramp.intensity(0.05), 1e-15)
        assertEquals(0.7, ramp.intensity(0.1), 0.0)
        assertEquals(0.9, ramp.intensity(0.2), 0.0)
        assertEquals(0.9, ramp.intensity(5.0), 0.0)
        assertEquals(0.0, RampEvent(0.0, 0.1, 0.5, emptyList()).intensity(0.05), 0.0)
    }

    @Test
    fun jsonShapeAndRoundTrip() {
        val spec = ChallengeGenerator.generate(1234uL)
        val text = RecordingCoding.encode(spec)
        val tree = kotlinx.serialization.json.Json.parseToJsonElement(text)
        assertEquals("0x00000000000004d2", tree.at("seed").str())
        for (event in tree.at("events").arr()) {
            val kind = event.at("kind").str()
            if (kind == "tap") {
                assertFalse(event.has("duration"))
                assertFalse(event.has("curve"))
                assertTrue(event.has("intensity"))
            } else {
                assertEquals("ramp", kind)
                assertFalse(event.has("intensity"))
                assertTrue(event.has("curve"))
                assertTrue(event.has("duration"))
            }
        }
        assertEquals(spec, RecordingCoding.decodeChallengeSpec(text))
    }

    @Test
    fun decodingIgnoresUnknownKeysAndRejectsUnknownKinds() {
        val json = """
            {"version": 1, "seed": "0x00000000000004d2", "duration": 1.25, "extra": true,
             "events": [
               {"kind": "tap", "time": 0.071, "intensity": 0.83, "sharpness": 0.41, "note": "x"},
               {"kind": "ramp", "time": 0.52, "duration": 0.19, "sharpness": 0.47,
                "curve": [{"time": 0.0, "intensity": 0.31}, {"time": 0.095, "intensity": 0.66}, {"time": 0.19, "intensity": 0.93}]}
             ]}
        """.trimIndent()
        val spec = RecordingCoding.decodeChallengeSpec(json)
        assertEquals(1234uL, spec.seed)
        assertEquals(2, spec.events.size)
        assertEquals(ChallengeEvent.Tap(TapEvent(0.071, 0.83, 0.41)), spec.events[0])
        assertEquals(0.52, spec.events[1].time, 0.0)
        assertEquals(3, spec.ramps.first().curve.size)
        val bad = """{"version": 1, "seed": "0x01", "duration": 1.0, "events": [{"kind": "buzz", "time": 0.1}]}"""
        val badSeed = """{"version": 1, "seed": "1234", "duration": 1.0, "events": []}"""
        for (text in listOf(bad, badSeed)) {
            try {
                RecordingCoding.decodeChallengeSpec(text)
                fail("decoding should fail: $text")
            } catch (e: IllegalArgumentException) {
                // expected
            }
        }
    }

    @Test
    fun configDefaults() {
        val c = ChallengeConfig.default
        assertEquals(6..8, c.tapCount)
        assertEquals(2, c.rampCount)
        assertEquals(0.08..0.15, c.tapGap)
        assertEquals(0.08..0.12, c.rampGap)
    }
}
