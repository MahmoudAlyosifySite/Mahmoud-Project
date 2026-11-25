package com.mahmoudalyosify.auditor.core

import kotlinx.serialization.json.JsonElement
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class DSPTest {
    private val tolerance = 1e-9

    private fun design(kind: String, fc: Double, fs: Double, q: Double): Biquad? = when (kind) {
        "highpass" -> Biquad.highPass(fc, fs, q)
        "lowpass" -> Biquad.lowPass(fc, fs, q)
        "bandpass" -> Biquad.bandPass(fc, fs, q)
        else -> null
    }

    private fun streamValues(e: JsonElement, key: String): List<Double> = e.opt(key)?.dbls() ?: emptyList()

    @Test
    fun dspFixtureParity() = checks { c ->
        val root = Fixtures.json("dsp_vectors")
        val biquads = root.at("biquads").arr()
        val filters = root.at("filters").arr()
        val resample = root.at("resample").arr()
        c.check(biquads.isNotEmpty() && filters.isNotEmpty() && resample.isNotEmpty()) { "empty DSP fixture blocks" }
        for ((i, b) in biquads.withIndex()) {
            val kind = b.at("kind").str()
            val q = design(kind, b.at("fc").dbl(), b.at("fs").dbl(), b.at("q").dbl())
            if (q == null) {
                c.fail("unknown biquad kind $kind")
                continue
            }
            c.close(listOf(q.b0, q.b1, q.b2, q.a1, q.a2), b.at("coeffs").dbls(), tolerance, "biquads[$i] $kind")
        }
        for ((i, f) in filters.withIndex()) {
            val kind = f.at("kind").str()
            val q = design(kind, f.at("fc").dbl(), f.at("fs").dbl(), f.at("q").dbl())
            if (q == null) {
                c.fail("unknown biquad kind $kind")
                continue
            }
            val x = f.at("x").dbls().toDoubleArray()
            val label = "filters[$i] ${f.opt("name")?.str() ?: kind} n=${x.size}"
            c.close(q.filter(x), f.at("lfilter").dbls(), tolerance, "$label lfilter")
            c.close(q.filtfiltZ(x, f.at("pad").int(), f.at("dcGain").dbl()), f.at("filtfilt").dbls(), tolerance, "$label filtfiltZ")
        }
        for ((i, r) in resample.withIndex()) {
            val out = Resampler.resample(
                r.at("t").dbls().toDoubleArray(),
                r.at("v").dbls().toDoubleArray(),
                r.at("grid").dbls().toDoubleArray(),
                r.at("gapTolerance").dbl(),
            )
            val label = "resample[$i] ${r.opt("name")?.str() ?: ""}"
            c.close(out.values, r.at("values").dbls(), tolerance, "$label values")
            c.equal(out.missing.toList(), r.at("missing").arr().map { it.bool() }, "$label missing")
        }
        for ((i, s) in (root.opt("sortDedup")?.arr() ?: emptyList()).withIndex()) {
            val label = "sortDedup[$i] ${s.opt("name")?.str() ?: ""}"
            val input = s.at("input")
            val output = s.at("output")
            if (input.has("gravityX")) {
                val m = StreamCleaner.sortDedup(motion(input))
                val expected = motion(output)
                c.equal(m.t, expected.t, "$label t")
                c.equal(m.gravityZ, expected.gravityZ, "$label gravityZ")
                c.equal(m.rotationX, expected.rotationX, "$label rotationX")
            } else {
                val got = StreamCleaner.sortDedup(
                    SensorStream(streamValues(input, "t"), streamValues(input, "x"), streamValues(input, "y"), streamValues(input, "z")),
                )
                val expected = SensorStream(
                    streamValues(output, "t"), streamValues(output, "x"), streamValues(output, "y"), streamValues(output, "z"),
                )
                c.equal(got, expected, label)
            }
        }
    }

    private fun motion(e: JsonElement): MotionStream = MotionStream(
        t = streamValues(e, "t"),
        gravityX = streamValues(e, "gravityX"), gravityY = streamValues(e, "gravityY"), gravityZ = streamValues(e, "gravityZ"),
        userAccelX = streamValues(e, "userAccelX"), userAccelY = streamValues(e, "userAccelY"),
        userAccelZ = streamValues(e, "userAccelZ"),
        rotationX = streamValues(e, "rotationX"), rotationY = streamValues(e, "rotationY"), rotationZ = streamValues(e, "rotationZ"),
        attitudeW = streamValues(e, "attitudeW"), attitudeX = streamValues(e, "attitudeX"),
        attitudeY = streamValues(e, "attitudeY"), attitudeZ = streamValues(e, "attitudeZ"),
    )

    @Test
    fun filtfiltEdgeCases() {
        val lp = Biquad.lowPass(3.0, 100.0, Biquad.butterworthQ)
        val hp = Biquad.highPass(5.0, 100.0, Biquad.butterworthQ)
        assertEquals(0, lp.filtfiltZ(DoubleArray(0), 30, 1.0).size)
        assertArrayEquals(doubleArrayOf(2.5), lp.filtfiltZ(doubleArrayOf(2.5), 30, 1.0), 0.0)
        assertArrayEquals(doubleArrayOf(0.0), hp.filtfiltZ(doubleArrayOf(2.5), 30, 0.0), 0.0)
        assertEquals(2, hp.filtfiltZ(doubleArrayOf(1.0, 2.0), 30, 0.0).size)
        val constant = DoubleArray(50) { -0.98 }
        for (v in lp.filtfiltZ(constant, 30, 1.0)) assertEquals(-0.98, v, 1e-12)
        for (v in hp.filtfiltZ(constant, 30, 0.0)) assertEquals(0.0, v, 1e-12)
        assertEquals(0, lp.filter(DoubleArray(0)).size)
        assertEquals(0.7071067811865475, Biquad.butterworthQ, 0.0)
    }

    @Test
    fun resampleEdgeCases() {
        val empty = Resampler.resample(DoubleArray(0), DoubleArray(0), doubleArrayOf(0.0, 0.01), 0.025)
        assertArrayEquals(doubleArrayOf(0.0, 0.0), empty.values, 0.0)
        assertEquals(listOf(true, true), empty.missing.toList())
    }
}
