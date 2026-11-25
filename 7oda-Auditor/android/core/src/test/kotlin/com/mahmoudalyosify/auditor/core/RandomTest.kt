package com.mahmoudalyosify.auditor.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class RandomTest {
    @Test
    fun rngFixtureParity() = checks { c ->
        val cases = Fixtures.json("rng_vectors").at("cases").arr()
        c.check(cases.isNotEmpty()) { "rng_vectors.json has no cases" }
        for (case in cases) {
            val seedText = case.at("seed").str()
            val seed = case.at("seed").hex()
            c.equal(SeedFormat.string(seed), seedText, "seed formatting")
            val sm = SplitMix64(seed)
            for ((i, v) in case.at("splitmix").arr().withIndex()) c.equal(sm.next(), v.hex(), "splitmix[$i] for $seedText")
            val x = Xoshiro256StarStar(seed)
            for ((i, v) in case.at("xoshiro").arr().withIndex()) c.equal(x.next(), v.hex(), "xoshiro[$i] for $seedText")
            val u = Xoshiro256StarStar(seed)
            for ((i, v) in case.at("uniform").dbls().withIndex()) c.equal(u.uniform(), v, "uniform[$i] for $seedText")
            for (entry in case.at("ints").arr()) {
                val lo = entry.at("lo").int()
                val hi = entry.at("hi").int()
                val g = Xoshiro256StarStar(seed)
                for ((i, v) in entry.at("values").ints().withIndex()) c.equal(g.int(lo, hi), v, "int($lo, $hi)[$i] for $seedText")
            }
            if (case.has("uniformRange")) {
                for (entry in case.at("uniformRange").arr()) {
                    val a = entry.at("a").dbl()
                    val b = entry.at("b").dbl()
                    val g = Xoshiro256StarStar(seed)
                    for ((i, v) in entry.at("values").dbls().withIndex()) c.equal(g.uniform(a, b), v, "uniform($a, $b)[$i] for $seedText")
                }
            }
        }
    }

    @Test
    fun splitMixKnownValue() {
        assertEquals(0xE220A8397B1DCDAFuL, SplitMix64(0uL).next())
    }

    @Test
    fun seedFormat() {
        assertEquals("0x00000000000004d2", SeedFormat.string(1234uL))
        assertEquals("0x0000000000000000", SeedFormat.string(0uL))
        assertEquals("0xffffffffffffffff", SeedFormat.string(ULong.MAX_VALUE))
        assertEquals(1234uL, SeedFormat.parse("0x00000000000004d2"))
        assertEquals(ULong.MAX_VALUE, SeedFormat.parse("0xFFFFFFFFFFFFFFFF"))
        assertNull(SeedFormat.parse("1234"))
        assertNull(SeedFormat.parse("0x"))
        assertNull(SeedFormat.parse("0x1g"))
        assertNull(SeedFormat.parse("0x00000000000000001"))
        for (seed in listOf(0uL, 1uL, 42uL, 0x8000000000000000uL, ULong.MAX_VALUE)) {
            assertEquals(seed, SeedFormat.parse(SeedFormat.string(seed)))
        }
    }

    @Test
    fun rangesAndDeterminism() {
        val a = Xoshiro256StarStar(7uL)
        val b = Xoshiro256StarStar(7uL)
        repeat(1000) {
            val u = a.uniform()
            assertTrue(u >= 0.0 && u < 1.0)
            assertEquals(u, b.uniform(), 0.0)
            val r = a.uniform(0.25, 0.5)
            assertTrue(r >= 0.25 && r <= 0.5)
            b.uniform(0.25, 0.5)
            val i = a.int(6, 8)
            assertTrue(i in 6..8)
            assertEquals(i, b.int(6, 8))
        }
        assertEquals(3, Xoshiro256StarStar(9uL).int(3, 3))
    }
}
