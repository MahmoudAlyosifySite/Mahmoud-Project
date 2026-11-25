package com.mahmoudalyosify.auditor.core

/** SplitMix64, used only to seed [Xoshiro256StarStar] (SPEC 2.1). */
class SplitMix64(seed: ULong) {
    /** Current generator state. */
    var state: ULong = seed
        private set

    /** Next 64-bit output (SPEC 2.1). Unsigned arithmetic wraps, as required. */
    fun next(): ULong {
        state += 0x9E3779B97F4A7C15uL
        var z = state
        z = (z xor (z shr 30)) * 0xBF58476D1CE4E5B9uL
        z = (z xor (z shr 27)) * 0x94D049BB133111EBuL
        return z xor (z shr 31)
    }
}

/**
 * xoshiro256** seeded through SplitMix64 (SPEC 2.2). Every random choice in a challenge comes from
 * this generator, so a seed fully determines a pattern.
 */
class Xoshiro256StarStar(seed: ULong) {
    private var s0: ULong
    private var s1: ULong
    private var s2: ULong
    private var s3: ULong

    init {
        val sm = SplitMix64(seed)
        s0 = sm.next()
        s1 = sm.next()
        s2 = sm.next()
        s3 = sm.next()
    }

    /** Next 64-bit output (SPEC 2.2). */
    fun next(): ULong {
        val result = rotl(s1 * 5uL, 7) * 9uL
        val t = s1 shl 17
        s2 = s2 xor s0
        s3 = s3 xor s1
        s1 = s1 xor s2
        s0 = s0 xor s3
        s2 = s2 xor t
        s3 = rotl(s3, 45)
        return result
    }

    /** Uniform double in [0, 1): `(next() >> 11) * 2^-53`. */
    fun uniform(): Double = (next() shr 11).toDouble() * TWO_POW_MINUS_53

    /** `a + (b - a) * uniform()`. */
    fun uniform(a: Double, b: Double): Double = a + (b - a) * uniform()

    /** Integer in `lo..hi` (inclusive): `lo + min(floor(uniform() * (hi - lo + 1)), hi - lo)`. */
    fun int(lo: Int, hi: Int): Int {
        val width = hi.toLong() - lo.toLong()
        val span = (width + 1L).toDouble()
        val scaled = Math.floor(uniform() * span)
        val offset: Long = if (scaled.isFinite() && scaled < 9.0e18 && scaled > -9.0e18) scaled.toLong() else width
        return (lo.toLong() + minOf(offset, width)).toInt()
    }

    private companion object {
        const val TWO_POW_MINUS_53: Double = 1.0 / 9007199254740992.0

        fun rotl(x: ULong, k: Int): ULong = (x shl k) or (x shr (64 - k))
    }
}

/**
 * Seeds are written as lowercase hex with a `0x` prefix and exactly 16 digits, e.g. `"0x00000000000004d2"`.
 */
object SeedFormat {
    /** `"0x"` followed by 16 lowercase hex digits. */
    fun string(seed: ULong): String = "0x" + seed.toString(16).padStart(16, '0')

    /** Parses `0x` followed by 1 to 16 hex digits (either case); null otherwise. */
    fun parse(s: String): ULong? {
        if (!s.startsWith("0x")) return null
        val digits = s.substring(2)
        if (digits.isEmpty() || digits.length > 16) return null
        for (ch in digits) {
            val ok = ch in '0'..'9' || ch in 'a'..'f' || ch in 'A'..'F'
            if (!ok) return null
        }
        return digits.toULongOrNull(16)
    }
}
