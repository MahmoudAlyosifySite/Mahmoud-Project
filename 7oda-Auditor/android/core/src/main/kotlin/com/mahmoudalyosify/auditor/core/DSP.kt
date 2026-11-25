package com.mahmoudalyosify.auditor.core

// MARK: - Biquads (SPEC 5.3, 5.4)

/** Normalized biquad (`a0 = 1`) designed with the RBJ cookbook formulas. */
data class Biquad(val b0: Double, val b1: Double, val b2: Double, val a1: Double, val a2: Double) {

    /** Direct-form I filter with zero initial state (SPEC 5.3). */
    fun filter(x: DoubleArray): DoubleArray {
        val out = DoubleArray(x.size)
        var x1 = 0.0
        var x2 = 0.0
        var y1 = 0.0
        var y2 = 0.0
        for (i in x.indices) {
            val xi = x[i]
            val yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            out[i] = yi
            x2 = x1
            x1 = xi
            y2 = y1
            y1 = yi
        }
        return out
    }

    /** Zero-phase filter with odd reflection padding and DC-offset start (SPEC 5.4). */
    fun filtfiltZ(x: DoubleArray, pad: Int, dcGain: Double): DoubleArray {
        val n = x.size
        if (n == 0) return DoubleArray(0)
        if (n == 1) return doubleArrayOf(dcGain * x[0])
        val p = minOf(maxOf(pad, 0), n - 1)
        val xp = DoubleArray(n + 2 * p)
        var w = 0
        var k = p
        while (k >= 1) {
            xp[w++] = 2.0 * x[0] - x[k]
            k -= 1
        }
        for (v in x) xp[w++] = v
        k = n - 2
        while (k >= n - 1 - p) {
            xp[w++] = 2.0 * x[n - 1] - x[k]
            k -= 1
        }
        val c1 = xp[0]
        val y = filter(DoubleArray(xp.size) { xp[it] - c1 })
        for (i in y.indices) y[i] = y[i] + dcGain * c1
        val r = y.reversedArray()
        val c2 = r[0]
        val z = filter(DoubleArray(r.size) { r[it] - c2 })
        for (i in z.indices) z[i] = z[i] + dcGain * c2
        val zr = z.reversedArray()
        return zr.copyOfRange(p, p + n)
    }

    companion object {
        val butterworthQ: Double = 1.0 / Math.sqrt(2.0)

        private fun normalized(b0: Double, b1: Double, b2: Double, a0: Double, a1: Double, a2: Double): Biquad =
            Biquad(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)

        fun highPass(cutoff: Double, fs: Double, q: Double): Biquad {
            val w0 = 2.0 * Math.PI * cutoff / fs
            val c = Math.cos(w0)
            val alpha = Math.sin(w0) / (2.0 * q)
            return normalized((1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0, 1.0 + alpha, -2.0 * c, 1.0 - alpha)
        }

        fun lowPass(cutoff: Double, fs: Double, q: Double): Biquad {
            val w0 = 2.0 * Math.PI * cutoff / fs
            val c = Math.cos(w0)
            val alpha = Math.sin(w0) / (2.0 * q)
            return normalized((1.0 - c) / 2.0, 1.0 - c, (1.0 - c) / 2.0, 1.0 + alpha, -2.0 * c, 1.0 - alpha)
        }

        fun bandPass(center: Double, fs: Double, q: Double): Biquad {
            val w0 = 2.0 * Math.PI * center / fs
            val c = Math.cos(w0)
            val alpha = Math.sin(w0) / (2.0 * q)
            return normalized(alpha, 0.0, -alpha, 1.0 + alpha, -2.0 * c, 1.0 - alpha)
        }
    }
}

// MARK: - Resampling (SPEC 5.2)

/** Result of [Resampler.resample]: interpolated values and the per-grid-point gap flags. */
class Resampled(val values: DoubleArray, val missing: BooleanArray)

object Resampler {
    /** Linear interpolation onto `grid`; `t`, `v` must already be sorted and deduplicated (SPEC 5.2). */
    fun resample(t: DoubleArray, v: DoubleArray, grid: DoubleArray, gapTolerance: Double): Resampled {
        val m = minOf(t.size, v.size)
        val count = grid.size
        if (m == 0) {
            return Resampled(DoubleArray(count), BooleanArray(count) { true })
        }
        val values = DoubleArray(count)
        val missing = BooleanArray(count)
        for (k in 0 until count) {
            val g = grid[k]
            var lo = 0
            var hi = m
            while (lo < hi) {
                val mid = (lo + hi) / 2
                if (t[mid] <= g) lo = mid + 1 else hi = mid
            }
            val j = lo - 1
            if (j < 0) {
                values[k] = v[0]
                missing[k] = true
            } else if (j == m - 1) {
                values[k] = v[m - 1]
                missing[k] = (g - t[m - 1]) > gapTolerance
            } else {
                val t0 = t[j]
                val t1 = t[j + 1]
                val w = (g - t0) / (t1 - t0)
                val v0 = v[j]
                values[k] = v0 + w * (v[j + 1] - v0)
                missing[k] = (t1 - t0) > gapTolerance
            }
        }
        return Resampled(values, missing)
    }
}

// MARK: - Sort and deduplicate (SPEC 4)

object StreamCleaner {
    /** Stable sort by time (non-finite times dropped), keeping only strictly increasing timestamps. */
    internal fun keptIndices(t: List<Double>, count: Int): List<Int> {
        val n = minOf(maxOf(count, 0), t.size)
        val order = (0 until n).filter { t[it].isFinite() }.sortedWith(Comparator { a, b ->
            val ta = t[a]
            val tb = t[b]
            when {
                ta < tb -> -1
                tb < ta -> 1
                else -> a.compareTo(b)
            }
        })
        val keep = ArrayList<Int>(order.size)
        var last = 0.0
        for (i in order) {
            if (keep.isEmpty() || t[i] > last) {
                keep.add(i)
                last = t[i]
            }
        }
        return keep
    }

    fun sortDedup(s: SensorStream): SensorStream {
        val n = minOf(minOf(s.t.size, s.x.size), minOf(s.y.size, s.z.size))
        val keep = keptIndices(s.t, n)
        return SensorStream(keep.map { s.t[it] }, keep.map { s.x[it] }, keep.map { s.y[it] }, keep.map { s.z[it] })
    }

    fun sortDedup(m: MotionStream): MotionStream {
        val n = listOf(
            m.t.size, m.gravityX.size, m.gravityY.size, m.gravityZ.size,
            m.rotationX.size, m.rotationY.size, m.rotationZ.size,
        ).min()
        val keep = keptIndices(m.t, n)
        fun pick(values: List<Double>): List<Double> {
            if (values.size < n) return emptyList()
            return keep.map { values[it] }
        }
        return MotionStream(
            t = pick(m.t),
            gravityX = pick(m.gravityX), gravityY = pick(m.gravityY), gravityZ = pick(m.gravityZ),
            userAccelX = pick(m.userAccelX), userAccelY = pick(m.userAccelY), userAccelZ = pick(m.userAccelZ),
            rotationX = pick(m.rotationX), rotationY = pick(m.rotationY), rotationZ = pick(m.rotationZ),
            attitudeW = pick(m.attitudeW), attitudeX = pick(m.attitudeX),
            attitudeY = pick(m.attitudeY), attitudeZ = pick(m.attitudeZ),
        )
    }
}
