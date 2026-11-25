package com.mahmoudalyosify.auditor.core

/**
 * Small numeric helpers whose semantics mirror `research/auditor_ref/_numerics.py` and the Swift port
 * exactly (left-to-right sums, numpy median and "linear" quantile, Swift `min`/`max` NaN behaviour).
 */
internal object Numerics {
    /** Bound used by [truncatingInt] so that later index arithmetic cannot overflow a 32-bit Int. */
    const val INT_LIMIT: Int = 1_000_000_000

    /** Truncation toward zero (Python `int()`), never throws: NaN gives 0 and large values are clamped. */
    fun truncatingInt(x: Double): Int {
        if (x.isNaN()) return 0
        if (x >= 1.0e9) return INT_LIMIT
        if (x <= -1.0e9) return -INT_LIMIT
        return x.toInt()
    }

    /** Swift `min(x, y)`: `y < x ? y : x`. */
    fun smin(x: Double, y: Double): Double = if (y < x) y else x

    /** Swift `max(x, y)`: `y >= x ? y : x`. */
    fun smax(x: Double, y: Double): Double = if (y >= x) y else x

    /** Left-to-right sum starting from 0.0. */
    fun sum(values: DoubleArray): Double {
        var s = 0.0
        for (v in values) s += v
        return s
    }

    fun sum(values: List<Double>): Double {
        var s = 0.0
        for (v in values) s += v
        return s
    }

    /** Left-to-right sum of `values[range]` (the range must be valid). */
    fun sum(values: DoubleArray, range: IntRange): Double {
        var s = 0.0
        for (i in range) s += values[i]
        return s
    }

    /** `sum / count`; 0.0 for an empty input. */
    fun mean(values: DoubleArray): Double = if (values.isEmpty()) 0.0 else sum(values) / values.size.toDouble()

    fun mean(values: List<Double>): Double = if (values.isEmpty()) 0.0 else sum(values) / values.size.toDouble()

    /** Mean of `values[range]`; 0.0 for an empty range (the range must be valid). */
    fun mean(values: DoubleArray, range: IntRange): Double {
        if (range.isEmpty()) return 0.0
        return sum(values, range) / (range.last - range.first + 1).toDouble()
    }

    /** Ascending order that places NaN last. */
    val totalOrder: Comparator<Double> = Comparator { a, b ->
        when {
            a.isNaN() && b.isNaN() -> 0
            a.isNaN() -> 1
            b.isNaN() -> -1
            a < b -> -1
            b < a -> 1
            else -> 0
        }
    }

    /** Median with numpy semantics (mean of the two middle values for even counts); 0.0 for an empty input. */
    fun median(values: List<Double>): Double {
        val n = values.size
        if (n == 0) return 0.0
        val sorted = values.sortedWith(totalOrder)
        val mid = n / 2
        if (n % 2 == 1) return sorted[mid]
        return (sorted[mid - 1] + sorted[mid]) / 2.0
    }

    fun median(values: DoubleArray): Double = median(values.asList())

    /** `numpy.quantile(values, q)` with the default "linear" method; null for an empty input. */
    fun quantileLinear(values: List<Double>, q: Double): Double? {
        val n = values.size
        if (n == 0) return null
        val sorted = values.sortedWith(totalOrder)
        val h = (n - 1).toDouble() * q
        if (h >= (n - 1).toDouble()) return sorted[n - 1]
        if (!(h >= 0.0)) return sorted[0]
        val i = minOf(maxOf(truncatingInt(Math.floor(h)), 0), n - 2)
        val g = h - i.toDouble()
        val a = sorted[i]
        val b = sorted[i + 1]
        val diff = b - a
        if (g >= 0.5) return b - diff * (1.0 - g)
        return a + diff * g
    }
}

/** Dense helpers for the Gaussian model and the v1 hand model, with left-to-right sums. */
internal object LinearAlgebra {
    fun zeros(d: Int): Array<DoubleArray> = Array(d) { DoubleArray(d) }

    /** `d x d` identity. */
    fun identity(d: Int): Array<DoubleArray> {
        val m = zeros(d)
        for (i in 0 until d) m[i][i] = 1.0
        return m
    }

    /** `s + jitter * I`. */
    fun addingDiagonal(s: Array<DoubleArray>, jitter: Double): Array<DoubleArray> {
        val m = Array(s.size) { s[it].copyOf() }
        for (i in m.indices) {
            if (i < m[i].size) m[i][i] = m[i][i] + jitter
        }
        return m
    }

    /** Lower-triangular `L` with `S = L Lᵀ` (Cholesky–Banachiewicz); null on a non-positive (or NaN) pivot. */
    fun cholesky(s: Array<DoubleArray>): Array<DoubleArray>? {
        val d = s.size
        if (!s.all { it.size == d }) return null
        val low = zeros(d)
        for (i in 0 until d) {
            for (j in 0..i) {
                var acc = s[i][j]
                for (k in 0 until j) {
                    acc -= low[i][k] * low[j][k]
                }
                if (i == j) {
                    if (!(acc > 0.0)) return null
                    low[i][i] = Math.sqrt(acc)
                } else {
                    low[i][j] = acc / low[j][j]
                }
            }
        }
        return low
    }

    /** `‖L⁻¹ v‖²` by forward substitution. */
    fun forwardSquaredNorm(low: Array<DoubleArray>, v: DoubleArray): Double {
        val d = low.size
        val y = DoubleArray(d)
        for (i in 0 until d) {
            var acc = v[i]
            for (k in 0 until i) {
                acc -= low[i][k] * y[k]
            }
            y[i] = acc / low[i][i]
        }
        var s = 0.0
        for (i in 0 until d) s += y[i] * y[i]
        return s
    }

    /** Solves `(L Lᵀ) x = b` for a lower Cholesky factor `L`. */
    fun choleskySolve(low: Array<DoubleArray>, b: DoubleArray): DoubleArray {
        val d = low.size
        val y = DoubleArray(d)
        for (i in 0 until d) {
            var acc = if (i < b.size) b[i] else 0.0
            for (k in 0 until i) {
                acc -= low[i][k] * y[k]
            }
            y[i] = acc / low[i][i]
        }
        val x = DoubleArray(d)
        var i = d - 1
        while (i >= 0) {
            var acc = y[i]
            var k = i + 1
            while (k < d) {
                acc -= low[k][i] * x[k]
                k += 1
            }
            x[i] = acc / low[i][i]
            i -= 1
        }
        return x
    }
}
