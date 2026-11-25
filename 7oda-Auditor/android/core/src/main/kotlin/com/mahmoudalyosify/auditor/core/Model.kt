package com.mahmoudalyosify.auditor.core

import java.time.Instant
import java.util.UUID

// MARK: - Prototypes and samples (SPEC 9.1)

/** Prototype key; ordered by person ID (Unicode code points), then hand (right, left), then accessory. */
data class PrototypeKey(
    val personID: String,
    val hand: Hand,
    val accessory: Accessory = Accessory.none,
) : Comparable<PrototypeKey> {
    override fun compareTo(other: PrototypeKey): Int {
        if (personID != other.personID) return compareCodePoints(personID, other.personID)
        if (hand != other.hand) return hand.ordinal.compareTo(other.hand.ordinal)
        return accessory.ordinal.compareTo(other.accessory.ordinal)
    }

    private companion object {
        fun compareCodePoints(a: String, b: String): Int {
            var i = 0
            var j = 0
            while (i < a.length && j < b.length) {
                val ca = a.codePointAt(i)
                val cb = b.codePointAt(j)
                if (ca != cb) return ca.compareTo(cb)
                i += Character.charCount(ca)
                j += Character.charCount(cb)
            }
            val restA = i < a.length
            val restB = j < b.length
            return when {
                restA == restB -> 0
                restA -> 1
                else -> -1
            }
        }
    }
}

@Suppress("EnumEntryName")
enum class SampleSource {
    enrollment, accessory, adaptive;

    val rawValue: String get() = name

    companion object {
        fun from(raw: String): SampleSource? = entries.firstOrNull { it.name == raw }
    }
}

/** One v0 feature vector of a prototype. */
data class TemplateSample(
    val id: UUID = UUID.randomUUID(),
    val prototype: PrototypeKey,
    val source: SampleSource,
    val createdAt: Instant = Instant.now(),
    val features: List<Double>,
    val featureVersion: Int = FeatureExtractor.version,
    val recordingID: UUID? = null,
)

// MARK: - Config (SPEC 9.2)

data class ModelConfig(
    val shrinkageMin: Double = 0.1,
    val looQuantile: Double = 0.95,
    val absFactor: Double = 1.5,
) {
    companion object {
        val default: ModelConfig = ModelConfig()
    }
}

// MARK: - Gaussian model (SPEC 9.2)

/** Standardized features, shared Ledoit–Wolf shrunk covariance, one mean per class. */
class GaussianModel internal constructor(
    val mean: List<Double>,
    val scale: List<Double>,
    val classMeans: List<List<Double>>,
    val classCounts: List<Int>,
    val shrinkage: Double,
    val absThreshold: Double,
    val looDistances: List<Double>,
    internal val cholesky: Array<DoubleArray>,
    internal val mu: Double,
    internal val shrinkageLW: Double,
    internal val degenerate: Boolean,
    internal val jitter: Double,
    internal val looIndices: List<Int>,
) {
    private val meanArray: DoubleArray = mean.toDoubleArray()
    private val scaleArray: DoubleArray = scale.toDoubleArray()
    private val classMeanArrays: List<DoubleArray> = classMeans.map { it.toDoubleArray() }

    /** Squared Mahalanobis distance `D²` to every class mean; +infinity for a vector of the wrong length. */
    fun distances(x: List<Double>): List<Double> {
        val d = meanArray.size
        if (x.size != d) return List(classMeanArrays.size) { Double.POSITIVE_INFINITY }
        val z = DoubleArray(d) { j -> (x[j] - meanArray[j]) / scaleArray[j] }
        return classMeanArrays.map { m ->
            val v = DoubleArray(d) { j -> z[j] - m[j] }
            LinearAlgebra.forwardSquaredNorm(cholesky, v)
        }
    }

    internal class LedoitWolf(val empirical: Array<DoubleArray>, val mu: Double, val lambda: Double)

    companion object {
        internal const val scaleFloor: Double = 1e-6
        internal const val jitterStart: Double = 1e-6
        internal const val jitterRetries: Int = 5

        /** Fits the model (SPEC 9.2); null for no samples or malformed input. */
        fun fit(features: List<List<Double>>, labels: List<Int>, classCount: Int, config: ModelConfig): GaussianModel? {
            val n = features.size
            if (n <= 0 || labels.size != n || classCount <= 0) return null
            val d = features[0].size
            if (!features.all { it.size == d }) return null
            if (!labels.all { it in 0 until classCount }) return null
            val nD = n.toDouble()
            val x = Array(n) { features[it].toDoubleArray() }

            // 1. Standardize.
            val mu = DoubleArray(d)
            val sig = DoubleArray(d)
            for (j in 0 until d) {
                var s = 0.0
                for (i in 0 until n) s += x[i][j]
                val m = s / nD
                var ss = 0.0
                for (i in 0 until n) {
                    val dv = x[i][j] - m
                    ss += dv * dv
                }
                mu[j] = m
                sig[j] = Numerics.smax(Math.sqrt(ss / nD), scaleFloor)
            }
            val z = Array(n) { i -> DoubleArray(d) { j -> (x[i][j] - mu[j]) / sig[j] } }

            // 2. Class means and counts.
            val counts = IntArray(classCount)
            for (k in labels) counts[k] += 1
            val cmeans = Array(classCount) { DoubleArray(d) }
            for (k in 0 until classCount) {
                if (counts[k] <= 0) continue
                val sums = DoubleArray(d)
                for (i in 0 until n) {
                    if (labels[i] != k) continue
                    for (j in 0 until d) sums[j] = sums[j] + z[i][j]
                }
                for (j in 0 until d) cmeans[k][j] = sums[j] / counts[k].toDouble()
            }

            // 3. Residuals.
            val r = Array(n) { i ->
                val m = cmeans[labels[i]]
                DoubleArray(d) { j -> z[i][j] - m[j] }
            }
            val kPresent = counts.count { it > 0 }

            // 4. Ledoit–Wolf shrinkage.
            val lw = ledoitWolf(r)
            var cov: Array<DoubleArray>
            var lambda: Double
            var degenerate = false
            var shrinkageLW = lw.lambda
            if (n - kPresent < 2 || !(lw.mu > 1e-12)) {
                cov = LinearAlgebra.identity(d)
                lambda = 1.0
                degenerate = true
                shrinkageLW = Double.NaN
            } else {
                lambda = Numerics.smin(Numerics.smax(Numerics.smax(lw.lambda, config.shrinkageMin), 0.0), 1.0)
                val diagonal = lambda * lw.mu
                cov = Array(d) { j ->
                    DoubleArray(d) { k -> (1.0 - lambda) * lw.empirical[j][k] + (if (j == k) diagonal else 0.0) }
                }
            }

            // 5. Cholesky with jitter.
            var jitterUsed = 0.0
            var low = LinearAlgebra.cholesky(cov)
            if (low == null) {
                var jitter = jitterStart * (if (lw.mu > 0.0) lw.mu else 1.0)
                for (attempt in 0 until jitterRetries) {
                    low = LinearAlgebra.cholesky(LinearAlgebra.addingDiagonal(cov, jitter))
                    jitterUsed = jitter
                    if (low != null) break
                    jitter *= 10.0
                }
            }
            val factor: Array<DoubleArray>
            if (low != null) {
                factor = low
            } else {
                factor = LinearAlgebra.identity(d)
                lambda = 1.0
                degenerate = true
            }

            // 7. Leave-one-out threshold.
            val loo = ArrayList<Double>()
            val looIndices = ArrayList<Int>()
            for (i in 0 until n) {
                val k = labels[i]
                val nk = counts[k]
                if (nk < 2) continue
                val diff = DoubleArray(d)
                for (j in 0 until d) {
                    val mPrime = (nk.toDouble() * cmeans[k][j] - z[i][j]) / (nk - 1).toDouble()
                    diff[j] = z[i][j] - mPrime
                }
                loo.add(LinearAlgebra.forwardSquaredNorm(factor, diff))
                looIndices.add(i)
            }
            val q = Numerics.quantileLinear(loo, config.looQuantile)
            val absThreshold = if (q != null) config.absFactor * q else config.absFactor * 2.0 * d.toDouble()

            return GaussianModel(
                mean = mu.toList(),
                scale = sig.toList(),
                classMeans = cmeans.map { it.toList() },
                classCounts = counts.toList(),
                shrinkage = lambda,
                absThreshold = absThreshold,
                looDistances = loo.toList(),
                cholesky = factor,
                mu = lw.mu,
                shrinkageLW = shrinkageLW,
                degenerate = degenerate,
                jitter = jitterUsed,
                looIndices = looIndices.toList(),
            )
        }

        /** Ledoit–Wolf estimate on residuals (sklearn formula, left-to-right sums as the reference). */
        internal fun ledoitWolf(r: Array<DoubleArray>): LedoitWolf {
            val n = r.size
            val d = if (r.isNotEmpty()) r[0].size else 0
            if (n == 0 || d == 0) return LedoitWolf(LinearAlgebra.identity(d), 0.0, 0.0)
            val nD = n.toDouble()
            val dD = d.toDouble()
            val gram = LinearAlgebra.zeros(d)
            val gram2 = LinearAlgebra.zeros(d)
            for (i in 0 until n) {
                val row = r[i]
                for (j in 0 until d) {
                    val rj = row[j]
                    val rj2 = rj * rj
                    for (k in j until d) {
                        val rk = row[k]
                        gram[j][k] = gram[j][k] + rj * rk
                        gram2[j][k] = gram2[j][k] + rj2 * (rk * rk)
                    }
                }
            }
            for (j in 0 until d) {
                for (k in 0 until j) {
                    gram[j][k] = gram[k][j]
                    gram2[j][k] = gram2[k][j]
                }
            }
            val empirical = LinearAlgebra.zeros(d)
            var traceSum = 0.0
            for (j in 0 until d) {
                for (k in 0 until d) empirical[j][k] = gram[j][k] / nD
                traceSum += gram[j][j] / nD
            }
            val mu = traceSum / dD
            var betaRaw = 0.0
            var deltaRaw = 0.0
            for (j in 0 until d) {
                for (k in 0 until d) {
                    betaRaw += gram2[j][k]
                    deltaRaw += gram[j][k] * gram[j][k]
                }
            }
            deltaRaw /= (nD * nD)
            var beta = (betaRaw / nD - deltaRaw) / (dD * nD)
            val delta = (deltaRaw - 2.0 * mu * traceSum + dD * (mu * mu)) / dD
            beta = Numerics.smin(beta, delta)
            val lambda = if (delta == 0.0) 0.0 else beta / delta
            return LedoitWolf(empirical, mu, lambda)
        }
    }
}
