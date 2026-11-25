package com.mahmoudalyosify.auditor.core

import java.time.Instant

// MARK: - Encoder input (SPEC 14.1)

object EncoderInput {
    const val channels: Int = 10
    const val length: Int = 256
    const val patternStart: Int = 25

    internal const val accelScale: Double = 0.02
    internal const val gyroScale: Double = 0.05
    internal const val clip: Double = 50.0

    /** Input tensor `x` of SPEC 14.1, row-major `[channel][u]`, size 2560 (ONNX shape `[1, 10, 256]`). */
    fun build(p: ProcessedSignal): FloatArray {
        val x = DoubleArray(channels * length)
        val n = p.n
        val lag = p.alignment.lag
        val channelsOK = p.accelHP.size == 3 && p.gyroHP.size == 3 && p.gravity.size == 3
        val streams = listOf(p.accelHP, p.gyroHP, p.gravity).flatten()
        val lengthsOK = streams.all { it.size == n } && p.template.size == n
        if (channelsOK && lengthsOK) {
            for (u in 0 until length) {
                val src = u + lag
                if (src < 0 || src >= n) continue
                for (c in 0 until 3) {
                    x[c * length + u] = p.accelHP[c][src] / accelScale
                    x[(3 + c) * length + u] = p.gyroHP[c][src] / gyroScale
                    x[(6 + c) * length + u] = p.gravity[c][src]
                }
                if (u < n) x[9 * length + u] = p.template[u]
            }
        }
        return FloatArray(x.size) { i -> Numerics.smin(Numerics.smax(x[i], -clip), clip).toFloat() }
    }
}

// MARK: - Embedding samples and config (SPEC 14.3, 14.5)

/** One encoder embedding of a prototype (128 values; renormalized in Double). */
data class EmbeddingSample(
    val prototype: PrototypeKey,
    val embedding: List<Double>,
    val source: SampleSource = SampleSource.enrollment,
    val createdAt: Instant = Instant.now(),
)

data class EmbeddingConfig(
    val looQuantile: Double = 0.95,
    val slack: Double = 0.0,
    val nameMargin: Double = 0.03,
    val handMargin: Double = 0.0,
    val handShrinkage: Double = 0.1,
    val maxPatterns: Int = 3,
    val adaptiveMaxPerPrototype: Int = 20,
) {
    companion object {
        val default: EmbeddingConfig = EmbeddingConfig()
    }
}

// MARK: - Embedding identifier (SPEC 14.3)

/** Cosine prototypes over encoder embeddings; decisions in cosine-distance units (SPEC 14.3). */
class EmbeddingIdentifier(
    samples: List<EmbeddingSample>,
    val config: EmbeddingConfig = EmbeddingConfig.default,
) {
    val prototypes: List<PrototypeKey>
    val absThreshold: Double

    internal val dimension: Int
    internal val prototypeVectors: List<DoubleArray>
    internal val looSimilarities: List<Double>
    internal val simThreshold: Double
    internal val handWeights: DoubleArray?
    internal val handBias: Double

    init {
        val dim = samples.firstOrNull()?.embedding?.size ?: 0
        val used = if (dim > 0) samples.filter { it.embedding.size == dim } else emptyList()
        dimension = dim
        val protos = used.map { it.prototype }.distinct().sorted()
        prototypes = protos
        val index = HashMap<PrototypeKey, Int>()
        for ((i, p) in protos.withIndex()) index[p] = i
        val unit = used.map { normalize(it.embedding.toDoubleArray()) }
        val sums = List(protos.size) { DoubleArray(dim) }
        val counts = IntArray(protos.size)
        val labels = ArrayList<Int>(used.size)
        for ((i, s) in used.withIndex()) {
            val k = index[s.prototype] ?: 0
            labels.add(k)
            counts[k] += 1
            for (j in 0 until dim) sums[k][j] = sums[k][j] + unit[i][j]
        }
        prototypeVectors = sums.map { normalize(it) }
        val loo = ArrayList<Double>()
        for (i in used.indices) {
            val k = labels[i]
            if (counts[k] < 2) continue
            val rest = sums[k].copyOf()
            for (j in 0 until dim) rest[j] = rest[j] - unit[i][j]
            loo.add(dot(normalize(rest), unit[i]))
        }
        looSimilarities = loo.toList()
        val q = Numerics.quantileLinear(loo, 1.0 - config.looQuantile)
        val threshold = if (q != null) q - config.slack else -1.0 - config.slack
        simThreshold = threshold
        absThreshold = 1.0 - threshold
        val hands = used.map { it.prototype.hand }
        val model = handModel(unit, hands, dim, config.handShrinkage)
        handWeights = model?.first
        handBias = model?.second ?: 0.0
    }

    internal fun similarities(pattern: List<Double>): List<Double> {
        if (pattern.size != dimension || dimension <= 0) {
            return List(prototypes.size) { Double.NEGATIVE_INFINITY }
        }
        val e = normalize(pattern.toDoubleArray())
        return prototypeVectors.map { dot(e, it) }
    }

    internal fun handLogOdds(pattern: List<Double>): Double {
        val w = handWeights ?: return 0.0
        if (pattern.size != w.size) return 0.0
        return dot(w, normalize(pattern.toDoubleArray())) + handBias
    }

    /** SPEC 14.3 on the given patterns (one embedding per pattern). */
    fun decide(patterns: List<List<Double>>): Decision {
        val m = patterns.size
        if (prototypes.isEmpty()) {
            return Decision(outcome = DecisionOutcome.Unknown(UnknownReason.noPeople), patternsUsed = m, absThreshold = absThreshold)
        }
        if (m == 0) {
            return Decision(outcome = DecisionOutcome.NeedMorePatterns, patternsUsed = 0, absThreshold = absThreshold)
        }
        val kCount = prototypes.size
        val perPattern = patterns.map { similarities(it) }
        val sbar = DoubleArray(kCount)
        for (k in 0 until kCount) {
            var s = 0.0
            for (j in 0 until m) s += perPattern[j][k]
            sbar[k] = s / m.toDouble()
        }
        val dbar = DoubleArray(kCount) { 1.0 - sbar[it] }
        val hasHead = handWeights != null
        val hbar = if (hasHead) Numerics.mean(patterns.map { handLogOdds(it) }) else 0.0
        val r = DecisionRule.outcome(
            prototypes, dbar, hbar, hasHead, m, absThreshold,
            config.nameMargin, config.handMargin, config.maxPatterns,
        )
        val distances = prototypes.mapIndexed { k, p -> PrototypeDistance(p, dbar[k]) }
        return Decision(
            outcome = r.outcome,
            patternsUsed = m,
            bestPrototype = prototypes[r.kstar],
            bestDistance = r.best,
            marginName = r.marginName,
            marginHand = r.marginHand,
            handLogOdds = hbar,
            absThreshold = absThreshold,
            personScores = personScores(sbar),
            prototypeDistances = distances,
        )
    }

    /** Softmax over persons of `S̄_max(person) / 0.05`. */
    internal fun personScores(sbar: DoubleArray): List<PersonScore> {
        val persons = ArrayList<String>()
        val best = ArrayList<Double>()
        for ((k, p) in prototypes.withIndex()) {
            val i = persons.indexOf(p.personID)
            if (i >= 0) {
                if (sbar[k] > best[i]) best[i] = sbar[k]
            } else {
                persons.add(p.personID)
                best.add(sbar[k])
            }
        }
        if (persons.isEmpty()) return emptyList()
        val logits = best.map { it / scoreTemperature }
        var top = logits[0]
        for (v in logits.drop(1)) {
            if (v > top) top = v
        }
        val probabilities: List<Double> = if (top.isFinite()) {
            val ex = logits.map { Math.exp(it - top) }
            val total = Numerics.sum(ex)
            ex.map { it / total }
        } else {
            List(persons.size) { 1.0 / persons.size.toDouble() }
        }
        val scores = persons.indices.map { PersonScore(persons[it], probabilities[it], 1.0 - best[it]) }
        return DecisionRule.descendingOrder(scores.map { it.probability }).map { scores[it] }
    }

    /** Template update analog of SPEC 9.5 (SPEC 14.3); empty if not eligible. */
    fun adaptiveSamples(decision: Decision, patterns: List<List<Double>>, now: Instant): List<EmbeddingSample> {
        if (prototypes.isEmpty()) return emptyList()
        val proto = decision.bestPrototype ?: return emptyList()
        if (decision.outcome !is DecisionOutcome.Identified) return emptyList()
        if (!(decision.marginName >= adaptiveNameFactor * config.nameMargin &&
                decision.bestDistance <= decision.absThreshold - adaptiveDistanceMargin)
        ) {
            return emptyList()
        }
        return patterns.map { EmbeddingSample(proto, it, SampleSource.adaptive, now) }
    }

    companion object {
        internal const val adaptiveNameFactor: Double = 2.0
        internal const val adaptiveDistanceMargin: Double = 0.05
        internal const val scoreTemperature: Double = 0.05

        /** Keeps at most `cap` adaptive samples per prototype, dropping the oldest. */
        fun applyAdaptiveCap(samples: List<EmbeddingSample>, cap: Int): List<EmbeddingSample> =
            AdaptiveCap.apply(samples, cap, { it.source == SampleSource.adaptive }, { it.prototype }, { it.createdAt })

        internal fun handModel(unit: List<DoubleArray>, hands: List<Hand>, d: Int, shrinkage: Double): Pair<DoubleArray, Double>? {
            val right = unit.indices.filter { hands[it] == Hand.right }
            val left = unit.indices.filter { hands[it] == Hand.left }
            if (right.size < 2 || left.size < 2 || d <= 0) return null
            fun meanOf(rows: List<Int>): DoubleArray {
                val m = DoubleArray(d)
                for (i in rows) {
                    for (j in 0 until d) m[j] = m[j] + unit[i][j]
                }
                return DoubleArray(d) { m[it] / rows.size.toDouble() }
            }
            val muR = meanOf(right)
            val muL = meanOf(left)
            val residuals = ArrayList<DoubleArray>()
            for (i in right) residuals.add(DoubleArray(d) { unit[i][it] - muR[it] })
            for (i in left) residuals.add(DoubleArray(d) { unit[i][it] - muL[it] })
            val n = residuals.size.toDouble()
            val cov = LinearAlgebra.zeros(d)
            for (r in residuals) {
                for (j in 0 until d) {
                    val rj = r[j]
                    for (k in j until d) cov[j][k] = cov[j][k] + rj * r[k]
                }
            }
            var trace = 0.0
            for (j in 0 until d) {
                for (k in j until d) {
                    cov[j][k] = cov[j][k] / n
                    cov[k][j] = cov[j][k]
                }
                trace += cov[j][j]
            }
            val diagonal = shrinkage * trace / d.toDouble()
            for (j in 0 until d) {
                for (k in 0 until d) {
                    cov[j][k] = (1.0 - shrinkage) * cov[j][k] + (if (j == k) diagonal else 0.0)
                }
            }
            val low = LinearAlgebra.cholesky(cov) ?: return null
            val delta = DoubleArray(d) { muR[it] - muL[it] }
            val w = LinearAlgebra.choleskySolve(low, delta)
            val sum = DoubleArray(d) { muR[it] + muL[it] }
            return Pair(w, -0.5 * dot(w, sum))
        }

        internal fun dot(a: DoubleArray, b: DoubleArray): Double {
            var s = 0.0
            for (i in 0 until minOf(a.size, b.size)) s += a[i] * b[i]
            return s
        }

        internal fun normalize(v: DoubleArray): DoubleArray {
            val norm = Math.sqrt(dot(v, v))
            val scale = Numerics.smax(norm, 1e-12)
            return DoubleArray(v.size) { v[it] / scale }
        }
    }
}
