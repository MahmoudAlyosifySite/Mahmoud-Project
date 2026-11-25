package com.mahmoudalyosify.auditor.core

import java.time.Instant

// MARK: - Config and results (SPEC 9.4, 9.5)

data class IdentifierConfig(
    val nameMargin: Double = 6.0,
    val handMargin: Double = 3.0,
    val maxPatterns: Int = 3,
    val adaptiveMaxPerPrototype: Int = 20,
    val adaptiveNameFactor: Double = 2.0,
    val adaptiveAbsFactor: Double = 0.75,
) {
    companion object {
        val default: IdentifierConfig = IdentifierConfig()
    }
}

@Suppress("EnumEntryName")
enum class UnknownReason {
    noPeople, noMatch, ambiguous;

    val rawValue: String get() = name

    companion object {
        fun from(raw: String): UnknownReason? = entries.firstOrNull { it.name == raw }
    }
}

sealed class DecisionOutcome {
    data class Identified(val personID: String, val hand: Hand) : DecisionOutcome()
    data class IdentifiedHandUnclear(val personID: String) : DecisionOutcome()
    data object NeedMorePatterns : DecisionOutcome()
    data class Unknown(val reason: UnknownReason) : DecisionOutcome()
}

data class PersonScore(val personID: String, val probability: Double, val bestDistance: Double)

data class PrototypeDistance(val prototype: PrototypeKey, val distance: Double)

/** Result of a check. Margins are +infinity when there is no competing person / other hand. */
data class Decision(
    val outcome: DecisionOutcome,
    val patternsUsed: Int,
    val bestPrototype: PrototypeKey? = null,
    val bestDistance: Double = Double.POSITIVE_INFINITY,
    val marginName: Double = Double.POSITIVE_INFINITY,
    val marginHand: Double = Double.POSITIVE_INFINITY,
    val handLogOdds: Double = 0.0,
    val absThreshold: Double = 0.0,
    /** Sorted by descending probability. */
    val personScores: List<PersonScore> = emptyList(),
    /** In prototype order. */
    val prototypeDistances: List<PrototypeDistance> = emptyList(),
)

/** Shared decision logic of SPEC 9.4 / 14.3 on fused distances `dbar` and fused hand log-odds `hbar`. */
internal object DecisionRule {
    fun outcome(
        prototypes: List<PrototypeKey>,
        dbar: DoubleArray,
        hbar: Double,
        hasHead: Boolean,
        m: Int,
        absThreshold: Double,
        nameMargin: Double,
        handMargin: Double,
        maxPatterns: Int,
    ): Partial {
        val kCount = prototypes.size
        var kstar = 0
        for (k in 1 until maxOf(kCount, 1)) {
            if (dbar[k] < dbar[kstar]) kstar = k
        }
        val best = dbar[kstar]
        val person = prototypes[kstar].personID
        val hand = prototypes[kstar].hand
        var otherPerson = Double.POSITIVE_INFINITY
        var otherHand = Double.POSITIVE_INFINITY
        for (k in 0 until kCount) {
            val p = prototypes[k]
            if (p.personID != person) {
                if (dbar[k] < otherPerson) otherPerson = dbar[k]
            } else if (p.hand != hand) {
                if (dbar[k] < otherHand) otherHand = dbar[k]
            }
        }
        val marginName = otherPerson - best
        val marginHand = otherHand - best
        val absOK = best <= absThreshold
        val nameOK = marginName >= nameMargin
        val headOK = !hasHead || (if (hand == Hand.right) hbar > 0.0 else hbar < 0.0)
        val handOK = marginHand >= handMargin && headOK
        val outcome: DecisionOutcome = when {
            absOK && nameOK && handOK -> DecisionOutcome.Identified(person, hand)
            m < maxPatterns -> DecisionOutcome.NeedMorePatterns
            absOK && nameOK -> DecisionOutcome.IdentifiedHandUnclear(person)
            !absOK -> DecisionOutcome.Unknown(UnknownReason.noMatch)
            else -> DecisionOutcome.Unknown(UnknownReason.ambiguous)
        }
        return Partial(outcome, kstar, best, marginName, marginHand)
    }

    class Partial(
        val outcome: DecisionOutcome,
        val kstar: Int,
        val best: Double,
        val marginName: Double,
        val marginHand: Double,
    )

    /** Indices sorted by descending probability (NaN last, ties by index). */
    fun descendingOrder(probabilities: List<Double>): List<Int> =
        probabilities.indices.sortedWith(Comparator { a, b ->
            val pa: Double = probabilities[a]
            val pb: Double = probabilities[b]
            if (pa != pb && !(pa.isNaN() || pb.isNaN())) {
                if (pa > pb) -1 else 1
            } else if (pa.isNaN() != pb.isNaN()) {
                if (pb.isNaN()) -1 else 1
            } else {
                a.compareTo(b)
            }
        })
}

// MARK: - Identifier (SPEC 9.1, 9.3-9.5)

class Identifier(
    samples: List<TemplateSample>,
    val modelConfig: ModelConfig = ModelConfig.default,
    val config: IdentifierConfig = IdentifierConfig.default,
) {
    val prototypes: List<PrototypeKey>
    val model: GaussianModel?
    val handModel: GaussianModel?

    init {
        val current = samples.filter { it.featureVersion == FeatureExtractor.version }
        val pool = if (current.isEmpty()) samples else current
        val dimension = pool.firstOrNull()?.features?.size ?: 0
        val used = pool.filter { it.features.size == dimension }
        val protos = used.map { it.prototype }.distinct().sorted()
        prototypes = protos
        val index = HashMap<PrototypeKey, Int>()
        for ((i, p) in protos.withIndex()) index[p] = i
        if (used.isEmpty()) {
            model = null
            handModel = null
        } else {
            val x = used.map { it.features }
            val labels = used.map { index[it.prototype] ?: 0 }
            model = GaussianModel.fit(x, labels, protos.size, modelConfig)
            val hands = used.map { it.prototype.hand }
            val rightCount = hands.count { it == Hand.right }
            val leftCount = hands.size - rightCount
            handModel = if (rightCount >= 2 && leftCount >= 2) {
                GaussianModel.fit(x, hands.map { it.ordinal }, 2, modelConfig)
            } else {
                null
            }
        }
    }

    internal fun handLogOdds(x: List<Double>): Double {
        val head = handModel ?: return 0.0
        val d = head.distances(x)
        if (d.size != 2) return 0.0
        return (d[1] - d[0]) / 2.0
    }

    /** SPEC 9.4 on the given patterns (one feature vector per pattern). */
    fun decide(patterns: List<List<Double>>): Decision {
        val m = patterns.size
        val model = this.model
        if (model == null || prototypes.isEmpty()) {
            return Decision(
                outcome = DecisionOutcome.Unknown(UnknownReason.noPeople),
                patternsUsed = m,
                absThreshold = this.model?.absThreshold ?: 0.0,
            )
        }
        if (m == 0) {
            return Decision(outcome = DecisionOutcome.NeedMorePatterns, patternsUsed = 0, absThreshold = model.absThreshold)
        }
        val kCount = prototypes.size
        val perPattern = patterns.map { model.distances(it) }
        val dbar = DoubleArray(kCount)
        for (k in 0 until kCount) {
            var s = 0.0
            for (j in 0 until m) {
                val row = perPattern[j]
                s += if (k < row.size) row[k] else Double.POSITIVE_INFINITY
            }
            dbar[k] = s / m.toDouble()
        }
        val hasHead = handModel != null
        val hbar = if (hasHead) Numerics.mean(patterns.map { handLogOdds(it) }) else 0.0
        val absThreshold = model.absThreshold
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
            personScores = personScores(dbar),
            prototypeDistances = distances,
        )
    }

    /** Softmax over persons of `-D²_min / 2`. */
    internal fun personScores(dbar: DoubleArray): List<PersonScore> {
        val persons = ArrayList<String>()
        val best = ArrayList<Double>()
        for ((k, p) in prototypes.withIndex()) {
            val i = persons.indexOf(p.personID)
            if (i >= 0) {
                if (dbar[k] < best[i]) best[i] = dbar[k]
            } else {
                persons.add(p.personID)
                best.add(dbar[k])
            }
        }
        if (persons.isEmpty()) return emptyList()
        val logits = best.map { -it / 2.0 }
        var top = logits[0]
        for (v in logits.drop(1)) {
            if (v > top) top = v
        }
        val ex = logits.map { Math.exp(it - top) }
        val total = Numerics.sum(ex)
        val scores = persons.indices.map { PersonScore(persons[it], ex[it] / total, best[it]) }
        return DecisionRule.descendingOrder(scores.map { it.probability }).map { scores[it] }
    }

    /** SPEC 9.5: the patterns as adaptive samples of the best prototype, or empty if not eligible. */
    fun adaptiveSamples(decision: Decision, patterns: List<List<Double>>, now: Instant): List<TemplateSample> {
        if (model == null) return emptyList()
        val proto = decision.bestPrototype ?: return emptyList()
        if (decision.outcome !is DecisionOutcome.Identified) return emptyList()
        if (!(decision.marginName >= config.adaptiveNameFactor * config.nameMargin &&
                decision.bestDistance <= config.adaptiveAbsFactor * decision.absThreshold)
        ) {
            return emptyList()
        }
        return patterns.map { x ->
            TemplateSample(
                prototype = proto,
                source = SampleSource.adaptive,
                createdAt = now,
                features = x,
                featureVersion = FeatureExtractor.version,
            )
        }
    }

    companion object {
        /** Keeps at most `cap` adaptive samples per prototype, dropping the oldest. */
        fun applyAdaptiveCap(samples: List<TemplateSample>, cap: Int): List<TemplateSample> =
            AdaptiveCap.apply(samples, cap, { it.source == SampleSource.adaptive }, { it.prototype }, { it.createdAt })
    }
}

internal object AdaptiveCap {
    fun <T> apply(
        samples: List<T>,
        cap: Int,
        isAdaptive: (T) -> Boolean,
        key: (T) -> PrototypeKey,
        createdAt: (T) -> Instant,
    ): List<T> {
        val limit = maxOf(cap, 0)
        val groups = LinkedHashMap<PrototypeKey, MutableList<Int>>()
        for ((i, s) in samples.withIndex()) {
            if (isAdaptive(s)) groups.getOrPut(key(s)) { ArrayList() }.add(i)
        }
        val drop = HashSet<Int>()
        for (indices in groups.values) {
            if (indices.size <= limit) continue
            val ordered = indices.sortedWith(Comparator { a, b ->
                val c = createdAt(samples[a]).compareTo(createdAt(samples[b]))
                if (c != 0) c else a.compareTo(b)
            })
            for (i in ordered.take(indices.size - limit)) drop.add(i)
        }
        return samples.filterIndexed { i, _ -> i !in drop }
    }
}
