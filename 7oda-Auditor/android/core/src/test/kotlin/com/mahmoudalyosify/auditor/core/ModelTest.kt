package com.mahmoudalyosify.auditor.core

import kotlinx.serialization.json.JsonElement
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelTest {
    private val tolerance = 1e-6

    private fun modelConfig(e: JsonElement): ModelConfig {
        val d = ModelConfig.default
        return ModelConfig(
            shrinkageMin = e.opt("shrinkageMin")?.dbl() ?: d.shrinkageMin,
            looQuantile = e.opt("looQuantile")?.dbl() ?: d.looQuantile,
            absFactor = e.opt("absFactor")?.dbl() ?: d.absFactor,
        )
    }

    private fun identifierConfig(e: JsonElement): IdentifierConfig {
        val d = IdentifierConfig.default
        return IdentifierConfig(
            nameMargin = e.opt("nameMargin")?.dbl() ?: d.nameMargin,
            handMargin = e.opt("handMargin")?.dbl() ?: d.handMargin,
            maxPatterns = e.opt("maxPatterns")?.int() ?: d.maxPatterns,
            adaptiveMaxPerPrototype = e.opt("adaptiveMaxPerPrototype")?.int() ?: d.adaptiveMaxPerPrototype,
            adaptiveNameFactor = e.opt("adaptiveNameFactor")?.dbl() ?: d.adaptiveNameFactor,
            adaptiveAbsFactor = e.opt("adaptiveAbsFactor")?.dbl() ?: d.adaptiveAbsFactor,
        )
    }

    @Test
    fun modelFixtureParity() {
        val root = Fixtures.json("model_vectors")
        val persons = root.at("samples").arr().map { it.at("personID").str() }.toSet()
        assertTrue("need at least 3 persons", persons.size >= 3)
        check(root, "main")
    }

    @Test
    fun singlePersonFixtureParity() {
        val block = Fixtures.json("model_vectors").opt("singlePerson")
        assertNotNull("model_vectors.json has no singlePerson block", block)
        check(block!!, "singlePerson")
    }

    private fun check(block: JsonElement, label: String) = checks { c ->
        val samples = block.at("samples").arr().map { s ->
            TemplateSample(
                prototype = PrototypeKey(
                    s.at("personID").str(),
                    Hand.from(s.at("hand").str())!!,
                    Accessory.from(s.at("accessory").str())!!,
                ),
                source = SampleSource.from(s.at("source").str())!!,
                createdAt = FIXED_DATE,
                features = s.at("features").dbls(),
            )
        }
        val identifier = Identifier(samples, modelConfig(block.at("config")), identifierConfig(block.at("identifierConfig")))
        val e = block.at("expected")
        c.equal(identifier.prototypes, e.at("prototypes").arr().map { it.prototypeKey() }, "$label prototypes")
        val model = identifier.model
        if (model == null) {
            c.fail("$label: model was not trained")
            return@checks
        }
        c.close(model.mean, e.at("mean").dbls(), tolerance, "$label mean")
        c.close(model.scale, e.at("scale").dbls(), tolerance, "$label scale")
        c.close(model.shrinkage, e.at("shrinkage").dbl(), tolerance, "$label shrinkage")
        c.close(model.absThreshold, e.at("absThreshold").dbl(), tolerance, "$label absThreshold")
        e.opt("mu")?.let { c.close(model.mu, it.dbl(), tolerance, "$label mu") }
        e.opt("shrinkageLW")?.let { c.close(model.shrinkageLW, it.dbl(), tolerance, "$label shrinkageLW") }
        e.opt("classCounts")?.let { c.equal(model.classCounts, it.ints(), "$label classCounts") }
        e.opt("classMeans")?.let { means ->
            val expected = means.matrix()
            c.equal(model.classMeans.size, expected.size, "$label classMeans count")
            for ((k, pair) in model.classMeans.zip(expected).withIndex()) {
                c.close(pair.first, pair.second, tolerance, "$label classMeans[$k]")
            }
        }
        e.opt("looDistances")?.let { c.close(model.looDistances, it.dbls(), tolerance, "$label looDistances") }
        e.opt("handShrinkage")?.let { hs ->
            val head = identifier.handModel
            if (head == null) {
                c.fail("$label: hand model expected")
            } else {
                c.close(head.shrinkage, hs.dbl(), tolerance, "$label handShrinkage")
                e.opt("handAbsThreshold")?.let { c.close(head.absThreshold, it.dbl(), tolerance, "$label handAbsThreshold") }
            }
        }

        val queries = block.at("queries").arr()
        c.check(queries.isNotEmpty()) { "$label has no queries" }
        for ((qi, q) in queries.withIndex()) {
            val name = "$label query ${q.opt("name")?.str() ?: qi.toString()}"
            val patterns = q.at("patterns").matrix()
            val distances = q.at("distances").matrix()
            val handLogOdds = q.at("handLogOdds").dbls()
            c.equal(distances.size, patterns.size, "$name distances per pattern")
            for ((j, pattern) in patterns.withIndex()) {
                if (j < distances.size) c.close(model.distances(pattern), distances[j], tolerance, "$name distances[$j]")
                if (j < handLogOdds.size) {
                    var h = 0.0
                    val head = identifier.handModel
                    if (head != null) {
                        val d = head.distances(pattern)
                        h = (d[1] - d[0]) / 2.0
                    }
                    c.close(h, handLogOdds[j], tolerance, "$name handLogOdds[$j]")
                }
            }
            val decision = identifier.decide(patterns)
            c.equal(decision.outcome, q.at("outcome").outcome(), "$name outcome")
            c.equal(decision.patternsUsed, patterns.size, "$name patternsUsed")
            c.close(decision.bestDistance, q.opt("bestDistance")?.dbl() ?: Double.POSITIVE_INFINITY, tolerance, "$name bestDistance")
            c.close(decision.marginName, q.opt("marginName")?.dbl() ?: Double.POSITIVE_INFINITY, tolerance, "$name marginName")
            c.close(decision.marginHand, q.opt("marginHand")?.dbl() ?: Double.POSITIVE_INFINITY, tolerance, "$name marginHand")
            q.opt("meanDistances")?.let {
                c.close(decision.prototypeDistances.map { d -> d.distance }, it.dbls(), tolerance, "$name meanDistances")
                c.equal(decision.prototypeDistances.map { d -> d.prototype }, identifier.prototypes, "$name prototype order")
            }
            q.opt("meanHandLogOdds")?.let { c.close(decision.handLogOdds, it.dbl(), tolerance, "$name handLogOdds") }
            q.opt("bestPrototype")?.let { c.equal(decision.bestPrototype, it.prototypeKey(), "$name bestPrototype") }
            q.opt("personScores")?.let { scores ->
                val expected = scores.arr()
                c.equal(decision.personScores.map { it.personID }, expected.map { it.at("personID").str() }, "$name personScores order")
                for ((s, pair) in decision.personScores.zip(expected).withIndex()) {
                    c.close(pair.first.probability, pair.second.at("probability").dbl(), tolerance, "$name personScores[$s].probability")
                    c.close(pair.first.bestDistance, pair.second.at("bestDistance").dbl(), tolerance, "$name personScores[$s].bestDistance")
                }
            }
            q.opt("adaptiveEligible")?.let {
                val eligible = it.bool()
                val adaptive = identifier.adaptiveSamples(decision, patterns, FIXED_DATE)
                c.equal(adaptive.isNotEmpty(), eligible, "$name adaptiveEligible")
                if (eligible) {
                    c.equal(adaptive.size, patterns.size, "$name adaptive count")
                    c.check(adaptive.all { a -> a.source == SampleSource.adaptive && a.prototype == decision.bestPrototype }) {
                        "$name adaptive samples"
                    }
                }
            }
        }
    }

    @Test
    fun fitRejectsEmptyAndMalformedInput() {
        val config = ModelConfig.default
        assertNull(GaussianModel.fit(emptyList(), emptyList(), 1, config))
        assertNull(GaussianModel.fit(listOf(listOf(1.0, 2.0), listOf(1.0)), listOf(0, 0), 1, config))
        assertNull(GaussianModel.fit(listOf(listOf(1.0, 2.0)), listOf(1), 1, config))
        assertNull(GaussianModel.fit(listOf(listOf(1.0, 2.0)), listOf(0, 0), 1, config))
    }

    @Test
    fun singleSampleIsDegenerateButUsable() {
        val m = GaussianModel.fit(listOf(listOf(1.0, 2.0, 3.0)), listOf(0), 1, ModelConfig.default)
        assertNotNull(m)
        m!!
        assertTrue(m.degenerate)
        assertEquals(1.0, m.shrinkage, 0.0)
        assertEquals(emptyList<Double>(), m.looDistances)
        assertEquals(1.5 * 2.0 * 3.0, m.absThreshold, 0.0)
        assertEquals(listOf(1e-6, 1e-6, 1e-6), m.scale)
    }

    @Test
    fun noPeopleAndAdaptiveCap() {
        val empty = Identifier(emptyList())
        assertEquals(DecisionOutcome.Unknown(UnknownReason.noPeople), empty.decide(listOf(listOf(1.0))).outcome)
        val key = PrototypeKey("p", Hand.right)
        val samples = (0 until 5).map { i ->
            TemplateSample(prototype = key, source = SampleSource.adaptive, createdAt = FIXED_DATE.plusSeconds(i.toLong()), features = listOf(i.toDouble()))
        } + TemplateSample(prototype = key, source = SampleSource.enrollment, createdAt = FIXED_DATE, features = listOf(9.0))
        val kept = Identifier.applyAdaptiveCap(samples, 2)
        assertEquals(listOf(3.0, 4.0, 9.0), kept.map { it.features[0] })
    }

    @Test
    fun prototypeOrder() {
        val keys = listOf(
            PrototypeKey("b", Hand.left), PrototypeKey("a", Hand.left, Accessory.watch),
            PrototypeKey("a", Hand.left), PrototypeKey("a", Hand.right),
        )
        assertEquals(
            listOf(PrototypeKey("a", Hand.right), PrototypeKey("a", Hand.left), PrototypeKey("a", Hand.left, Accessory.watch), PrototypeKey("b", Hand.left)),
            keys.sorted(),
        )
    }
}
