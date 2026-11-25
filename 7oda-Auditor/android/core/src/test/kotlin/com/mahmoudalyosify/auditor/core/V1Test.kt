package com.mahmoudalyosify.auditor.core

import kotlinx.serialization.json.JsonElement
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class V1Test {
    private val tolerance = 1e-6

    private fun embeddingConfig(e: JsonElement): EmbeddingConfig {
        val d = EmbeddingConfig.default
        return EmbeddingConfig(
            looQuantile = e.opt("looQuantile")?.dbl() ?: d.looQuantile,
            slack = e.opt("slack")?.dbl() ?: d.slack,
            nameMargin = e.opt("nameMargin")?.dbl() ?: d.nameMargin,
            handMargin = e.opt("handMargin")?.dbl() ?: d.handMargin,
            handShrinkage = e.opt("handShrinkage")?.dbl() ?: d.handShrinkage,
            maxPatterns = e.opt("maxPatterns")?.int() ?: d.maxPatterns,
            adaptiveMaxPerPrototype = e.opt("adaptiveMaxPerPrototype")?.int() ?: d.adaptiveMaxPerPrototype,
        )
    }

    @Test
    fun encoderInputFixtureParity() = checks { c ->
        val block = Fixtures.json("v1_vectors").at("tensor")
        c.equal(block.at("channels").int(), EncoderInput.channels, "channels")
        c.equal(block.at("length").int(), EncoderInput.length, "length")
        c.equal(block.at("patternStart").int(), EncoderInput.patternStart, "patternStart")
        val cases = block.at("cases").arr()
        c.check(cases.isNotEmpty()) { "no tensor cases" }
        for (case in cases) {
            val name = case.at("name").str()
            val p = SignalProcessor.process(RecordingCoding.recordingFromJson(case.at("recording")))
            c.equal(p.n, case.at("n").int(), "$name n")
            c.equal(p.alignment.lag, case.at("lag").int(), "$name lag")
            val x = EncoderInput.build(p)
            c.equal(x.size, EncoderInput.channels * EncoderInput.length, "$name tensor size")
            c.close(x.map { it.toDouble() }, case.at("tensor").dbls(), 1e-5, "$name tensor")
        }
    }

    @Test
    fun embeddingIdentifierFixtureParity() = checks { c ->
        val block = Fixtures.json("v1_vectors").at("identifier")
        val samples = block.at("samples").arr().map { s ->
            EmbeddingSample(
                prototype = PrototypeKey(
                    s.at("personID").str(),
                    Hand.from(s.at("hand").str())!!,
                    Accessory.from(s.at("accessory").str())!!,
                ),
                embedding = s.at("embedding").dbls(),
                source = SampleSource.from(s.at("source").str())!!,
                createdAt = FIXED_DATE,
            )
        }
        val identifier = EmbeddingIdentifier(samples, embeddingConfig(block.at("config")))
        val e = block.at("expected")
        c.equal(identifier.dimension, block.at("dimension").int(), "dimension")
        c.equal(identifier.prototypes, e.at("prototypes").arr().map { it.prototypeKey() }, "prototypes")
        val vectors = e.at("prototypeVectors").matrix()
        c.equal(identifier.prototypeVectors.size, vectors.size, "prototypeVectors count")
        for ((k, pair) in identifier.prototypeVectors.zip(vectors).withIndex()) {
            c.close(pair.first, pair.second, tolerance, "prototypeVectors[$k]")
        }
        c.close(identifier.looSimilarities, e.at("loo").dbls(), tolerance, "loo")
        c.close(identifier.simThreshold, e.at("simThreshold").dbl(), tolerance, "simThreshold")
        c.close(identifier.absThreshold, e.at("absThreshold").dbl(), tolerance, "absThreshold")
        val w = e.opt("handWeights")?.dbls()
        val actualW = identifier.handWeights
        if (w != null) {
            if (actualW == null) {
                c.fail("hand model expected")
            } else {
                c.close(actualW, w, tolerance, "handWeights")
                c.close(identifier.handBias, e.opt("handBias")?.dbl() ?: 0.0, tolerance, "handBias")
            }
        } else {
            c.check(actualW == null) { "no hand model expected" }
        }

        val queries = block.at("queries").arr()
        c.check(queries.isNotEmpty()) { "no identifier queries" }
        for (q in queries) {
            val name = "query ${q.at("name").str()}"
            val patterns = q.at("patterns").matrix()
            val similarities = q.at("similarities").matrix()
            val handLogOdds = q.at("handLogOdds").dbls()
            for ((j, pattern) in patterns.withIndex()) {
                if (j < similarities.size) c.close(identifier.similarities(pattern), similarities[j], tolerance, "$name similarities[$j]")
                if (j < handLogOdds.size) c.close(identifier.handLogOdds(pattern), handLogOdds[j], tolerance, "$name handLogOdds[$j]")
            }
            val decision = identifier.decide(patterns)
            c.equal(decision.outcome, q.at("outcome").outcome(), "$name outcome")
            c.equal(decision.patternsUsed, patterns.size, "$name patternsUsed")
            c.close(decision.bestDistance, q.opt("bestDistance")?.dbl() ?: Double.POSITIVE_INFINITY, tolerance, "$name bestDistance")
            c.close(decision.marginName, q.opt("marginName")?.dbl() ?: Double.POSITIVE_INFINITY, tolerance, "$name marginName")
            c.close(decision.marginHand, q.opt("marginHand")?.dbl() ?: Double.POSITIVE_INFINITY, tolerance, "$name marginHand")
            c.close(decision.prototypeDistances.map { it.distance }, q.at("meanDistances").dbls(), tolerance, "$name meanDistances")
            c.close(decision.handLogOdds, q.at("meanHandLogOdds").dbl(), tolerance, "$name handLogOdds")
            c.close(decision.absThreshold, e.at("absThreshold").dbl(), tolerance, "$name absThreshold")
            c.equal(decision.bestPrototype, q.at("bestPrototype").prototypeKey(), "$name bestPrototype")
            val scores = q.at("personScores").arr()
            c.equal(decision.personScores.map { it.personID }, scores.map { it.at("personID").str() }, "$name personScores order")
            for ((s, pair) in decision.personScores.zip(scores).withIndex()) {
                c.close(pair.first.probability, pair.second.at("probability").dbl(), tolerance, "$name personScores[$s].probability")
                c.close(pair.first.bestDistance, pair.second.at("bestDistance").dbl(), tolerance, "$name personScores[$s].bestDistance")
            }
            val adaptive = identifier.adaptiveSamples(decision, patterns, FIXED_DATE)
            c.equal(adaptive.isNotEmpty(), q.at("adaptiveEligible").bool(), "$name adaptiveEligible")
        }
    }

    @Test
    fun encoderInputOfEmptySignalIsZero() {
        val spec = ChallengeGenerator.generate(5uL)
        val rec = Recording(
            purpose = RecordingPurpose.check, challenge = spec, hapticStart = 10.0,
            accel = SensorStream.empty, gyro = SensorStream.empty,
        )
        val x = EncoderInput.build(SignalProcessor.process(rec))
        assertEquals(2560, x.size)
        assertTrue(x.all { it.isFinite() && it >= -50f && it <= 50f })
    }

    @Test
    fun embeddingNoPeopleAndCap() {
        val empty = EmbeddingIdentifier(emptyList())
        assertEquals(DecisionOutcome.Unknown(UnknownReason.noPeople), empty.decide(listOf(listOf(1.0, 0.0))).outcome)
        val key = PrototypeKey("p", Hand.left)
        val samples = (0 until 4).map { i ->
            EmbeddingSample(key, listOf(1.0, i.toDouble()), SampleSource.adaptive, FIXED_DATE.plusSeconds(10L - i))
        }
        val kept = EmbeddingIdentifier.applyAdaptiveCap(samples, 1)
        assertEquals(listOf(0.0), kept.map { it.embedding[1] })
    }
}
