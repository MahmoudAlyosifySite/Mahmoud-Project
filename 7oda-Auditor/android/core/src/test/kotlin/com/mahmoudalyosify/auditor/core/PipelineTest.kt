package com.mahmoudalyosify.auditor.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PipelineTest {
    private val tolerance = 1e-6

    private fun metric(m: QualityMetrics, key: String): Double? = when (key) {
        "gapFraction" -> m.gapFraction
        "sampleRate" -> m.sampleRate
        "quietGyroRMS" -> m.quietGyroRMS
        "quietAccelRMS" -> m.quietAccelRMS
        "gravityZ" -> m.gravityZ
        "motionRMS" -> m.motionRMS
        "gravityAngleDeg" -> m.gravityAngleDeg
        "ncc" -> m.ncc
        "peakMargin" -> m.peakMargin
        "snr" -> m.snr
        "lag" -> m.lag.toDouble()
        else -> null
    }

    @Test
    fun pipelineFixtureParity() = checks { c ->
        val cases = Fixtures.json("pipeline_vectors").at("cases").arr()
        c.check(cases.size >= 8) { "expected at least 8 pipeline cases, found ${cases.size}" }
        for (case in cases) {
            val name = case.at("name").str()
            val e = case.at("expected")
            val mode = QualityMode.from(case.at("mode").str())
            if (mode == null) {
                c.fail("$name: unknown mode")
                continue
            }
            val recording = RecordingCoding.recordingFromJson(case.at("recording"))
            val calibration = case.opt("calibration")?.let { RecordingCoding.calibrationFromJson(it) }
            val p = SignalProcessor.process(recording)
            c.equal(p.n, e.at("n").int(), "$name n")
            c.equal(p.alignment.lag, e.at("lag").int(), "$name lag")
            c.close(p.alignment.ncc, e.at("ncc").dbl(), tolerance, "$name ncc")
            c.close(p.alignment.peakMargin, e.at("peakMargin").dbl(), tolerance, "$name peakMargin")
            c.close(p.noiseAccel, e.at("noiseAccel").dbl(), tolerance, "$name noiseAccel")
            c.close(p.noiseGyro, e.at("noiseGyro").dbl(), tolerance, "$name noiseGyro")
            c.close(p.gapFraction, e.at("gapFraction").dbl(), tolerance, "$name gapFraction")
            c.close(p.sampleRate, e.at("sampleRate").dbl(), tolerance, "$name sampleRate")
            c.equal(listOf(p.quietRange.first, p.quietRange.last + 1), e.at("quietRange").ints(), "$name quietRange")
            c.equal(listOf(p.eventRange.first, p.eventRange.last + 1), e.at("eventRange").ints(), "$name eventRange")

            val report = QualityGate.evaluate(p, calibration, mode)
            val metrics = e.at("metrics")
            for (key in (metrics as kotlinx.serialization.json.JsonObject).keys.sorted()) {
                val actual = metric(report.metrics, key)
                if (actual == null) {
                    c.fail("$name: unknown metric $key")
                    continue
                }
                val expected = metrics.at(key).dbl()
                if (key == "lag") c.equal(actual, expected, "$name metrics.lag") else c.close(actual, expected, tolerance, "$name metrics.$key")
            }
            val issues = e.at("issues").arr().map { it.str() }
            c.equal(report.issues.map { it.name }, issues, "$name issues")
            c.equal(report.passed, issues.isEmpty(), "$name passed")
            e.opt("passed")?.let { c.equal(report.passed, it.bool(), "$name passed flag") }

            val absolute = FeatureExtractor.extractAbsolute(p)
            c.close(absolute, e.at("featuresAbsolute").dbls(), tolerance, "$name featuresAbsolute")
            c.close(FeatureExtractor.extract(p, calibration), e.at("features").dbls(), tolerance, "$name features")
            c.check(absolute.all { it.isFinite() }) { "$name features must be finite" }
        }
    }

    @Test
    fun calibrationBuild() {
        val cases = Fixtures.json("pipeline_vectors").at("cases").arr()
        val table = cases.firstOrNull { it.at("expected").at("issues").arr().isEmpty() && it.at("mode").str() == "onTable" }
        assertNotNull("pipeline_vectors.json has no passing onTable case", table)
        val p = SignalProcessor.process(RecordingCoding.recordingFromJson(table!!.at("recording")))
        val report = QualityGate.evaluate(p, null, QualityMode.onTable)
        assertTrue(report.passed)
        val failed = report.copy(issues = listOf(QualityIssue.notResting), passed = false)
        assertNull(Calibration.build(List(11) { p }, List(11) { report }, "X", FIXED_DATE))
        val reports = List(12) { report } + List(8) { failed }
        val calibration = Calibration.build(List(20) { p }, reports, "iPhone16,1", FIXED_DATE)
        assertNotNull(calibration)
        calibration!!
        assertEquals(12, calibration.patternsUsed)
        assertEquals(FeatureExtractor.names.size, calibration.tableFeatures.size)
        assertEquals(report.metrics.quietGyroRMS, calibration.quietGyroRMS, 0.0)
        val text = RecordingCoding.encode(calibration)
        assertEquals(calibration, RecordingCoding.decodeCalibration(text))
    }

    @Test
    fun featureNames() {
        assertEquals(39, FeatureExtractor.names.size)
        assertEquals(FeatureExtractor.names.size, FeatureExtractor.names.toSet().size)
    }
}
