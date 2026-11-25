package com.mahmoudalyosify.auditor.core

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.Instant
import java.util.UUID

class RecordingTest {
    private fun sample(label: RecordingLabel?, device: DeviceInfo?): Recording = Recording(
        id = UUID.fromString("c6d7ffb7-827e-41d1-8fc6-6894407e02a8"),
        createdAt = Instant.parse("2026-11-24T10:00:00.750Z"),
        purpose = RecordingPurpose.enrollment,
        label = label,
        challenge = ChallengeGenerator.generate(51000uL),
        hapticStart = 81234.5612,
        accel = SensorStream(listOf(1.0, 1.01), listOf(0.1, 0.2), listOf(0.0, -0.1), listOf(-0.98, -0.97)),
        gyro = SensorStream(listOf(1.0, 1.01), listOf(0.01, 0.02), listOf(0.0, 0.0), listOf(1e-5, 2e-5)),
        device = device,
    )

    @Test
    fun jsonMatchesTheIOSExportShape() {
        val label = RecordingLabel("7A1C", "Amira", Hand.left, Accessory.watch, Posture.freshGrip, "S1")
        val rec = sample(label, DeviceInfo("Pixel 8", "Android 15", "0.1.0 (1)"))
        val text = RecordingCoding.encode(rec)
        assertFalse(text.contains('\n'))
        val tree = Json.parseToJsonElement(text) as JsonObject
        assertEquals(
            listOf("accel", "challenge", "createdAt", "device", "gyro", "hapticStart", "id", "label", "motion", "purpose", "schema"),
            tree.keys.toList(),
        )
        assertEquals("2026-11-24T10:00:00Z", tree.at("createdAt").str())
        assertEquals("C6D7FFB7-827E-41D1-8FC6-6894407E02A8", tree.at("id").str())
        assertEquals(Recording.currentSchema, tree.at("schema").str())
        assertEquals("enrollment", tree.at("purpose").str())
        assertEquals("freshGrip", tree.at("label").at("posture").str())
        assertEquals(SeedFormat.string(51000uL), tree.at("challenge").at("seed").str())
        assertEquals(14, (tree.at("motion") as JsonObject).size)
        val back = RecordingCoding.decodeRecording(text)
        assertEquals(rec.copy(createdAt = Instant.parse("2026-11-24T10:00:00Z")), back)
    }

    @Test
    fun optionalKeysAreLeftOutAndJsonLinesRoundTrip() {
        val bare = sample(null, null)
        val tree = Json.parseToJsonElement(RecordingCoding.encode(bare)) as JsonObject
        assertFalse(tree.containsKey("label"))
        assertFalse(tree.containsKey("device"))
        val withLabel = sample(RecordingLabel("p", "P", Hand.right, sessionID = "S2"), null)
        val lines = RecordingCoding.jsonLines(listOf(bare, withLabel))
        assertTrue(lines.endsWith("\n"))
        assertEquals(2, lines.count { it == '\n' })
        val parsed = RecordingCoding.parseJSONLines("﻿" + lines.replace("\n", "\r\n") + "\n  \n")
        assertEquals(2, parsed.size)
        assertEquals(withLabel.label, parsed[1].label)
        assertEquals(bare.accel, parsed[0].accel)
    }

    @Test
    fun nonFiniteNumbersUseStrings() {
        val rec = sample(null, null).copy(hapticStart = Double.POSITIVE_INFINITY)
        val text = RecordingCoding.encode(rec)
        assertTrue(text.contains("\"hapticStart\":\"Infinity\""))
        assertEquals(Double.POSITIVE_INFINITY, RecordingCoding.decodeRecording(text).hapticStart, 0.0)
    }

    @Test
    fun fixtureRecordingsRoundTrip() {
        val case = Fixtures.json("pipeline_vectors").at("cases").arr().first()
        val rec = RecordingCoding.recordingFromJson(case.at("recording"))
        assertEquals(rec, RecordingCoding.decodeRecording(RecordingCoding.encode(rec)))
    }

    @Test
    fun samplesRoundTrip() {
        val key = PrototypeKey("p", Hand.left, Accessory.ring)
        val t = listOf(TemplateSample(prototype = key, source = SampleSource.enrollment, createdAt = FIXED_DATE, features = listOf(1.0, -2.5)))
        assertEquals(t, RecordingCoding.decodeTemplateSamples(RecordingCoding.encodeTemplateSamples(t)))
        val e = listOf(EmbeddingSample(key, listOf(0.6, 0.8), SampleSource.adaptive, FIXED_DATE))
        assertEquals(e, RecordingCoding.decodeEmbeddingSamples(RecordingCoding.encodeEmbeddingSamples(e)))
    }
}
