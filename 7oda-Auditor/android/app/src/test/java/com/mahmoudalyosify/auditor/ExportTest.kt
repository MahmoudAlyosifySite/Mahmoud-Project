package com.mahmoudalyosify.auditor

import com.mahmoudalyosify.auditor.core.Accessory
import com.mahmoudalyosify.auditor.core.ChallengeGenerator
import com.mahmoudalyosify.auditor.core.EmbeddingSample
import com.mahmoudalyosify.auditor.core.Hand
import com.mahmoudalyosify.auditor.core.Posture
import com.mahmoudalyosify.auditor.core.PrototypeKey
import com.mahmoudalyosify.auditor.core.Recording
import com.mahmoudalyosify.auditor.core.RecordingCoding
import com.mahmoudalyosify.auditor.core.RecordingLabel
import com.mahmoudalyosify.auditor.core.RecordingPurpose
import com.mahmoudalyosify.auditor.core.SampleSource
import com.mahmoudalyosify.auditor.core.SensorStream
import com.mahmoudalyosify.auditor.model.AppJson
import com.mahmoudalyosify.auditor.model.AppSettings
import com.mahmoudalyosify.auditor.model.CheckLogEntry
import com.mahmoudalyosify.auditor.model.Exporter
import com.mahmoudalyosify.auditor.model.Person
import com.mahmoudalyosify.auditor.model.RecognizerKind
import com.mahmoudalyosify.auditor.model.StoredEmbedding
import com.mahmoudalyosify.auditor.model.applyAdaptiveCap
import com.mahmoudalyosify.auditor.storage.AesGcmSealer
import com.mahmoudalyosify.auditor.storage.EncryptedStore
import kotlinx.serialization.builtins.ListSerializer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.file.Files
import java.time.Instant
import java.util.UUID
import javax.crypto.KeyGenerator

class ExportTest {
    private val ana = Person(id = "A", name = "Ana", createdAt = Instant.parse("2026-01-01T10:00:00Z"))
    private val bob = Person(id = "B", name = "Bob, Jr.", createdAt = Instant.parse("2026-01-02T10:00:00Z"))

    private fun recording(personID: String?, n: Int = 5): Recording {
        val t = List(n) { 100.0 + it * 0.01 }
        val stream = SensorStream(t, List(n) { 0.01 * it }, List(n) { 0.0 }, List(n) { -1.0 })
        val label = personID?.let {
            RecordingLabel(personID = it, personName = "secret", hand = Hand.left, accessory = Accessory.none,
                posture = Posture.sitting, sessionID = "S")
        }
        return Recording(purpose = RecordingPurpose.enrollment, label = label, challenge = ChallengeGenerator.generate(7UL),
            hapticStart = 100.02, accel = stream, gyro = stream, createdAt = Instant.parse("2026-01-03T10:00:00Z"))
    }

    @Test
    fun anonymizedNamesFollowEnrollmentOrder() {
        val names = Exporter.nameMap(listOf(bob, ana), anonymize = true)
        assertEquals("P01", names["A"])
        assertEquals("P02", names["B"])
        assertEquals("Bob, Jr.", Exporter.nameMap(listOf(bob, ana), anonymize = false)["B"])
    }

    @Test
    fun jsonLinesRoundTripWithAnonymizedLabels() {
        val names = Exporter.nameMap(listOf(ana, bob), anonymize = true)
        val prepared = Exporter.prepare(listOf(recording("A"), recording(null), recording("Z")), names)
        val text = RecordingCoding.jsonLines(prepared)
        assertEquals(3, text.trimEnd('\n').split('\n').size)
        assertFalse(text.contains("secret"))
        val back = RecordingCoding.parseJSONLines(text)
        assertEquals(prepared, back)
        assertEquals("P01", back[0].label?.personName)
        assertNull(back[1].label)
        assertEquals("P??", back[2].label?.personName)
    }

    @Test
    fun csvQuotingAndNumbers() {
        assertEquals("\"Bob, Jr.\"", Exporter.csvField("Bob, Jr."))
        assertEquals("\"say \"\"hi\"\"\"", Exporter.csvField("say \"hi\""))
        assertEquals("0.5", Exporter.number(0.5))
        assertEquals("1.23457e+06", Exporter.number(1234567.0))
        assertEquals("inf", Exporter.number(Double.POSITIVE_INFINITY))
        assertEquals("", Exporter.number(null))
    }

    @Test
    fun checksCSVHasOneRowPerCheck() {
        val entry = CheckLogEntry(
            id = UUID.randomUUID(), createdAt = Instant.parse("2026-01-03T10:00:00Z"), purpose = RecordingPurpose.test,
            recognizer = RecognizerKind.v1, recordingIDs = listOf(UUID.randomUUID()), patternsUsed = 1, attempts = 2,
            seconds = 3.5, truthPersonID = "B", truthHand = Hand.right, challengePassed = true,
        )
        val csv = Exporter.checksCSV(listOf(entry), mapOf("B" to "Bob, Jr."))
        val lines = csv.trimEnd('\n').split('\n')
        assertEquals(2, lines.size)
        assertTrue(lines[1].contains("\"Bob, Jr.\""))
        assertTrue(lines[1].contains(",test,v1,"))
        // Logged entries also survive the app's JSON.
        val json = AppJson.encodeToString(ListSerializer(CheckLogEntry.serializer()), listOf(entry))
        assertEquals(listOf(entry), AppJson.decodeFromString(ListSerializer(CheckLogEntry.serializer()), json))
    }

    @Test
    fun settingsDecodeWithMissingKeys() {
        val s = AppJson.decodeFromString(AppSettings.serializer(), """{"recognizer":"v0","maxPatterns":5,"future":1}""")
        assertEquals(RecognizerKind.v0, s.recognizer)
        assertEquals(5, s.maxPatterns)
        assertEquals(AppSettings().absFactor, s.absFactor, 0.0)
    }

    @Test
    fun storedEmbeddingsRoundTripAndAdaptiveCap() {
        val key = PrototypeKey("A", Hand.right, Accessory.none)
        fun e(source: SampleSource, second: Long) = StoredEmbedding(
            recordingID = UUID.randomUUID(), encoderVersion = "test",
            sample = EmbeddingSample(key, List(4) { it * 0.5 }, source, Instant.ofEpochSecond(second)),
        )
        val items = listOf(e(SampleSource.enrollment, 1), e(SampleSource.adaptive, 5), e(SampleSource.adaptive, 3),
            e(SampleSource.adaptive, 9))
        val json = AppJson.encodeToString(ListSerializer(StoredEmbedding.serializer()), items)
        assertEquals(items, AppJson.decodeFromString(ListSerializer(StoredEmbedding.serializer()), json))
        val capped = applyAdaptiveCap(items, 2)
        assertEquals(3, capped.size)
        assertEquals(listOf(1L, 5L, 9L), capped.map { it.sample.createdAt.epochSecond })
    }

    @Test
    fun encryptedStoreRoundTrip() {
        val key = KeyGenerator.getInstance("AES").apply { init(256) }.generateKey()
        val dir = Files.createTempDirectory("auditor-store").toFile()
        try {
            val store = EncryptedStore(dir, AesGcmSealer { key })
            store.writeText(EncryptedStore.peopleFile, "[\"hello\"]")
            assertEquals("[\"hello\"]", store.readText(EncryptedStore.peopleFile))
            assertNotEquals("[\"hello\"]", String(dir.resolve(EncryptedStore.peopleFile).readBytes()))
            val r = recording("A")
            store.writeRecording(r)
            assertEquals(r, store.readRecording(r.id))
            assertEquals(listOf(r.id), store.recordingIDs())
            store.deleteAll()
            assertNull(store.readText(EncryptedStore.peopleFile))
            assertTrue(store.recordingIDs().isEmpty())
        } finally {
            dir.deleteRecursively()
        }
    }
}
