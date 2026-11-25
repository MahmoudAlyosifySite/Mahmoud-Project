package com.mahmoudalyosify.auditor.model

import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.mahmoudalyosify.auditor.core.Accessory
import com.mahmoudalyosify.auditor.core.Calibration
import com.mahmoudalyosify.auditor.core.ConfusionMatrix
import com.mahmoudalyosify.auditor.core.Decision
import com.mahmoudalyosify.auditor.core.DecisionOutcome
import com.mahmoudalyosify.auditor.core.EmbeddingIdentifier
import com.mahmoudalyosify.auditor.core.EmbeddingSample
import com.mahmoudalyosify.auditor.core.FeatureExtractor
import com.mahmoudalyosify.auditor.core.Hand
import com.mahmoudalyosify.auditor.core.Identifier
import com.mahmoudalyosify.auditor.core.PrototypeKey
import com.mahmoudalyosify.auditor.core.QualityConfig
import com.mahmoudalyosify.auditor.core.QualityMode
import com.mahmoudalyosify.auditor.core.QualityReport
import com.mahmoudalyosify.auditor.core.Recording
import com.mahmoudalyosify.auditor.core.RecordingCoding
import com.mahmoudalyosify.auditor.core.RecordingLabel
import com.mahmoudalyosify.auditor.core.RecordingPurpose
import com.mahmoudalyosify.auditor.core.SampleSource
import com.mahmoudalyosify.auditor.core.SignalProcessor
import com.mahmoudalyosify.auditor.core.TemplateSample
import com.mahmoudalyosify.auditor.core.UnknownReason
import com.mahmoudalyosify.auditor.core.Xoshiro256StarStar
import com.mahmoudalyosify.auditor.device.CheckRunner
import com.mahmoudalyosify.auditor.device.DeviceSupport
import com.mahmoudalyosify.auditor.device.HapticPlayer
import com.mahmoudalyosify.auditor.device.MotionRecorder
import com.mahmoudalyosify.auditor.device.PatternRecording
import com.mahmoudalyosify.auditor.ml.EmbeddingEncoder
import com.mahmoudalyosify.auditor.storage.EncryptedStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlinx.serialization.builtins.ListSerializer
import java.io.File
import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import java.util.UUID
import kotlin.coroutines.coroutineContext

/** The result of one "Who's holding?" check (or one Lab test check). */
class CheckResult(
    val id: UUID = UUID.randomUUID(),
    val purpose: RecordingPurpose,
    /** The recognizer whose decision is shown. */
    val answeredBy: RecognizerKind,
    /** Final decision of `answeredBy`; null when no clean pattern could be recorded. */
    val decision: Decision?,
    val v0: Decision?,
    val v1: Decision?,
    /** Patterns that passed the quality gate and were used. */
    val patterns: List<AnalyzedPattern>,
    val attempts: Int,
    val seconds: Double,
    val requestedHand: Hand?,
    val qualityFailure: String?,
    val notes: List<String>,
    var learned: Int = 0,
) {
    /** Hand challenge (SPEC 9.6): passes only for identified(p, requested hand). */
    val challengePassed: Boolean?
        get() {
            val requested = requestedHand ?: return null
            val outcome = decision?.outcome
            return outcome is DecisionOutcome.Identified && outcome.hand == requested
        }

    fun decision(kind: RecognizerKind): Decision? = if (kind == RecognizerKind.v0) v0 else v1
}

/** Result of the Lab signal check. */
class SignalCheckResult(
    val patterns: List<AnalyzedPattern>,
    /** Mean SNR per channel over the patterns (accel HP x/y/z, gyro HP x/y/z). */
    val channels: List<Pair<String, Double>>,
    val visible: Boolean,
)

/** Sample counts of one prototype (v0 template samples and v1 embeddings). */
data class PrototypeCount(val key: PrototypeKey, val v0: Int, val v1: Int)

/**
 * App state: people, samples of both recognizers, calibration, settings and the flows.
 * Lives on the main thread; Compose observes the state properties.
 */
class AppModel(
    val store: EncryptedStore?,
    val encoder: PatternEncoder?,
    encoderError: String?,
    val runner: PatternRecording,
    val motion: MotionRecorder?,
    val haptics: HapticPlayer?,
    private val cacheDir: File?,
) {
    companion object {
        const val maxRetriesPerSlot = 3

        /** The real app: Keystore-keyed store, bundled model, device sensors and vibrator. */
        fun live(context: Context): AppModel {
            val app = context.applicationContext
            var storeError: String? = null
            val store = try {
                EncryptedStore.live(app)
            } catch (e: Exception) {
                storeError = e.message ?: e.javaClass.simpleName
                null
            }
            var encoderError: String? = null
            val encoder = try {
                EmbeddingEncoder.load(app)
            } catch (e: Throwable) {
                encoderError = e.message ?: e.javaClass.simpleName
                null
            }
            val motion = MotionRecorder(app)
            val haptics = HapticPlayer(app)
            val runner = CheckRunner(motion, haptics)
            val model = AppModel(store, encoder, encoderError, runner, motion, haptics, app.cacheDir)
            if (storeError != null) model.errorMessage = "Storage is unavailable: $storeError"
            return model
        }
    }

    val encoderError: String? =
        if (encoder == null) (encoderError ?: "The v1 model is missing from the app.") else null

    var people: List<Person> by mutableStateOf(emptyList())
        private set
    var samples: List<TemplateSample> by mutableStateOf(emptyList())
        private set
    var embeddings: List<StoredEmbedding> by mutableStateOf(emptyList())
        private set
    var calibration: Calibration? by mutableStateOf(null)
        private set
    var settings: AppSettings by mutableStateOf(AppSettings())
        private set
    var checkLog: List<CheckLogEntry> by mutableStateOf(emptyList())
        private set
    var identifier: Identifier by mutableStateOf(Identifier(emptyList()))
        private set
    var embeddingIdentifier: EmbeddingIdentifier by mutableStateOf(EmbeddingIdentifier(emptyList()))
        private set

    var lastAnalysis: AnalyzedPattern? by mutableStateOf(null)
    var lastCheck: CheckResult? by mutableStateOf(null)
    var errorMessage: String? by mutableStateOf(null)
    var busyStatus: String? by mutableStateOf(null)
        private set

    init {
        load()
    }

    // Loading and saving

    private fun load() {
        store?.let { s ->
            try {
                people = s.readText(EncryptedStore.peopleFile)
                    ?.let { AppJson.decodeFromString(ListSerializer(Person.serializer()), it) } ?: emptyList()
                samples = s.readText(EncryptedStore.samplesFile)
                    ?.let { RecordingCoding.decodeTemplateSamples(it) } ?: emptyList()
                embeddings = s.readText(EncryptedStore.embeddingsFile)
                    ?.let { AppJson.decodeFromString(ListSerializer(StoredEmbedding.serializer()), it) } ?: emptyList()
                calibration = s.readText(EncryptedStore.calibrationFile)?.let { RecordingCoding.decodeCalibration(it) }
                settings = s.readText(EncryptedStore.settingsFile)
                    ?.let { AppJson.decodeFromString(AppSettings.serializer(), it) } ?: AppSettings()
                checkLog = s.readText(EncryptedStore.checksFile)
                    ?.let { AppJson.decodeFromString(ListSerializer(CheckLogEntry.serializer()), it) } ?: emptyList()
            } catch (e: Exception) {
                errorMessage = "Could not read stored data: ${e.message ?: e.javaClass.simpleName}"
            }
        }
        rebuild()
    }

    private fun save(name: String, text: () -> String) {
        val s = store ?: return
        try {
            s.writeText(name, text())
        } catch (e: Exception) {
            errorMessage = "Could not save $name: ${e.message ?: e.javaClass.simpleName}"
        }
    }

    private fun savePeople() = save(EncryptedStore.peopleFile) {
        AppJson.encodeToString(ListSerializer(Person.serializer()), people)
    }

    private fun saveSamples() = save(EncryptedStore.samplesFile) { RecordingCoding.encodeTemplateSamples(samples) }

    private fun saveEmbeddings() = save(EncryptedStore.embeddingsFile) {
        AppJson.encodeToString(ListSerializer(StoredEmbedding.serializer()), embeddings)
    }

    private fun saveChecks() = save(EncryptedStore.checksFile) {
        AppJson.encodeToString(ListSerializer(CheckLogEntry.serializer()), checkLog)
    }

    private fun saveSettings() = save(EncryptedStore.settingsFile) {
        AppJson.encodeToString(AppSettings.serializer(), settings)
    }

    private fun persist(recording: Recording) {
        val s = store ?: return
        try {
            s.writeRecording(recording)
        } catch (e: Exception) {
            errorMessage = "Could not save a recording: ${e.message ?: e.javaClass.simpleName}"
        }
    }

    private fun deleteRecordings(ids: Collection<UUID>) {
        val s = store ?: return
        for (id in ids.toSet()) runCatching { s.deleteRecording(id) }
    }

    /** Rebuilds both recognizers from the current samples and settings. */
    fun rebuild() {
        val ids = people.map { it.id }.toSet()
        val v0 = samples.filter { it.featureVersion == FeatureExtractor.version && it.prototype.personID in ids }
        identifier = Identifier(v0, settings.modelConfig, settings.identifierConfig)
        val version = encoder?.version
        val v1 = embeddings.filter { it.prototype.personID in ids && (version == null || it.encoderVersion == version) }
        embeddingIdentifier = EmbeddingIdentifier(v1.map { it.sample }, settings.embeddingConfig)
    }

    // Settings

    fun updateSettings(new: AppSettings) {
        if (new == settings) return
        settings = new
        saveSettings()
        rebuild()
    }

    val needsCalibration: Boolean get() = calibration == null && !settings.calibrationSkipped

    fun skipCalibration() = updateSettings(settings.copy(calibrationSkipped = true))

    /** The recognizer that answers: v1 unless it is unavailable or v0 is selected. */
    val effectiveRecognizer: RecognizerKind
        get() = if (settings.recognizer == RecognizerKind.v1 && encoder != null) RecognizerKind.v1 else RecognizerKind.v0

    /** Visible note when v1 is selected but cannot run. */
    val recognizerNote: String?
        get() {
            if (settings.recognizer != RecognizerKind.v1 || encoder != null) return null
            return "The v1 model could not be loaded (${encoderError ?: "unknown error"}). v0 answers instead."
        }

    // Device

    val canRecord: Boolean get() = runner.canRecord

    val recordingUnavailableReason: String? get() = runner.unavailableReason

    /** Starts the motion sensors while a recording screen is visible (reference counted). */
    fun beginSensorSession() {
        motion?.let { if (it.isAvailable) it.start() }
    }

    fun endSensorSession() {
        motion?.let { if (it.isAvailable) it.stop() }
    }

    fun analyze(recording: Recording, mode: QualityMode): AnalyzedPattern =
        PatternAnalyzer.analyze(recording, mode, calibration, settings.qualityConfig, encoder)

    private suspend fun analyzeInBackground(recording: Recording, mode: QualityMode): AnalyzedPattern {
        val cal = calibration
        val quality = settings.qualityConfig
        val enc = encoder
        return withContext(Dispatchers.Default) { PatternAnalyzer.analyze(recording, mode, cal, quality, enc) }
    }

    /** Records one pattern and runs the whole pipeline on it (used by enrollment). */
    suspend fun recordAndAnalyze(purpose: RecordingPurpose, label: RecordingLabel?, mode: QualityMode): AnalyzedPattern {
        val recording = runner.record(purpose, label)
        val a = analyzeInBackground(recording, mode)
        lastAnalysis = a
        return a
    }

    // People

    fun personName(id: String?): String {
        if (id == null) return "?"
        return people.firstOrNull { it.id == id }?.name ?: "Removed person"
    }

    fun handName(hand: Hand): String = if (hand == Hand.right) "right hand" else "left hand"

    fun prototypeName(key: PrototypeKey): String {
        var s = "${personName(key.personID)} · ${key.hand.rawValue}"
        if (key.accessory != Accessory.none) s += " · ${key.accessory.rawValue}"
        return s
    }

    fun headline(outcome: DecisionOutcome): String = when (outcome) {
        is DecisionOutcome.Identified -> "${personName(outcome.personID)} · ${handName(outcome.hand)}"
        is DecisionOutcome.IdentifiedHandUnclear -> "${personName(outcome.personID)} · hand unclear"
        DecisionOutcome.NeedMorePatterns -> "Undecided"
        is DecisionOutcome.Unknown -> "Unknown"
    }

    fun reasonText(outcome: DecisionOutcome): String? {
        if (outcome !is DecisionOutcome.Unknown) return null
        return when (outcome.reason) {
            UnknownReason.noPeople -> "Nobody is enrolled yet."
            UnknownReason.noMatch -> "This hand does not match anyone enrolled."
            UnknownReason.ambiguous -> "Too close between two people."
        }
    }

    fun prototypeCounts(personID: String): List<PrototypeCount> {
        val keys = HashSet<PrototypeKey>()
        samples.filter { it.prototype.personID == personID }.forEach { keys.add(it.prototype) }
        embeddings.filter { it.prototype.personID == personID }.forEach { keys.add(it.prototype) }
        return keys.sorted().map { key ->
            PrototypeCount(key, samples.count { it.prototype == key }, embeddings.count { it.prototype == key })
        }
    }

    fun rename(personID: String, name: String) {
        val trimmed = name.trim()
        if (trimmed.isEmpty() || people.none { it.id == personID }) return
        people = people.map { if (it.id == personID) it.copy(name = trimmed) else it }
        savePeople()
    }

    fun deletePerson(personID: String) {
        val recs = samples.filter { it.prototype.personID == personID }.mapNotNull { it.recordingID } +
            embeddings.filter { it.prototype.personID == personID }.mapNotNull { it.recordingID }
        deleteRecordings(recs)
        samples = samples.filterNot { it.prototype.personID == personID }
        embeddings = embeddings.filterNot { it.prototype.personID == personID }
        people = people.filterNot { it.id == personID }
        savePeople(); saveSamples(); saveEmbeddings()
        rebuild()
    }

    /** Stores the accepted patterns of a finished enrollment, accessory or re-enrollment. */
    fun commitEnrollment(flow: EnrollmentFlow, patterns: List<Pair<EnrollmentSlot, AnalyzedPattern>>) {
        val now = Instant.now()
        val personID = flow.personID
        var newSamples = samples
        var newEmbeddings = embeddings
        when (val kind = flow.kind) {
            EnrollmentKind.NewPerson -> {
                if (people.none { it.id == personID }) {
                    people = people + Person(id = personID, name = flow.name, createdAt = now)
                }
            }
            is EnrollmentKind.Reenroll -> {
                fun replaced(key: PrototypeKey, source: SampleSource) =
                    key.personID == personID && key.accessory == Accessory.none && source != SampleSource.accessory
                deleteRecordings(newSamples.filter { replaced(it.prototype, it.source) }.mapNotNull { it.recordingID })
                newSamples = newSamples.filterNot { replaced(it.prototype, it.source) }
                newEmbeddings = newEmbeddings.filterNot { replaced(it.prototype, it.source) }
            }
            is EnrollmentKind.AccessoryEnroll -> {
                fun replaced(key: PrototypeKey) = key.personID == personID && key.accessory == kind.accessory
                deleteRecordings(newSamples.filter { replaced(it.prototype) }.mapNotNull { it.recordingID })
                newSamples = newSamples.filterNot { replaced(it.prototype) }
                newEmbeddings = newEmbeddings.filterNot { replaced(it.prototype) }
            }
        }
        val addedSamples = ArrayList<TemplateSample>()
        val addedEmbeddings = ArrayList<StoredEmbedding>()
        for ((slot, analysis) in patterns) {
            val key = PrototypeKey(personID, slot.hand, flow.accessory)
            val recording = analysis.recording
            persist(recording)
            addedSamples.add(TemplateSample(
                prototype = key, source = flow.sampleSource, createdAt = now, features = analysis.features,
                featureVersion = FeatureExtractor.version, recordingID = recording.id,
            ))
            val e = analysis.embedding
            val enc = encoder
            if (e != null && enc != null) {
                addedEmbeddings.add(StoredEmbedding(
                    recordingID = recording.id, encoderVersion = enc.version,
                    sample = EmbeddingSample(prototype = key, embedding = e, source = flow.sampleSource, createdAt = now),
                ))
            }
        }
        samples = newSamples + addedSamples
        embeddings = newEmbeddings + addedEmbeddings
        savePeople(); saveSamples(); saveEmbeddings()
        rebuild()
    }

    // Calibration (SPEC 8)

    /** Records 20 patterns on a table; null when fewer than 12 pass. */
    suspend fun runCalibration(progress: (done: Int, passed: Int, last: AnalyzedPattern) -> Unit): Calibration? {
        val processed = ArrayList<com.mahmoudalyosify.auditor.core.ProcessedSignal>()
        val reports = ArrayList<QualityReport>()
        for (i in 0 until Calibration.patternCount) {
            coroutineContext.ensureActive()
            val recording = runner.record(RecordingPurpose.calibration, null)
            persist(recording)
            val a = analyzeInBackground(recording, QualityMode.onTable)
            lastAnalysis = a
            processed.add(a.processed)
            reports.add(a.quality)
            progress(i + 1, reports.count { it.passed }, a)
        }
        val cal = withContext(Dispatchers.Default) {
            Calibration.build(processed, reports, DeviceSupport.modelIdentifier, Instant.now())
        } ?: return null
        calibration = cal
        save(EncryptedStore.calibrationFile) { RecordingCoding.encode(cal) }
        recomputeAll()
        return cal
    }

    // Checks (SPEC 9.4 / 14.3)

    /**
     * Sequential check: records patterns until the answering recognizer stops (at most
     * `maxPatterns`), retrying each slot up to 3 times on quality failure. The other
     * recognizer decides on the same patterns and both are logged.
     */
    suspend fun runCheck(
        purpose: RecordingPurpose = RecordingPurpose.check,
        truth: RecordingLabel? = null,
        requestedHand: Hand? = null,
        status: (String) -> Unit,
    ): CheckResult {
        val started = Instant.now()
        val startedNanos = System.nanoTime()
        var answeredBy = effectiveRecognizer
        val notes = ArrayList<String>()
        recognizerNote?.let { notes.add(it) }

        val maxPatterns = maxOf(1, settings.maxPatterns)
        val accepted = ArrayList<AnalyzedPattern>()
        val recordingIDs = ArrayList<UUID>()
        var attempts = 0
        var qualityFailure: String? = null
        var v0: Decision? = null
        var v1: Decision? = null

        while (accepted.size < maxPatterns) {
            var tries = 0
            var got: AnalyzedPattern? = null
            var lastMessage = "Try again"
            while (got == null && tries <= maxRetriesPerSlot) {
                coroutineContext.ensureActive()
                status(if (tries == 0) "Pattern ${accepted.size + 1} — keep still…" else "$lastMessage — retrying…")
                val recording = runner.record(purpose, truth)
                attempts += 1
                tries += 1
                persist(recording)
                recordingIDs.add(recording.id)
                val a = analyzeInBackground(recording, QualityMode.handheld)
                lastAnalysis = a
                if (a.passed) {
                    got = a
                } else {
                    lastMessage = a.qualityMessage ?: "Try again"
                    status(lastMessage)
                }
            }
            if (got == null) {
                qualityFailure = lastMessage
                break
            }
            accepted.add(got)
            v0 = identifier.decide(accepted.map { it.features })
            val embeddingsOK = encoder != null && accepted.all { it.embedding != null }
            if (embeddingsOK) {
                v1 = embeddingIdentifier.decide(accepted.mapNotNull { it.embedding })
            } else {
                v1 = null
                if (answeredBy == RecognizerKind.v1) {
                    answeredBy = RecognizerKind.v0
                    notes.add("The v1 model failed on a pattern; v0 answered this check.")
                }
            }
            val answering = if (answeredBy == RecognizerKind.v1) v1 else v0
            if (answering != null && answering.outcome != DecisionOutcome.NeedMorePatterns) break
        }

        val decision = if (qualityFailure == null) (if (answeredBy == RecognizerKind.v1) v1 else v0) else null
        val result = CheckResult(
            purpose = purpose, answeredBy = answeredBy, decision = decision, v0 = v0, v1 = v1,
            patterns = accepted, attempts = attempts, seconds = (System.nanoTime() - startedNanos) / 1e9,
            requestedHand = requestedHand, qualityFailure = qualityFailure, notes = notes,
        )
        if (purpose == RecordingPurpose.check && settings.learnFromConfidentChecks && qualityFailure == null) {
            result.learned = learn(result)
        }
        lastCheck = result
        checkLog = checkLog + CheckLogEntry(
            id = result.id, createdAt = started, purpose = purpose, recognizer = answeredBy,
            recordingIDs = recordingIDs, patternsUsed = accepted.size, attempts = attempts, seconds = result.seconds,
            v0 = v0?.let { DecisionSummary.of(it) }, v1 = v1?.let { DecisionSummary.of(it) },
            truthPersonID = truth?.personID, truthHand = truth?.hand, requestedHand = requestedHand,
            challengePassed = result.challengePassed, qualityFailure = qualityFailure,
        )
        saveChecks()
        return result
    }

    /** Random hand for the hand challenge (SPEC 9.6): `uniform() < 0.5` → right. */
    fun drawChallengeHand(): Hand {
        val rng = Xoshiro256StarStar(DeviceSupport.randomSeed())
        return if (rng.uniform() < 0.5) Hand.right else Hand.left
    }

    /**
     * Adaptive templates (SPEC 9.5 / 14.3): each recognizer learns from its own confident
     * decision when it names the same person and hand as the answer (and the hand challenge,
     * if any, passed). Returns the number of samples added.
     */
    private fun learn(result: CheckResult): Int {
        val answer = result.decision?.outcome as? DecisionOutcome.Identified ?: return 0
        if (result.challengePassed == false) return 0
        fun agrees(x: Decision?): Boolean {
            val o = x?.outcome as? DecisionOutcome.Identified ?: return false
            return o.personID == answer.personID && o.hand == answer.hand
        }
        val now = Instant.now()
        var added = 0
        val v0 = result.v0
        if (v0 != null && agrees(v0)) {
            val features = result.patterns.map { it.features }
            val new = identifier.adaptiveSamples(v0, features, now).mapIndexed { i, s ->
                if (i < result.patterns.size) s.copy(recordingID = result.patterns[i].recording.id) else s
            }
            if (new.isNotEmpty()) {
                samples = Identifier.applyAdaptiveCap(samples + new, settings.identifierConfig.adaptiveMaxPerPrototype)
                saveSamples()
                added += new.size
            }
        }
        val v1 = result.v1
        val enc = encoder
        if (v1 != null && enc != null && agrees(v1)) {
            val vectors = result.patterns.mapNotNull { it.embedding }
            val new = embeddingIdentifier.adaptiveSamples(v1, vectors, now)
            if (new.isNotEmpty() && vectors.size == result.patterns.size) {
                val stored = new.mapIndexed { i, s ->
                    StoredEmbedding(
                        recordingID = if (i < result.patterns.size) result.patterns[i].recording.id else null,
                        encoderVersion = enc.version, sample = s,
                    )
                }
                embeddings = applyAdaptiveCap(embeddings + stored, settings.embeddingConfig.adaptiveMaxPerPrototype)
                saveEmbeddings()
                added += stored.size
            }
        }
        if (added > 0) rebuild()
        return added
    }

    // Lab: test runs and confusion matrices

    fun predictedLabel(summary: DecisionSummary?, qualityFailure: String?): String {
        if (qualityFailure != null) return "No answer"
        val s = summary ?: return "n/a"
        return when (s.outcome) {
            "identified" -> "${personName(s.personID)} · ${s.hand?.rawValue ?: "?"}"
            "handUnclear" -> "${personName(s.personID)} · ?"
            "needMorePatterns" -> "Undecided"
            else -> "Unknown"
        }
    }

    val testEntries: List<CheckLogEntry>
        get() = checkLog.filter { it.purpose == RecordingPurpose.test && it.truthPersonID != null }

    fun confusionMatrix(kind: RecognizerKind): ConfusionMatrix {
        val m = ConfusionMatrix()
        for (e in testEntries) {
            val truth = "${personName(e.truthPersonID)} · ${e.truthHand?.rawValue ?: "?"}"
            m.add(truth, predictedLabel(if (kind == RecognizerKind.v0) e.v0 else e.v1, e.qualityFailure))
        }
        return m
    }

    fun clearTestResults() {
        checkLog = checkLog.filterNot { it.purpose == RecordingPurpose.test }
        saveChecks()
    }

    // Lab: signal check

    suspend fun runSignalCheck(progress: (Int) -> Unit): SignalCheckResult {
        val patterns = ArrayList<AnalyzedPattern>()
        for (i in 0 until 3) {
            coroutineContext.ensureActive()
            val recording = runner.record(RecordingPurpose.signalCheck, null)
            persist(recording)
            val a = analyzeInBackground(recording, QualityMode.handheld)
            lastAnalysis = a
            patterns.add(a)
            progress(i + 1)
        }
        val perPattern = patterns.map { SignalSummary.channelSNR(it.processed) }
        val channels = ArrayList<Pair<String, Double>>()
        perPattern.firstOrNull()?.forEachIndexed { c, entry ->
            val values = perPattern.mapNotNull { it.getOrNull(c)?.second }
            channels.add(entry.first to values.sum() / maxOf(values.size, 1))
        }
        val minSNR = QualityConfig.default.minSNR
        val visibleCount = patterns.count { it.quality.metrics.snr >= minSNR && it.quality.metrics.ncc >= settings.minNCC }
        return SignalCheckResult(patterns, channels, visibleCount >= 2)
    }

    // Recomputation

    /** True when stored features or embeddings are out of date (feature version, model). */
    val needsRecompute: Boolean
        get() {
            if (samples.any { it.featureVersion != FeatureExtractor.version }) return true
            val enc = encoder ?: return false
            if (embeddings.any { it.encoderVersion != enc.version }) return true
            val linked = embeddings.mapNotNull { it.recordingID }.toSet()
            return samples.any { s ->
                val id = s.recordingID
                s.source != SampleSource.adaptive && id != null && id !in linked
            }
        }

    /**
     * Recomputes every sample's features (and embeddings) from the stored raw recordings with
     * the current calibration, feature extractor and model.
     */
    suspend fun recomputeAll() {
        val s = store ?: return
        if (busyStatus != null) return
        val ids = (samples.mapNotNull { it.recordingID } + embeddings.mapNotNull { it.recordingID })
            .toSet().sortedBy { it.toString().uppercase() }
        val cal = calibration
        val enc = encoder
        val featuresByRecording = HashMap<UUID, List<Double>>()
        val embeddingByRecording = HashMap<UUID, List<Double>>()
        busyStatus = "Recomputing 0 of ${ids.size}…"
        try {
            for ((i, id) in ids.withIndex()) {
                if (i % 4 == 0) busyStatus = "Recomputing ${i + 1} of ${ids.size}…"
                withContext(Dispatchers.Default) {
                    val recording = runCatching { s.readRecording(id) }.getOrNull() ?: return@withContext
                    val p = SignalProcessor.process(recording)
                    featuresByRecording[id] = FeatureExtractor.extract(p, cal)
                    if (enc != null) {
                        runCatching { enc.encode(p) }.getOrNull()?.let { embeddingByRecording[id] = it.embedding }
                    }
                }
            }
            samples = samples.mapNotNull { sample ->
                val f = sample.recordingID?.let { featuresByRecording[it] }
                when {
                    f != null -> sample.copy(features = f, featureVersion = FeatureExtractor.version)
                    sample.featureVersion == FeatureExtractor.version -> sample
                    else -> null
                }
            }
            if (enc != null) {
                val updated = embeddings.mapNotNull { e ->
                    val v = e.recordingID?.let { embeddingByRecording[it] }
                    when {
                        v != null -> e.copy(sample = e.sample.copy(embedding = v), encoderVersion = enc.version)
                        e.encoderVersion == enc.version -> e
                        else -> null
                    }
                }.toMutableList()
                val linked = updated.mapNotNull { it.recordingID }.toMutableSet()
                for (sample in samples) {
                    if (sample.source == SampleSource.adaptive) continue
                    val id = sample.recordingID ?: continue
                    if (id in linked) continue
                    val v = embeddingByRecording[id] ?: continue
                    updated.add(StoredEmbedding(
                        recordingID = id, encoderVersion = enc.version,
                        sample = EmbeddingSample(prototype = sample.prototype, embedding = v, source = sample.source,
                            createdAt = sample.createdAt),
                    ))
                    linked.add(id)
                }
                embeddings = updated
            }
            saveSamples()
            saveEmbeddings()
            rebuild()
        } finally {
            busyStatus = null
        }
    }

    // Data management

    fun deleteAllData() {
        try {
            store?.deleteAll()
        } catch (e: Exception) {
            errorMessage = "Could not delete all data: ${e.message ?: e.javaClass.simpleName}"
        }
        people = emptyList()
        samples = emptyList()
        embeddings = emptyList()
        calibration = null
        settings = AppSettings()
        checkLog = emptyList()
        lastAnalysis = null
        lastCheck = null
        rebuild()
    }

    /** Writes recordings.jsonl, features.csv and checks.csv to a cache folder shared through FileProvider. */
    suspend fun makeExport(): List<File> {
        val base = File(cacheDir ?: throw IllegalStateException("No cache folder"), "exports")
        val stamp = DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss").withZone(ZoneOffset.UTC).format(Instant.now())
        val names = Exporter.nameMap(people, settings.anonymizeExports)
        val currentSamples = samples
        val currentChecks = checkLog
        val s = store
        return withContext(Dispatchers.IO) {
            base.listFiles()?.forEach { it.deleteRecursively() }
            val folder = File(base, "7oda-export-$stamp")
            if (!folder.isDirectory && !folder.mkdirs()) throw IllegalStateException("Could not create the export folder")
            val recordings = Exporter.prepare(s?.allRecordings() ?: emptyList(), names)
            val files = listOf(
                "recordings.jsonl" to RecordingCoding.jsonLines(recordings),
                "features.csv" to Exporter.featuresCSV(currentSamples, names),
                "checks.csv" to Exporter.checksCSV(currentChecks, names),
            )
            files.map { (name, text) -> File(folder, name).also { it.writeText(text, Charsets.UTF_8) } }
        }
    }
}
