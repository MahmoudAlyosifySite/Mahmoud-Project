import Foundation
import Observation
import AuditorCore

/// The result of one "Who's holding?" check (or one Lab test check).
struct CheckResult: Identifiable {
    let id = UUID()
    var purpose: RecordingPurpose
    /// The recognizer whose decision is shown.
    var answeredBy: RecognizerKind
    /// Final decision of `answeredBy`; nil when no clean pattern could be recorded.
    var decision: Decision?
    var v0: Decision?
    var v1: Decision?
    /// Patterns that passed the quality gate and were used.
    var patterns: [AnalyzedPattern]
    var attempts: Int
    var seconds: Double
    var requestedHand: Hand?
    var qualityFailure: String?
    var notes: [String]
    var learned: Int

    /// Hand challenge (SPEC 9.6): passes only for identified(p, requested hand).
    var challengePassed: Bool? {
        guard let requested = requestedHand else { return nil }
        if let d = decision, case .identified(_, let hand) = d.outcome { return hand == requested }
        return false
    }

    func decision(of kind: RecognizerKind) -> Decision? { kind == .v0 ? v0 : v1 }
}

/// Result of the Lab signal check.
struct SignalCheckResult {
    var patterns: [AnalyzedPattern]
    /// Mean SNR per channel over the patterns (accel HP x/y/z, gyro HP x/y/z).
    var channels: [(name: String, snr: Double)]
    var visible: Bool
}

/// Sample counts of one prototype (v0 template samples and v1 embeddings).
struct PrototypeCount: Identifiable {
    var key: PrototypeKey
    var v0: Int
    var v1: Int
    var id: String { "\(key.personID)|\(key.hand.rawValue)|\(key.accessory.rawValue)" }
}

/// App state: people, samples of both recognizers, calibration, settings and the flows.
@Observable
@MainActor
final class AppModel {
    static let maxRetriesPerSlot = 3

    private(set) var people: [Person] = []
    private(set) var samples: [TemplateSample] = []
    private(set) var embeddings: [StoredEmbedding] = []
    private(set) var calibration: Calibration?
    private(set) var settings = AppSettings()
    private(set) var checkLog: [CheckLogEntry] = []
    private(set) var identifier = Identifier(samples: [])
    private(set) var embeddingIdentifier = EmbeddingIdentifier(samples: [])

    var lastAnalysis: AnalyzedPattern?
    var lastCheck: CheckResult?
    var errorMessage: String?
    private(set) var busyStatus: String?

    let store: EncryptedStore?
    let encoder: EmbeddingEncoder?
    let encoderError: String?
    let motion: MotionRecorder?
    let runner: PatternRecording

    init(store: EncryptedStore?, encoder: EmbeddingEncoder?, encoderError: String?,
         runner: PatternRecording, motion: MotionRecorder?) {
        self.store = store
        self.encoder = encoder
        self.encoderError = encoder == nil ? (encoderError ?? EncoderError.modelMissing.localizedDescription) : nil
        self.runner = runner
        self.motion = motion
        load()
    }

    /// The real app: Keychain-keyed store, bundled model, device sensors and haptics.
    static func live() -> AppModel {
        var store: EncryptedStore?
        var storeError: String?
        do {
            store = try EncryptedStore(directory: try EncryptedStore.defaultDirectory(),
                                       keyProvider: KeychainKeyProvider())
        } catch {
            storeError = error.localizedDescription
        }
        var encoder: EmbeddingEncoder?
        var encoderError: String?
        do {
            encoder = try EmbeddingEncoder()
        } catch {
            encoderError = error.localizedDescription
        }
        let motion = MotionRecorder.shared
        let runner = CheckRunner(recorder: motion, haptics: HapticPlayer())
        let model = AppModel(store: store, encoder: encoder, encoderError: encoderError, runner: runner, motion: motion)
        if let e = storeError { model.errorMessage = "Storage is unavailable: \(e)" }
        return model
    }

    // MARK: Loading and saving

    private func load() {
        if let store = store {
            do {
                people = try store.read([Person].self, from: EncryptedStore.peopleFile) ?? []
                samples = try store.read([TemplateSample].self, from: EncryptedStore.samplesFile) ?? []
                embeddings = try store.read([StoredEmbedding].self, from: EncryptedStore.embeddingsFile) ?? []
                calibration = try store.read(Calibration.self, from: EncryptedStore.calibrationFile)
                settings = try store.read(AppSettings.self, from: EncryptedStore.settingsFile) ?? AppSettings()
                checkLog = try store.read([CheckLogEntry].self, from: EncryptedStore.checksFile) ?? []
            } catch {
                errorMessage = "Could not read stored data: \(error.localizedDescription)"
            }
        }
        rebuild()
    }

    private func save<T: Encodable>(_ value: T, _ name: String) {
        guard let store = store else { return }
        do {
            try store.write(value, to: name)
        } catch {
            errorMessage = "Could not save \(name): \(error.localizedDescription)"
        }
    }

    private func savePeople() { save(people, EncryptedStore.peopleFile) }
    private func saveSamples() { save(samples, EncryptedStore.samplesFile) }
    private func saveEmbeddings() { save(embeddings, EncryptedStore.embeddingsFile) }
    private func saveChecks() { save(checkLog, EncryptedStore.checksFile) }

    private func persist(_ recording: Recording) {
        guard let store = store else { return }
        do {
            try store.writeRecording(recording)
        } catch {
            errorMessage = "Could not save a recording: \(error.localizedDescription)"
        }
    }

    private func deleteRecordings(_ ids: [UUID]) {
        guard let store = store else { return }
        for id in Set(ids) { try? store.deleteRecording(id: id) }
    }

    /// Rebuilds both recognizers from the current samples and settings.
    func rebuild() {
        let ids = Set(people.map { $0.id })
        let v0 = samples.filter { $0.featureVersion == FeatureExtractor.version && ids.contains($0.prototype.personID) }
        identifier = Identifier(samples: v0, modelConfig: settings.modelConfig, config: settings.identifierConfig)
        let version = encoder?.version
        let v1 = embeddings.filter { ids.contains($0.prototype.personID) && (version == nil || $0.encoderVersion == version) }
        embeddingIdentifier = EmbeddingIdentifier(samples: v1.map { $0.sample }, config: settings.embeddingConfig)
    }

    // MARK: Settings

    func updateSettings(_ new: AppSettings) {
        guard new != settings else { return }
        settings = new
        save(settings, EncryptedStore.settingsFile)
        rebuild()
    }

    var needsCalibration: Bool { calibration == nil && !settings.calibrationSkipped }

    func skipCalibration() {
        var s = settings
        s.calibrationSkipped = true
        updateSettings(s)
    }

    /// The recognizer that answers: v1 unless it is unavailable or v0 is selected.
    var effectiveRecognizer: RecognizerKind {
        settings.recognizer == .v1 && encoder != nil ? .v1 : .v0
    }

    /// Visible note when v1 is selected but cannot run.
    var recognizerNote: String? {
        guard settings.recognizer == .v1, encoder == nil else { return nil }
        return "The v1 model could not be loaded (\(encoderError ?? "unknown error")). v0 answers instead."
    }

    // MARK: Device

    var canRecord: Bool { runner.canRecord }

    var recordingUnavailableReason: String? {
        if let r = runner as? CheckRunner { return r.unavailableReason }
        return runner.canRecord ? nil : RecordingError.sensorsUnavailable.localizedDescription
    }

    /// Starts the motion sensors while a recording screen is visible (reference counted).
    func beginSensorSession() { motion?.start() }
    func endSensorSession() { motion?.stop() }

    func analyze(_ recording: Recording, mode: QualityMode) -> AnalyzedPattern {
        PatternAnalyzer.analyze(recording, mode: mode, calibration: calibration,
                                qualityConfig: settings.qualityConfig, encoder: encoder)
    }

    // MARK: People

    func personName(_ id: String?) -> String {
        guard let id = id else { return "?" }
        return people.first { $0.id == id }?.name ?? "Removed person"
    }

    func prototypeCounts(for personID: String) -> [PrototypeCount] {
        var keys = Set<PrototypeKey>()
        for s in samples where s.prototype.personID == personID { keys.insert(s.prototype) }
        for e in embeddings where e.prototype.personID == personID { keys.insert(e.prototype) }
        return keys.sorted().map { key in
            PrototypeCount(key: key,
                           v0: samples.filter { $0.prototype == key }.count,
                           v1: embeddings.filter { $0.prototype == key }.count)
        }
    }

    func rename(personID: String, to name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let i = people.firstIndex(where: { $0.id == personID }) else { return }
        people[i].name = trimmed
        savePeople()
    }

    func deletePerson(_ personID: String) {
        let recs = samples.filter { $0.prototype.personID == personID }.compactMap { $0.recordingID }
            + embeddings.filter { $0.prototype.personID == personID }.compactMap { $0.recordingID }
        deleteRecordings(recs)
        samples.removeAll { $0.prototype.personID == personID }
        embeddings.removeAll { $0.prototype.personID == personID }
        people.removeAll { $0.id == personID }
        savePeople(); saveSamples(); saveEmbeddings()
        rebuild()
    }

    /// Stores the accepted patterns of a finished enrollment, accessory or re-enrollment.
    func commitEnrollment(flow: EnrollmentFlow, patterns: [(slot: EnrollmentSlot, analysis: AnalyzedPattern)]) {
        let now = Date()
        let personID = flow.personID
        switch flow.kind {
        case .newPerson:
            if !people.contains(where: { $0.id == personID }) {
                people.append(Person(id: personID, name: flow.name, createdAt: now))
            }
        case .reenroll:
            let replaced: (PrototypeKey, SampleSource) -> Bool = { key, source in
                key.personID == personID && key.accessory == .none && source != .accessory
            }
            deleteRecordings(samples.filter { replaced($0.prototype, $0.source) }.compactMap { $0.recordingID })
            samples.removeAll { replaced($0.prototype, $0.source) }
            embeddings.removeAll { replaced($0.prototype, $0.source) }
        case .accessory(_, _, let accessory):
            let replaced: (PrototypeKey) -> Bool = { $0.personID == personID && $0.accessory == accessory }
            deleteRecordings(samples.filter { replaced($0.prototype) }.compactMap { $0.recordingID })
            samples.removeAll { replaced($0.prototype) }
            embeddings.removeAll { replaced($0.prototype) }
        }
        for item in patterns {
            let key = PrototypeKey(personID: personID, hand: item.slot.hand, accessory: flow.accessory)
            let recording = item.analysis.recording
            persist(recording)
            samples.append(TemplateSample(prototype: key, source: flow.sampleSource, createdAt: now,
                                          features: item.analysis.features,
                                          featureVersion: FeatureExtractor.version, recordingID: recording.id))
            if let e = item.analysis.embedding, let enc = encoder {
                embeddings.append(StoredEmbedding(
                    recordingID: recording.id, encoderVersion: enc.version,
                    sample: EmbeddingSample(prototype: key, embedding: e, source: flow.sampleSource, createdAt: now)))
            }
        }
        savePeople(); saveSamples(); saveEmbeddings()
        rebuild()
    }

    // MARK: Calibration (SPEC 8)

    /// Records 20 patterns on a table; nil when fewer than 12 pass.
    func runCalibration(progress: (_ done: Int, _ passed: Int, _ last: AnalyzedPattern) -> Void) async throws -> Calibration? {
        var processed: [ProcessedSignal] = []
        var reports: [QualityReport] = []
        for i in 0..<Calibration.patternCount {
            try Task.checkCancellation()
            let recording = try await runner.record(purpose: .calibration, label: nil)
            persist(recording)
            let a = analyze(recording, mode: .onTable)
            lastAnalysis = a
            processed.append(a.processed)
            reports.append(a.quality)
            progress(i + 1, reports.filter { $0.passed }.count, a)
        }
        guard let cal = Calibration.build(from: processed, reports: reports, device: DeviceSupport.modelIdentifier,
                                          now: Date()) else { return nil }
        calibration = cal
        save(cal, EncryptedStore.calibrationFile)
        await recomputeAll()
        return cal
    }

    // MARK: Checks (SPEC 9.4 / 14.3)

    /// Sequential check: records patterns until the answering recognizer stops (at most
    /// `maxPatterns`), retrying each slot up to 3 times on quality failure. The other
    /// recognizer decides on the same patterns and both are logged.
    func runCheck(purpose: RecordingPurpose = .check, truth: RecordingLabel? = nil,
                  status: (String) -> Void) async throws -> CheckResult {
        let started = Date()
        var answeredBy = effectiveRecognizer
        var notes: [String] = []
        if let note = recognizerNote { notes.append(note) }

        var requestedHand: Hand?
        if purpose == .check && settings.handChallenge {
            let seed = (try? DeviceSupport.randomSeed()) ?? UInt64.random(in: 0...UInt64.max)
            var rng = Xoshiro256StarStar(seed: seed)
            requestedHand = rng.uniform() < 0.5 ? .right : .left
        }

        let maxPatterns = max(1, settings.maxPatterns)
        var accepted: [AnalyzedPattern] = []
        var recordingIDs: [UUID] = []
        var attempts = 0
        var qualityFailure: String?
        var v0: Decision?
        var v1: Decision?

        while accepted.count < maxPatterns {
            var tries = 0
            var got: AnalyzedPattern?
            var lastMessage = "Try again"
            while got == nil && tries <= Self.maxRetriesPerSlot {
                try Task.checkCancellation()
                status(tries == 0 ? "Pattern \(accepted.count + 1) — keep still…" : "\(lastMessage) — retrying…")
                let recording = try await runner.record(purpose: purpose, label: truth)
                attempts += 1
                tries += 1
                persist(recording)
                recordingIDs.append(recording.id)
                let a = analyze(recording, mode: .handheld)
                lastAnalysis = a
                if a.passed {
                    got = a
                } else {
                    lastMessage = a.qualityMessage ?? "Try again"
                    status(lastMessage)
                }
            }
            guard let pattern = got else {
                qualityFailure = lastMessage
                break
            }
            accepted.append(pattern)
            v0 = identifier.decide(accepted.map { $0.features })
            let embeddingsOK = encoder != nil && accepted.allSatisfy { $0.embedding != nil }
            if embeddingsOK {
                v1 = embeddingIdentifier.decide(accepted.compactMap { $0.embedding })
            } else {
                v1 = nil
                if answeredBy == .v1 {
                    answeredBy = .v0
                    notes.append("The v1 model failed on a pattern; v0 answered this check.")
                }
            }
            let answering = answeredBy == .v1 ? v1 : v0
            if let d = answering, d.outcome != .needMorePatterns { break }
        }

        let decision = qualityFailure == nil ? (answeredBy == .v1 ? v1 : v0) : nil
        var result = CheckResult(purpose: purpose, answeredBy: answeredBy, decision: decision, v0: v0, v1: v1,
                                 patterns: accepted, attempts: attempts,
                                 seconds: Date().timeIntervalSince(started), requestedHand: requestedHand,
                                 qualityFailure: qualityFailure, notes: notes, learned: 0)
        if purpose == .check && settings.learnFromConfidentChecks && qualityFailure == nil {
            result.learned = learn(from: result)
        }
        lastCheck = result
        checkLog.append(CheckLogEntry(
            id: result.id, createdAt: started, purpose: purpose, recognizer: answeredBy, recordingIDs: recordingIDs,
            patternsUsed: accepted.count, attempts: attempts, seconds: result.seconds,
            v0: v0.map { DecisionSummary($0) }, v1: v1.map { DecisionSummary($0) },
            truthPersonID: truth?.personID, truthHand: truth?.hand, requestedHand: requestedHand,
            challengePassed: result.challengePassed, qualityFailure: qualityFailure))
        saveChecks()
        return result
    }

    /// Adaptive templates (SPEC 9.5 / 14.3): each recognizer learns from its own confident
    /// decision when it names the same person and hand as the answer (and the hand
    /// challenge, if any, passed). Returns the number of samples added.
    private func learn(from result: CheckResult) -> Int {
        guard let d = result.decision, case .identified(let person, let hand) = d.outcome else { return 0 }
        if let passed = result.challengePassed, !passed { return 0 }
        func agrees(_ x: Decision?) -> Bool {
            guard let x = x, case .identified(let p, let h) = x.outcome else { return false }
            return p == person && h == hand
        }
        let now = Date()
        var added = 0
        if agrees(result.v0) {
            let features = result.patterns.map { $0.features }
            var new = identifier.adaptiveSamples(for: result.v0!, patterns: features, now: now)
            for i in new.indices where i < result.patterns.count {
                new[i].recordingID = result.patterns[i].recording.id
            }
            if !new.isEmpty {
                samples = Identifier.applyAdaptiveCap(samples + new, cap: settings.identifierConfig.adaptiveMaxPerPrototype)
                saveSamples()
                added += new.count
            }
        }
        if agrees(result.v1), let enc = encoder {
            let vectors = result.patterns.compactMap { $0.embedding }
            let new = embeddingIdentifier.adaptiveSamples(for: result.v1!, patterns: vectors, now: now)
            if !new.isEmpty && vectors.count == result.patterns.count {
                let stored = new.enumerated().map { i, s in
                    StoredEmbedding(recordingID: i < result.patterns.count ? result.patterns[i].recording.id : nil,
                                    encoderVersion: enc.version, sample: s)
                }
                embeddings = applyAdaptiveCap(embeddings + stored, cap: settings.embeddingConfig.adaptiveMaxPerPrototype)
                saveEmbeddings()
                added += stored.count
            }
        }
        if added > 0 { rebuild() }
        return added
    }

    // MARK: Lab: test runs and confusion matrices

    func predictedLabel(_ summary: DecisionSummary?, qualityFailure: String?) -> String {
        if qualityFailure != nil { return "No answer" }
        guard let s = summary else { return "n/a" }
        switch s.outcome {
        case "identified": return "\(personName(s.personID)) · \(s.hand?.rawValue ?? "?")"
        case "handUnclear": return "\(personName(s.personID)) · ?"
        case "needMorePatterns": return "Undecided"
        default: return "Unknown"
        }
    }

    var testEntries: [CheckLogEntry] { checkLog.filter { $0.purpose == .test && $0.truthPersonID != nil } }

    func confusionMatrix(for kind: RecognizerKind) -> ConfusionMatrix {
        var m = ConfusionMatrix()
        for e in testEntries {
            let truth = "\(personName(e.truthPersonID)) · \(e.truthHand?.rawValue ?? "?")"
            m.add(truth: truth, predicted: predictedLabel(kind == .v0 ? e.v0 : e.v1, qualityFailure: e.qualityFailure))
        }
        return m
    }

    func clearTestResults() {
        checkLog.removeAll { $0.purpose == .test }
        saveChecks()
    }

    // MARK: Lab: signal check

    func runSignalCheck(progress: (Int) -> Void) async throws -> SignalCheckResult {
        var patterns: [AnalyzedPattern] = []
        for i in 0..<3 {
            try Task.checkCancellation()
            let recording = try await runner.record(purpose: .signalCheck, label: nil)
            persist(recording)
            let a = analyze(recording, mode: .handheld)
            lastAnalysis = a
            patterns.append(a)
            progress(i + 1)
        }
        let perPattern = patterns.map { SignalSummary.channelSNR($0.processed) }
        var channels: [(name: String, snr: Double)] = []
        if let first = perPattern.first {
            for (c, entry) in first.enumerated() {
                let values = perPattern.compactMap { c < $0.count ? $0[c].snr : nil }
                channels.append((name: entry.name, snr: values.reduce(0, +) / Double(max(values.count, 1))))
            }
        }
        let minSNR = QualityConfig.default.minSNR
        let visibleCount = patterns.filter { $0.quality.metrics.snr >= minSNR && $0.quality.metrics.ncc >= settings.minNCC }.count
        return SignalCheckResult(patterns: patterns, channels: channels, visible: visibleCount >= 2)
    }

    // MARK: Recomputation

    /// True when stored features or embeddings are out of date (feature version, model).
    var needsRecompute: Bool {
        if samples.contains(where: { $0.featureVersion != FeatureExtractor.version }) { return true }
        if let enc = encoder {
            if embeddings.contains(where: { $0.encoderVersion != enc.version }) { return true }
            let linked = Set(embeddings.compactMap { $0.recordingID })
            if samples.contains(where: { s in
                guard s.source != .adaptive, let id = s.recordingID else { return false }
                return !linked.contains(id)
            }) { return true }
        }
        return false
    }

    /// Recomputes every sample's features (and embeddings) from the stored raw recordings
    /// with the current calibration, feature extractor and model.
    func recomputeAll() async {
        guard let store = store else { return }
        let ids = Set(samples.compactMap { $0.recordingID } + embeddings.compactMap { $0.recordingID })
            .sorted { $0.uuidString < $1.uuidString }
        var featuresByRecording: [UUID: [Double]] = [:]
        var embeddingByRecording: [UUID: [Double]] = [:]
        for (i, id) in ids.enumerated() {
            if i % 4 == 0 {
                busyStatus = "Recomputing \(i + 1) of \(ids.count)…"
                await Task.yield()
            }
            guard let recording = try? store.readRecording(id: id) else { continue }
            let p = SignalProcessor.process(recording)
            featuresByRecording[id] = FeatureExtractor.extract(p, calibration: calibration)
            if let enc = encoder, let out = try? enc.encode(p) {
                embeddingByRecording[id] = out.embedding
            }
        }
        samples = samples.compactMap { s -> TemplateSample? in
            var s = s
            if let id = s.recordingID, let f = featuresByRecording[id] {
                s.features = f
                s.featureVersion = FeatureExtractor.version
                return s
            }
            return s.featureVersion == FeatureExtractor.version ? s : nil
        }
        if let enc = encoder {
            embeddings = embeddings.compactMap { e -> StoredEmbedding? in
                var e = e
                if let id = e.recordingID, let v = embeddingByRecording[id] {
                    e.sample.embedding = v
                    e.encoderVersion = enc.version
                    return e
                }
                return e.encoderVersion == enc.version ? e : nil
            }
            let linked = Set(embeddings.compactMap { $0.recordingID })
            for s in samples where s.source != .adaptive {
                guard let id = s.recordingID, !linked.contains(id), let v = embeddingByRecording[id] else { continue }
                embeddings.append(StoredEmbedding(
                    recordingID: id, encoderVersion: enc.version,
                    sample: EmbeddingSample(prototype: s.prototype, embedding: v, source: s.source, createdAt: s.createdAt)))
            }
        }
        saveSamples()
        saveEmbeddings()
        rebuild()
        busyStatus = nil
    }

    // MARK: Data management

    func deleteAllData() {
        do {
            try store?.deleteAll()
        } catch {
            errorMessage = "Could not delete all data: \(error.localizedDescription)"
        }
        people = []
        samples = []
        embeddings = []
        calibration = nil
        settings = AppSettings()
        checkLog = []
        lastAnalysis = nil
        lastCheck = nil
        rebuild()
    }

    /// Writes recordings.jsonl, features.csv and checks.csv to a temporary folder.
    func makeExport() throws -> [URL] {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("7oda-export-\(formatter.string(from: Date()))", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let anonymize = settings.anonymizeExports
        let names = Exporter.nameMap(people: people, anonymize: anonymize)
        let recordings = Exporter.prepare(store?.allRecordings() ?? [], names: names)
        let files: [(String, Data)] = [
            ("recordings.jsonl", try RecordingCoding.jsonLines(recordings)),
            ("features.csv", Data(Exporter.featuresCSV(samples: samples, names: names).utf8)),
            ("checks.csv", Data(Exporter.checksCSV(entries: checkLog, names: names).utf8)),
        ]
        var urls: [URL] = []
        for (name, data) in files {
            let url = folder.appendingPathComponent(name)
            try data.write(to: url, options: [.atomic, .completeFileProtection])
            urls.append(url)
        }
        return urls
    }
}
