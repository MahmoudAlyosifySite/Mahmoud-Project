import CryptoKit
import XCTest
@testable import Auditor
import AuditorCore

/// Synthetic recordings → processing → calibration → enrollment → both recognizers decide.
final class EndToEndTests: XCTestCase {
    @MainActor
    private func makeModel(_ directory: URL, key: SymmetricKey, runner: FakeRunner) throws -> AppModel {
        let store = try EncryptedStore(directory: directory, keyProvider: StaticKeyProvider(symmetricKey: key))
        let encoder = try EmbeddingEncoder(bundle: .main)
        return AppModel(store: store, encoder: encoder, encoderError: nil, runner: runner, motion: nil)
    }

    @MainActor
    private func enroll(_ model: AppModel, runner: FakeRunner, syntheticPerson: Int, name: String,
                        silentFirst: Bool) async throws -> String {
        let controller = EnrollmentController(kind: .newPerson, model: model)
        let personID = controller.flow.personID
        runner.personIndex[personID] = syntheticPerson
        let named = controller.flow.submitName(name)
        XCTAssertTrue(named)
        for item in controller.flow.checklistItems {
            controller.flow.setChecked(item, true)
        }
        let started = controller.flow.confirmChecklist()
        XCTAssertTrue(started)
        if silentFirst { runner.silentCount = 1 }
        var steps = 0
        while let slot = controller.flow.currentSlot, steps < 80 {
            steps += 1
            controller.flow.confirmReady()
            let recording = try await runner.record(purpose: .enrollment, label: controller.flow.label(for: slot))
            let analysis = model.analyze(recording, mode: .handheld)
            let before = controller.flow.acceptedCount
            controller.apply(analysis, to: slot)
            if analysis.passed {
                if model.encoder != nil {
                    XCTAssertNotNil(analysis.embedding, analysis.encoderError ?? "")
                }
            } else {
                let after = controller.flow.acceptedCount
                let message = controller.flow.lastMessage
                XCTAssertEqual(after, before, "a failed pattern repeats its slot")
                XCTAssertNotNil(message)
            }
        }
        let step = controller.flow.step
        let failures = controller.flow.totalFailures
        let accepted = controller.accepted.count
        XCTAssertEqual(step, .summary)
        XCTAssertEqual(accepted, 30)
        XCTAssertEqual(failures, silentFirst ? 1 : 0, "only the silent pattern may fail")
        controller.commit()
        return personID
    }

    @MainActor
    func testCalibrateEnrollAndCheckWithBothRecognizers() async throws {
        let directory = TestSupport.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let key = SymmetricKey(size: .bits256)
        let runner = FakeRunner()
        let model = try makeModel(directory, key: key, runner: runner)
        XCTAssertTrue(model.needsCalibration)
        XCTAssertEqual(model.effectiveRecognizer, .v1)
        XCTAssertNil(model.recognizerNote)

        // Calibration: 20 patterns flat on a (synthetic) table.
        var progressCalls = 0
        let calibration = try await model.runCalibration { _, _, _ in progressCalls += 1 }
        XCTAssertEqual(progressCalls, Calibration.patternCount)
        let cal = try XCTUnwrap(calibration, "synthetic table patterns should calibrate")
        XCTAssertGreaterThanOrEqual(cal.patternsUsed, Calibration.minimumPassing)
        XCTAssertFalse(model.needsCalibration)

        // Enrollment of two synthetic people (one failed pattern on the first).
        let first = try await enroll(model, runner: runner, syntheticPerson: 0, name: "First", silentFirst: true)
        let second = try await enroll(model, runner: runner, syntheticPerson: 1, name: "Second", silentFirst: false)
        XCTAssertEqual(model.people.count, 2)
        XCTAssertEqual(model.samples.count, 60)
        XCTAssertEqual(model.embeddings.count, 60)
        XCTAssertEqual(model.identifier.prototypes.count, 4)
        XCTAssertEqual(model.embeddingIdentifier.prototypes.count, 4)
        XCTAssertEqual(model.prototypeCounts(for: first).map { $0.v0 }, [15, 15])
        XCTAssertFalse(model.needsRecompute)

        // Check with v1 answering; v0 decides on the same patterns.
        runner.holder = (0, .right)
        var statuses: [String] = []
        let result = try await model.runCheck(purpose: .check) { statuses.append($0) }
        XCTAssertNil(result.qualityFailure)
        XCTAssertFalse(statuses.isEmpty)
        XCTAssertEqual(result.answeredBy, .v1)
        XCTAssertGreaterThanOrEqual(result.patterns.count, 1)
        XCTAssertLessThanOrEqual(result.patterns.count, model.settings.maxPatterns)
        let v1 = try XCTUnwrap(result.v1)
        let v0 = try XCTUnwrap(result.v0)
        XCTAssertNotEqual(v1.outcome, .needMorePatterns)
        XCTAssertNotEqual(v1.outcome, .unknown(.noPeople))
        XCTAssertNotEqual(v0.outcome, .unknown(.noPeople))
        XCTAssertEqual(v1.prototypeDistances.count, 4)
        XCTAssertEqual(v0.prototypeDistances.count, 4)
        XCTAssertEqual(v0.bestPrototype?.personID, first, "v0 should find the synthetic holder")
        XCTAssertEqual(model.checkLog.count, 1)
        XCTAssertEqual(model.checkLog.first?.v0?.patternsUsed, result.patterns.count)
        XCTAssertNotNil(model.lastCheck)

        // Check with v0 answering.
        var settings = model.settings
        settings.recognizer = .v0
        model.updateSettings(settings)
        XCTAssertEqual(model.effectiveRecognizer, .v0)
        runner.holder = (1, .left)
        let result2 = try await model.runCheck(purpose: .check) { _ in }
        XCTAssertEqual(result2.answeredBy, .v0)
        let d2 = try XCTUnwrap(result2.decision)
        XCTAssertNotEqual(d2.outcome, .needMorePatterns)
        XCTAssertEqual(d2.bestPrototype?.personID, second)
        XCTAssertNotNil(result2.v1)

        // Hand challenge and a test run with ground truth.
        let challenged = try await model.runCheck(purpose: .check, requestedHand: .left) { _ in }
        XCTAssertNotNil(challenged.challengePassed)
        let truth = RecordingLabel(personID: second, personName: "Second", hand: .left, sessionID: "T")
        _ = try await model.runCheck(purpose: .test, truth: truth) { _ in }
        XCTAssertEqual(model.testEntries.count, 1)
        XCTAssertEqual(model.confusionMatrix(for: .v0).total, 1)
        XCTAssertEqual(model.confusionMatrix(for: .v1).total, 1)

        // Adaptive learning keeps the sample bookkeeping consistent.
        settings = model.settings
        settings.learnFromConfidentChecks = true
        model.updateSettings(settings)
        let before = model.samples.count + model.embeddings.count
        let learned = try await model.runCheck(purpose: .check) { _ in }
        XCTAssertEqual(model.samples.count + model.embeddings.count, before + learned.learned)

        // Quality failure: four silent patterns exhaust the slot's retries.
        runner.silentCount = 4
        let failed = try await model.runCheck(purpose: .check) { _ in }
        XCTAssertNotNil(failed.qualityFailure)
        XCTAssertNil(failed.decision)
        XCTAssertEqual(failed.attempts, 1 + AppModel.maxRetriesPerSlot)

        // Recompute from raw recordings keeps every sample.
        await model.recomputeAll()
        XCTAssertGreaterThanOrEqual(model.samples.count, 60)
        XCTAssertEqual(model.identifier.prototypes.count, 4)

        // Everything persists (encrypted) and reloads.
        let reloaded = try makeModel(directory, key: key, runner: runner)
        XCTAssertEqual(reloaded.people.map { $0.id }, model.people.map { $0.id })
        XCTAssertEqual(reloaded.samples.count, model.samples.count)
        XCTAssertEqual(reloaded.embeddings.count, model.embeddings.count)
        XCTAssertNotNil(reloaded.calibration)
        XCTAssertEqual(reloaded.settings.recognizer, .v0)
        XCTAssertEqual(reloaded.checkLog.count, model.checkLog.count)

        // Export: recordings decode back; CSVs exist.
        let urls = try model.makeExport()
        XCTAssertEqual(urls.map { $0.lastPathComponent }, ["recordings.jsonl", "features.csv", "checks.csv"])
        let exported = try RecordingCoding.parseJSONLines(Data(contentsOf: urls[0]))
        XCTAssertEqual(exported.count, model.store?.recordingIDs().count)
        XCTAssertGreaterThan(exported.count, 80)
        if let folder = urls.first?.deletingLastPathComponent() {
            try? FileManager.default.removeItem(at: folder)
        }

        // Delete one person, then everything.
        model.deletePerson(second)
        XCTAssertEqual(model.people.count, 1)
        XCTAssertTrue(model.samples.allSatisfy { $0.prototype.personID == first })
        model.deleteAllData()
        XCTAssertTrue(model.people.isEmpty)
        XCTAssertTrue(model.needsCalibration)
        XCTAssertEqual(model.store?.recordingIDs().count, 0)
    }

    @MainActor
    func testFallsBackToV0WithoutTheModel() async throws {
        let directory = TestSupport.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = try EncryptedStore(directory: directory,
                                       keyProvider: StaticKeyProvider(symmetricKey: SymmetricKey(size: .bits256)))
        let runner = FakeRunner()
        let model = AppModel(store: store, encoder: nil, encoderError: "missing", runner: runner, motion: nil)
        XCTAssertEqual(model.settings.recognizer, .v1)
        XCTAssertEqual(model.effectiveRecognizer, .v0)
        XCTAssertNotNil(model.recognizerNote)

        let id = try await enroll(model, runner: runner, syntheticPerson: 0, name: "Solo", silentFirst: false)
        XCTAssertEqual(model.embeddings.count, 0)
        runner.holder = (0, .left)
        let result = try await model.runCheck(purpose: .check) { _ in }
        XCTAssertEqual(result.answeredBy, .v0)
        XCTAssertNil(result.v1)
        XCTAssertFalse(result.notes.isEmpty)
        XCTAssertEqual(result.v0?.bestPrototype?.personID, id)
    }
}
