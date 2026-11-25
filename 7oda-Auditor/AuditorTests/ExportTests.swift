import XCTest
@testable import Auditor
import AuditorCore

final class ExportTests: XCTestCase {
    func testJSONLinesRoundTripWithAnonymizedNames() throws {
        let alice = Person(id: "A", name: "Alice", createdAt: Date(timeIntervalSince1970: 1_000))
        let bob = Person(id: "B", name: "Bob, Jr.", createdAt: Date(timeIntervalSince1970: 2_000))
        let labelA = RecordingLabel(personID: "A", personName: "Alice", hand: .right, posture: .sitting, sessionID: "s1")
        let labelB = RecordingLabel(personID: "B", personName: "Bob, Jr.", hand: .left, accessory: .watch,
                                    posture: .unspecified, sessionID: "s2")
        let recordings = [
            SyntheticHands.recording(seed: 11, profile: SyntheticHands.profile(person: 0, hand: .right),
                                     purpose: .enrollment, label: labelA),
            SyntheticHands.recording(seed: 12, profile: SyntheticHands.profile(person: 1, hand: .left),
                                     purpose: .accessory, label: labelB),
            SyntheticHands.recording(seed: 13, profile: SyntheticHands.profile(person: 0, hand: .left)),
        ]

        let names = Exporter.nameMap(people: [bob, alice], anonymize: true)
        XCTAssertEqual(names, ["A": "P01", "B": "P02"])
        let prepared = Exporter.prepare(recordings, names: names)
        let data = try RecordingCoding.jsonLines(prepared)
        let lines = String(decoding: data, as: UTF8.self).split(separator: "\n")
        XCTAssertEqual(lines.count, 3)
        XCTAssertFalse(String(decoding: data, as: UTF8.self).contains("Alice"))

        let back = try RecordingCoding.parseJSONLines(data)
        XCTAssertEqual(back.count, 3)
        XCTAssertEqual(back.map { $0.id }, recordings.map { $0.id })
        XCTAssertEqual(back[0].label?.personName, "P01")
        XCTAssertEqual(back[1].label?.personName, "P02")
        XCTAssertEqual(back[1].label?.accessory, .watch)
        XCTAssertNil(back[2].label)
        XCTAssertEqual(back[0].accel, recordings[0].accel)
        XCTAssertEqual(back[1].challenge, recordings[1].challenge)
        XCTAssertEqual(back[2].purpose, .check)

        let plain = Exporter.nameMap(people: [alice, bob], anonymize: false)
        XCTAssertEqual(Exporter.prepare(recordings, names: plain)[1].label?.personName, "Bob, Jr.")
    }

    func testCSVs() throws {
        let key = PrototypeKey(personID: "B", hand: .left, accessory: .ring)
        let sample = TemplateSample(prototype: key, source: .accessory, createdAt: Date(timeIntervalSince1970: 0),
                                    features: [1.5, -2.0, 0.25], featureVersion: FeatureExtractor.version,
                                    recordingID: UUID())
        let names = ["B": "Bob, Jr."]
        let csv = Exporter.featuresCSV(samples: [sample], names: names)
        let rows = csv.split(separator: "\n").map(String.init)
        XCTAssertEqual(rows.count, 2)
        XCTAssertTrue(rows[0].hasPrefix("sample_id,recording_id,person,person_id,hand,accessory,source"))
        XCTAssertTrue(rows[1].contains("\"Bob, Jr.\""))
        XCTAssertTrue(rows[1].contains(",left,ring,accessory,"))

        let decision = Decision(outcome: .identified(personID: "B", hand: .left), patternsUsed: 2,
                                bestPrototype: key, bestDistance: 0.1, marginName: 0.2, marginHand: .infinity,
                                handLogOdds: -1.0, absThreshold: 0.3)
        let entry = CheckLogEntry(id: UUID(), createdAt: Date(timeIntervalSince1970: 0), purpose: .check,
                                  recognizer: .v1, recordingIDs: [UUID(), UUID()], patternsUsed: 2, attempts: 3,
                                  seconds: 4.2, v0: nil, v1: DecisionSummary(decision), truthPersonID: nil,
                                  truthHand: nil, requestedHand: .left, challengePassed: true, qualityFailure: nil)
        let checks = Exporter.checksCSV(entries: [entry], names: names)
        let checkRows = checks.split(separator: "\n").map(String.init)
        XCTAssertEqual(checkRows.count, 2)
        let headerCount = checkRows[0].split(separator: ",", omittingEmptySubsequences: false).count
        XCTAssertEqual(headerCount, 13 + 20)
        XCTAssertTrue(checkRows[1].contains("identified"))
        XCTAssertTrue(checkRows[1].contains(",yes,"))
        XCTAssertEqual(DecisionSummary(decision).marginHand, nil, "infinite margins are stored as empty")
    }
}
