import Foundation
import XCTest
import AuditorCore

final class RecordingTests: XCTestCase {
    static func sampleRecording(label: RecordingLabel? = nil, device: DeviceInfo? = nil) -> Recording {
        let spec = ChallengeGenerator.generate(seed: 42)
        let accel = SensorStream(t: [10.0, 10.01, 10.02], x: [0.0, 0.01, -0.02], y: [0.1, 0.1, 0.1], z: [-1.0, -0.99, -1.01])
        let gyro = SensorStream(t: [10.0, 10.01], x: [0.001, 0.002], y: [0, 0], z: [0.5, -0.25])
        let motion = MotionStream(t: [10.0], gravityX: [0], gravityY: [0], gravityZ: [-1],
                                  userAccelX: [0], userAccelY: [0], userAccelZ: [0],
                                  rotationX: [0], rotationY: [0], rotationZ: [0],
                                  attitudeW: [1], attitudeX: [0], attitudeY: [0], attitudeZ: [0])
        return Recording(
            id: UUID(uuidString: "8E0B1C2D-3E4F-4A5B-8C6D-7E8F9A0B1C2D") ?? UUID(),
            createdAt: Date(timeIntervalSince1970: 1_795_514_400),  // 2026-11-24T10:00:00Z
            purpose: .check, label: label, challenge: spec, hapticStart: 10.005,
            accel: accel, gyro: gyro, motion: motion, device: device)
    }

    func testRoundTripAndDateFormat() throws {
        let label = RecordingLabel(personID: "P1", personName: "Amira", hand: .left, accessory: .watch,
                                   posture: .freshGrip, sessionID: "S1")
        let recording = Self.sampleRecording(label: label, device: DeviceInfo(model: "iPhone16,1", system: "iOS 26.1", app: "0.1.0 (1)"))
        let data = try RecordingCoding.encoder().encode(recording)
        let text = String(decoding: data, as: UTF8.self)
        XCTAssertTrue(text.contains("\"createdAt\":\"2026-11-24T10:00:00Z\""), text.prefix(200).description)
        XCTAssertTrue(text.contains("\"schema\":\"7oda-auditor.recording.v1\""))
        XCTAssertTrue(text.contains("\"seed\":\"0x000000000000002a\""))
        XCTAssertFalse(text.contains("\n"), "no pretty printing")
        // Keys are sorted: "accel" comes before "challenge", which comes before "schema".
        if let a = text.range(of: "\"accel\""), let c = text.range(of: "\"challenge\""), let s = text.range(of: "\"schema\"") {
            XCTAssertTrue(a.lowerBound < c.lowerBound && c.lowerBound < s.lowerBound)
        } else {
            XCTFail("missing keys in \(text.prefix(200))")
        }
        let decoded = try RecordingCoding.decoder().decode(Recording.self, from: data)
        XCTAssertEqual(decoded, recording)
    }

    func testOptionalKeysAndFractionalSeconds() throws {
        let recording = Self.sampleRecording()
        let data = try RecordingCoding.encoder().encode(recording)
        let text = String(decoding: data, as: UTF8.self)
        XCTAssertFalse(text.contains("\"label\""))
        XCTAssertFalse(text.contains("\"device\""))
        let fractional = text.replacingOccurrences(of: "2026-11-24T10:00:00Z", with: "2026-11-24T10:00:00.250Z")
        let decoded = try RecordingCoding.decoder().decode(Recording.self, from: Data(fractional.utf8))
        XCTAssertEqual(decoded.createdAt.timeIntervalSince1970, 1_795_514_400.25, accuracy: 1e-3)
        XCTAssertNil(decoded.label)
        XCTAssertNil(decoded.device)
        let withNull = text.replacingOccurrences(of: "\"purpose\"", with: "\"label\":null,\"device\":null,\"purpose\"")
        let decodedNull = try RecordingCoding.decoder().decode(Recording.self, from: Data(withNull.utf8))
        XCTAssertNil(decodedNull.label)
        XCTAssertEqual(decodedNull, recording)
    }

    func testJSONLines() throws {
        let a = Self.sampleRecording()
        var b = Self.sampleRecording(label: RecordingLabel(personID: "P2", personName: "Basem", hand: .right, sessionID: "S2"))
        b.id = UUID(uuidString: "00000000-0000-4000-8000-000000000001") ?? UUID()
        b.purpose = .enrollment
        let data = try RecordingCoding.jsonLines([a, b])
        let text = String(decoding: data, as: UTF8.self)
        XCTAssertEqual(text.filter { $0 == "\n" }.count, 2)
        XCTAssertTrue(text.hasSuffix("\n"))
        XCTAssertEqual(try RecordingCoding.parseJSONLines(data), [a, b])
        // Blank lines and CRLF line ends are accepted.
        let lines = text.split(separator: "\n").map(String.init)
        let messy = "\n" + lines[0] + "\r\n\r\n  \n" + lines[1]
        XCTAssertEqual(try RecordingCoding.parseJSONLines(Data(messy.utf8)), [a, b])
        XCTAssertEqual(try RecordingCoding.parseJSONLines(Data()), [])
        XCTAssertThrowsError(try RecordingCoding.parseJSONLines(Data("{\"not\": \"a recording\"}\n".utf8)))
    }

    func testNonFiniteValuesRoundTrip() throws {
        var recording = Self.sampleRecording()
        recording.accel.x[1] = Double.nan
        recording.gyro.z[0] = Double.infinity
        let data = try RecordingCoding.encoder().encode(recording)
        let decoded = try RecordingCoding.decoder().decode(Recording.self, from: data)
        XCTAssertTrue(decoded.accel.x[1].isNaN)
        XCTAssertEqual(decoded.gyro.z[0], Double.infinity)
    }

    func testEnumsAndEmptyStreams() throws {
        XCTAssertEqual(Hand.allCases, [.right, .left])
        XCTAssertEqual(Accessory.allCases.map { $0.rawValue }, ["none", "watch", "ring", "bracelet", "other"])
        XCTAssertEqual(Posture.allCases.count, 4)
        XCTAssertEqual(RecordingPurpose(rawValue: "signalCheck"), .signalCheck)
        XCTAssertEqual(SensorStream.empty.t.count, 0)
        XCTAssertEqual(MotionStream.empty.attitudeW.count, 0)
        let partial = try JSONDecoder().decode(MotionStream.self, from: Data("{\"t\": [1.0], \"gravityZ\": [-1.0]}".utf8))
        XCTAssertEqual(partial.t, [1.0])
        XCTAssertEqual(partial.gravityZ, [-1.0])
        XCTAssertEqual(partial.rotationX, [])
    }
}
