import Foundation
import XCTest
import AuditorCore

private struct ChallengeFixtureFile: Decodable {
    struct Case: Decodable {
        let seed: String
        let spec: ChallengeSpec
    }
    let cases: [Case]
}

final class ChallengeTests: XCTestCase {
    /// SPEC 11.2: generated specs equal the reference bit for bit.
    func testChallengeFixtureParity() {
        guard let file = Fixture.decode(ChallengeFixtureFile.self, "challenge_vectors") else { return }
        XCTAssertGreaterThanOrEqual(file.cases.count, 50)
        for c in file.cases {
            guard let seed = SeedFormat.parse(c.seed) else {
                XCTFail("invalid seed \(c.seed)")
                continue
            }
            let generated = ChallengeGenerator.generate(seed: seed)
            XCTAssertEqual(generated.seed, c.spec.seed, "seed \(c.seed)")
            XCTAssertEqual(generated.version, c.spec.version, "version for seed \(c.seed)")
            XCTAssertEqual(generated.duration, c.spec.duration, "duration for seed \(c.seed)")
            XCTAssertEqual(generated.events.count, c.spec.events.count, "event count for seed \(c.seed)")
            for (i, pair) in zip(generated.events, c.spec.events).enumerated() where pair.0 != pair.1 {
                XCTFail("event \(i) for seed \(c.seed): got \(pair.0), expected \(pair.1)")
            }
            XCTAssertEqual(generated, c.spec, "spec for seed \(c.seed)")
        }
    }

    func testStructureInvariants() {
        for seed in UInt64(0)..<300 {
            let spec = ChallengeGenerator.generate(seed: seed &* 0x9E37_79B9_7F4A_7C15)
            let taps = spec.taps
            let ramps = spec.ramps
            XCTAssertTrue((6...8).contains(taps.count))
            XCTAssertEqual(ramps.count, 2)
            XCTAssertEqual(spec.events.count, taps.count + 2)
            guard case .tap = spec.events[0] else {
                XCTFail("first event must be a tap")
                continue
            }
            // Ramps are separated by at least one tap.
            var lastWasRamp = false
            for event in spec.events {
                if case .ramp = event {
                    XCTAssertFalse(lastWasRamp, "two consecutive ramps for seed \(seed)")
                    lastWasRamp = true
                } else {
                    lastWasRamp = false
                }
            }
            // Times strictly increase and the duration covers the last event.
            for i in 1..<spec.events.count {
                XCTAssertGreaterThan(spec.events[i].time, spec.events[i - 1].time)
            }
            XCTAssertGreaterThan(spec.duration, spec.events.last?.time ?? 0)
            XCTAssertTrue(spec.duration > 0.8 && spec.duration < 2.2, "duration \(spec.duration)")
            for ramp in ramps {
                XCTAssertEqual(ramp.curve.count, 3)
                XCTAssertEqual(ramp.curve[0].time, 0.0)
                XCTAssertEqual(ramp.curve[1].time, ramp.duration / 2.0)
                XCTAssertEqual(ramp.curve[2].time, ramp.duration)
            }
        }
    }

    func testCurveInterpolationAndClamping() {
        let ramp = RampEvent(time: 0.5, duration: 0.2, sharpness: 0.5, curve: [
            CurvePoint(time: 0.0, intensity: 0.3),
            CurvePoint(time: 0.1, intensity: 0.7),
            CurvePoint(time: 0.2, intensity: 0.9),
        ])
        XCTAssertEqual(ramp.intensity(at: -1.0), 0.3)
        XCTAssertEqual(ramp.intensity(at: 0.0), 0.3)
        XCTAssertEqual(ramp.intensity(at: 0.05), 0.3 + 0.5 * (0.7 - 0.3), accuracy: 1e-15)
        XCTAssertEqual(ramp.intensity(at: 0.1), 0.7)
        XCTAssertEqual(ramp.intensity(at: 0.2), 0.9)
        XCTAssertEqual(ramp.intensity(at: 5.0), 0.9)
        let empty = RampEvent(time: 0, duration: 0.1, sharpness: 0.5, curve: [])
        XCTAssertEqual(empty.intensity(at: 0.05), 0.0)
    }

    func testJSONShape() throws {
        let spec = ChallengeGenerator.generate(seed: 1234)
        let data = try RecordingCoding.encoder().encode(spec)
        let tree = try JSONDecoder().decode(JSON.self, from: data)
        XCTAssertEqual(try tree.at("seed").string(), "0x00000000000004d2")
        for event in try tree.at("events").array() {
            let kind = try event.at("kind").string()
            if kind == "tap" {
                XCTAssertFalse(event.has("duration"))
                XCTAssertFalse(event.has("curve"))
                XCTAssertTrue(event.has("intensity"))
            } else {
                XCTAssertEqual(kind, "ramp")
                XCTAssertFalse(event.has("intensity"))
                XCTAssertTrue(event.has("curve"))
                XCTAssertTrue(event.has("duration"))
            }
        }
        let decoded = try RecordingCoding.decoder().decode(ChallengeSpec.self, from: data)
        XCTAssertEqual(decoded, spec)
    }

    func testDecodingIgnoresUnknownKeysAndRejectsUnknownKinds() throws {
        let json = """
        {"version": 1, "seed": "0x00000000000004d2", "duration": 1.25, "extra": true,
         "events": [
           {"kind": "tap", "time": 0.071, "intensity": 0.83, "sharpness": 0.41, "note": "x"},
           {"kind": "ramp", "time": 0.52, "duration": 0.19, "sharpness": 0.47,
            "curve": [{"time": 0.0, "intensity": 0.31}, {"time": 0.095, "intensity": 0.66}, {"time": 0.19, "intensity": 0.93}]}
         ]}
        """
        let spec = try JSONDecoder().decode(ChallengeSpec.self, from: Data(json.utf8))
        XCTAssertEqual(spec.seed, 1234)
        XCTAssertEqual(spec.events.count, 2)
        XCTAssertEqual(spec.events[0], .tap(TapEvent(time: 0.071, intensity: 0.83, sharpness: 0.41)))
        XCTAssertEqual(spec.events[1].time, 0.52)
        XCTAssertEqual(spec.ramps.first?.curve.count, 3)

        let bad = """
        {"version": 1, "seed": "0x01", "duration": 1.0, "events": [{"kind": "buzz", "time": 0.1}]}
        """
        XCTAssertThrowsError(try JSONDecoder().decode(ChallengeSpec.self, from: Data(bad.utf8)))
        let badSeed = """
        {"version": 1, "seed": "1234", "duration": 1.0, "events": []}
        """
        XCTAssertThrowsError(try JSONDecoder().decode(ChallengeSpec.self, from: Data(badSeed.utf8)))
    }

    func testConfigDefaults() {
        let c = ChallengeConfig.default
        XCTAssertEqual(c.tapCount, 6...8)
        XCTAssertEqual(c.rampCount, 2)
        XCTAssertEqual(c.tapGap, 0.08...0.15)
        XCTAssertEqual(c.rampGap, 0.08...0.12)
    }
}
