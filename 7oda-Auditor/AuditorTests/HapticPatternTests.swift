import CoreHaptics
import XCTest
@testable import Auditor
import AuditorCore

final class HapticPatternTests: XCTestCase {
    private func value(_ event: CHHapticEvent, _ id: CHHapticEvent.ParameterID) -> Float? {
        event.eventParameters.first { $0.parameterID == id }?.value
    }

    func testComponentsFollowTheSpecMapping() throws {
        for seed: UInt64 in [1, 1234, 0xDEADBEEF] {
            let spec = ChallengeGenerator.generate(seed: seed)
            let parts = HapticPlayer.components(for: spec)
            XCTAssertEqual(parts.events.count, spec.events.count)
            XCTAssertEqual(parts.curves.count, spec.ramps.count)
            XCTAssertEqual(spec.ramps.count, 2)

            var curveIndex = 0
            for (event, hapticEvent) in zip(spec.events, parts.events) {
                switch event {
                case .tap(let tap):
                    XCTAssertEqual(hapticEvent.type, .hapticTransient)
                    XCTAssertEqual(hapticEvent.relativeTime, tap.time, accuracy: 1e-9)
                    XCTAssertEqual(value(hapticEvent, .hapticIntensity) ?? -1, Float(tap.intensity), accuracy: 1e-6)
                    XCTAssertEqual(value(hapticEvent, .hapticSharpness) ?? -1, Float(tap.sharpness), accuracy: 1e-6)
                case .ramp(let ramp):
                    XCTAssertEqual(hapticEvent.type, .hapticContinuous)
                    XCTAssertEqual(hapticEvent.relativeTime, ramp.time, accuracy: 1e-9)
                    XCTAssertEqual(hapticEvent.duration, ramp.duration, accuracy: 1e-9)
                    XCTAssertEqual(value(hapticEvent, .hapticIntensity) ?? -1, 1.0, accuracy: 1e-6)
                    XCTAssertEqual(value(hapticEvent, .hapticSharpness) ?? -1, Float(ramp.sharpness), accuracy: 1e-6)

                    let curve = parts.curves[curveIndex]
                    curveIndex += 1
                    XCTAssertEqual(curve.parameterID, .hapticIntensityControl)
                    XCTAssertEqual(curve.relativeTime, ramp.time, accuracy: 1e-9)
                    let points = curve.controlPoints
                    XCTAssertEqual(points.count, 4)
                    guard points.count == 4 else { continue }
                    let d = ramp.duration
                    XCTAssertEqual(points[0].relativeTime, 0, accuracy: 1e-9)
                    XCTAssertEqual(points[1].relativeTime, d / 2, accuracy: 1e-9)
                    XCTAssertEqual(points[2].relativeTime, d, accuracy: 1e-9)
                    XCTAssertEqual(points[3].relativeTime, d + 0.01, accuracy: 1e-9)
                    XCTAssertEqual(points[0].value, Float(ramp.curve[0].intensity), accuracy: 1e-6)
                    XCTAssertEqual(points[1].value, Float(ramp.curve[1].intensity), accuracy: 1e-6)
                    XCTAssertEqual(points[2].value, Float(ramp.curve[2].intensity), accuracy: 1e-6)
                    XCTAssertEqual(points[3].value, 1.0, accuracy: 1e-9)
                }
            }
            XCTAssertNoThrow(try HapticPlayer.pattern(for: spec))
        }
    }

    func testFirstEventIsATapAndTimesIncrease() {
        let spec = ChallengeGenerator.generate(seed: 42)
        let events = HapticPlayer.components(for: spec).events
        XCTAssertEqual(events.first?.type, .hapticTransient)
        let times = events.map { $0.relativeTime }
        XCTAssertEqual(times, times.sorted())
    }
}
