import Foundation
import XCTest
import AuditorCore

/// Deterministic synthetic recordings: the hand response follows the template shape
/// (SPEC 5.6) delayed by `delay`, carried by a 37 Hz rotating vibration so that the
/// high-passed energy is smooth. The same construction in Python (reference
/// `signal.process`) gives lag = round(100 · delay) and ncc > 0.97 for these seeds.
enum Synthetic {
    static func amplitude(_ spec: ChallengeSpec, _ tau: Double, decay: Double = 0.03) -> Double {
        var s = 0.0
        for event in spec.events {
            switch event {
            case .tap(let tap):
                if tau >= tap.time { s += tap.intensity * exp(-(tau - tap.time) / decay) }
            case .ramp(let ramp):
                let t = ramp.time
                let d = ramp.duration
                if t <= tau && tau < t + d {
                    s += ramp.intensity(at: tau - t)
                } else if tau >= t + d {
                    s += ramp.intensity(at: d) * exp(-(tau - t - d) / decay)
                }
            }
        }
        return s
    }

    static func recording(seed: UInt64, delay: Double, amplitude scale: Double = 0.02, noise: Double = 2e-4,
                          gyroScale: Double = 0.05, purpose: RecordingPurpose = .check) -> Recording {
        let spec = ChallengeGenerator.generate(seed: seed)
        let hapticStart = 500.0
        let start = hapticStart - 0.45
        let count = Int((spec.duration + 0.9) / 0.01)
        var rng = Xoshiro256StarStar(seed: seed ^ 0xABCDEF)
        var accel = SensorStream()
        var gyro = SensorStream()
        var motion = MotionStream()
        for k in 0..<count {
            let s = start + 0.002 + Double(k) * 0.01
            let a = amplitude(spec, s - hapticStart - delay)
            let phase = 2.0 * Double.pi * 37.0 * (s - hapticStart)
            accel.t.append(s)
            gyro.t.append(s)
            motion.t.append(s)
            accel.x.append(scale * a * sin(phase) + noise * rng.uniform(-1, 1))
            accel.y.append(scale * a * cos(phase) + noise * rng.uniform(-1, 1))
            accel.z.append(-1.0 + noise * rng.uniform(-1, 1))
            gyro.x.append(gyroScale * scale * a * cos(phase) + noise * rng.uniform(-1, 1))
            gyro.y.append(gyroScale * scale * a * sin(phase) + noise * rng.uniform(-1, 1))
            gyro.z.append(noise * rng.uniform(-1, 1))
            motion.gravityX.append(0.0)
            motion.gravityY.append(-0.5)
            motion.gravityZ.append(-0.866)
            motion.userAccelX.append(0.0)
            motion.userAccelY.append(0.0)
            motion.userAccelZ.append(0.0)
            motion.rotationX.append(0.0)
            motion.rotationY.append(0.0)
            motion.rotationZ.append(0.0)
            motion.attitudeW.append(1.0)
            motion.attitudeX.append(0.0)
            motion.attitudeY.append(0.0)
            motion.attitudeZ.append(0.0)
        }
        return Recording(createdAt: Date(timeIntervalSince1970: 1_795_514_400), purpose: purpose,
                         challenge: spec, hapticStart: hapticStart, accel: accel, gyro: gyro, motion: motion)
    }
}

final class SignalTests: XCTestCase {
    func testAlignmentRecoversTheDelay() {
        // (seed, delay, n, quietRange, eventRange) from the Python reference.
        let cases: [(UInt64, Double, Int, Range<Int>, Range<Int>)] = [
            (7, 0.05, 184, 0..<37, 39..<164),
            (99, 0.0, 180, 0..<30, 32..<155),
            (1234, 0.12, 207, 0..<40, 42..<194),
            (5, -0.03, 203, 0..<26, 28..<175),
        ]
        for (seed, delay, n, quiet, event) in cases {
            let p = SignalProcessor.process(Synthetic.recording(seed: seed, delay: delay))
            let expectedLag = Int((delay * 100.0).rounded())
            XCTAssertEqual(p.n, n, "n for seed \(seed)")
            XCTAssertEqual(p.alignment.lag, expectedLag, "lag for seed \(seed)")
            XCTAssertGreaterThan(p.alignment.ncc, 0.9, "ncc for seed \(seed)")
            XCTAssertGreaterThan(p.alignment.peakMargin, 0.1, "peakMargin for seed \(seed)")
            XCTAssertEqual(p.quietRange, quiet, "quietRange for seed \(seed)")
            XCTAssertEqual(p.eventRange, event, "eventRange for seed \(seed)")
            XCTAssertEqual(p.gapFraction, 0.0)
            XCTAssertEqual(p.sampleRate, 100.0, accuracy: 1e-6)
            XCTAssertEqual(p.accel.count, 3)
            XCTAssertEqual(p.gyroHP.count, 3)
            XCTAssertEqual(p.envelope.count, p.n)
            XCTAssertEqual(p.template.count, p.n)
            XCTAssertEqual(p.energyAccel.count, p.n)
            XCTAssertEqual(p.gravity[2].count, p.n)
            XCTAssertEqual(p.gridStart, 500.0 - 0.25, accuracy: 1e-12)
            XCTAssertGreaterThan(p.noiseAccel, 0.0)
            XCTAssertEqual(p.index(ofPatternTime: 0.0), 25 + expectedLag)
        }
    }

    func testEmptyStreamsDoNotCrash() {
        let spec = ChallengeGenerator.generate(seed: 3)
        let recording = Recording(createdAt: Date(timeIntervalSince1970: 0), purpose: .check, challenge: spec,
                                  hapticStart: 100.0, accel: .empty, gyro: .empty)
        let p = SignalProcessor.process(recording)
        XCTAssertGreaterThan(p.n, 100)
        XCTAssertEqual(p.gapFraction, 1.0)
        XCTAssertEqual(p.sampleRate, 0.0)
        XCTAssertEqual(p.alignment.lag, 0)
        XCTAssertEqual(p.alignment.ncc, 0.0)
        XCTAssertEqual(p.noiseAccel, SignalConfig.default.accelNoiseFloor)
        XCTAssertTrue(p.envelope.allSatisfy { $0 == 0.0 })
        XCTAssertTrue(p.quietRange.lowerBound >= 0 && p.quietRange.upperBound <= p.n)
        XCTAssertTrue(p.eventRange.lowerBound >= 0 && p.eventRange.upperBound <= p.n)
    }

    func testDegenerateDurationsAndTimestamps() {
        var spec = ChallengeGenerator.generate(seed: 4)
        let single = SensorStream(t: [100.0], x: [0.1], y: [0.2], z: [-1.0])
        for duration in [0.0, -5.0, Double.nan, Double.infinity, 0.03] {
            spec.duration = duration
            let r = Recording(createdAt: Date(timeIntervalSince1970: 0), purpose: .test, challenge: spec,
                              hapticStart: 100.0, accel: single, gyro: single)
            let p = SignalProcessor.process(r)
            XCTAssertGreaterThanOrEqual(p.n, 0)
            XCTAssertEqual(p.envelope.count, p.n)
            XCTAssertTrue(p.quietRange.lowerBound >= 0 && p.quietRange.upperBound <= max(p.n, 0))
            XCTAssertTrue(p.eventRange.lowerBound >= 0 && p.eventRange.upperBound <= max(p.n, 0))
            XCTAssertEqual(p.sampleRate, 0.0)
        }
        let noEvents = ChallengeSpec(seed: 1, duration: 0.5, events: [])
        let r = Recording(createdAt: Date(timeIntervalSince1970: 0), purpose: .test, challenge: noEvents,
                          hapticStart: Double.nan, accel: single, gyro: single)
        let p = SignalProcessor.process(r)
        XCTAssertEqual(p.n, 100)
        XCTAssertTrue(p.template.allSatisfy { $0 == 0.0 })
        XCTAssertEqual(p.alignment.ncc, 0.0)
    }

    func testFilterDesignsUseSpecDefaults() {
        let c = SignalConfig.default
        XCTAssertEqual(c.fs, 100.0)
        XCTAssertEqual(c.tremorCenter, 9.797958971132712)
        XCTAssertEqual(c.tremorQ, 9.797958971132712 / 4.0)
        XCTAssertEqual(c.lagMin, -10)
        XCTAssertEqual(c.lagMax, 25)
        XCTAssertEqual(c.pad, 30)
    }
}
